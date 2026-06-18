#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>

using namespace std;
using namespace cv;
using namespace Eigen;

enum class StitchingMode {
    FAST,
    QUALITY
};

class PanoramaStitcher {
private:
    struct ImageData {
        Mat image;
        Mat color_image;
        Mat weight_map;
        float gain;
        vector<KeyPoint> keypoints;
        Mat descriptors;
    };

    struct NormalizationParams {
        Mat T1, T2;
        float scale1, scale2;
        Point2f centroid1, centroid2;
    };

    float akaze_threshold;
    int akaze_octaves;
    int akaze_octave_layers;
    float lowe_ratio;
    int grid_rows;
    int grid_cols;
    int min_points_for_homography;
    int max_points_for_homography;
    float blend_width;
    int ransac_iterations;
    float ransac_threshold;
    double ransac_confidence;
    StitchingMode mode;

    vector<ImageData> images;
    int ref_index;

    vector<Mat> global_homographies;
    vector<Mat> inverse_global;

    Mat panorama;
    bool is_panorama_computed;

    Ptr<AKAZE> akaze_detector;

    // Статистика
    double time_detection = 0;
    double time_matching_homography = 0;
    double time_stitching = 0;
    vector<int> keypoints_count;
    vector<int> matches_after_ratio;
    vector<float> inlier_ratios;

public:
    PanoramaStitcher(
        float threshold = 0.001f, int octaves = 4, int octave_layers = 4,
        float ratio = 0.75f, int grid_r = 6, int grid_c = 6,
        int min_points = 4, int max_points = 300, float blend = 30.0f,
        int ransac_iters = 2000, float ransac_thresh = 4.0f, double ransac_conf = 0.99,
        StitchingMode m = StitchingMode::FAST
    ) : akaze_threshold(threshold), akaze_octaves(octaves),
        akaze_octave_layers(octave_layers), lowe_ratio(ratio),
        grid_rows(grid_r), grid_cols(grid_c),
        min_points_for_homography(max(4, min_points)),
        max_points_for_homography(max_points), blend_width(blend),
        ransac_iterations(ransac_iters), ransac_threshold(ransac_thresh),
        ransac_confidence(ransac_conf), mode(m),
        is_panorama_computed(false), ref_index(0) {

        akaze_detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3,
            akaze_threshold, akaze_octaves, akaze_octave_layers, KAZE::DIFF_PM_G2);
    }

    void setMode(StitchingMode m) { mode = m; }

    bool addImage(const string& path) {
        Mat img = imread(path, IMREAD_COLOR);
        if (img.empty()) { cerr << "Ошибка: " << path << endl; return false; }

        ImageData d;
        d.color_image = img.clone();
        cvtColor(img, d.image, COLOR_BGR2GRAY);
        d.gain = 1.0f;
        computeWeightMap(d);

        images.push_back(d);
        is_panorama_computed = false;
        cout << "Загружено: " << path << " (" << d.image.cols << "x" << d.image.rows << ")" << endl;
        return true;
    }

    void computeWeightMap(ImageData& img) {
        Mat mask(img.color_image.rows, img.color_image.cols, CV_8U, Scalar(255));
        Mat dist;
        distanceTransform(mask, dist, DIST_L2, DIST_MASK_PRECISE);
        img.weight_map = Mat(dist.rows, dist.cols, CV_32F);
        for (int y = 0; y < dist.rows; y++) {
            float* dst = img.weight_map.ptr<float>(y);
            float* src = dist.ptr<float>(y);
            for (int x = 0; x < dist.cols; x++) {
                dst[x] = min(1.0f, src[x] / blend_width);
            }
        }
    }

    void computeExposureCompensation() {
        cout << "\n--- Компенсация экспозиции ---" << endl;
        int n = (int)images.size();
        if (n < 2) return;

        vector<float> gains(n, 1.0f);
        gains[ref_index] = 1.0f;

        for (int i = ref_index; i < n - 1; i++) {
            float ratio = computeOverlapRatio(i, i + 1);
            if (ratio > 0) gains[i + 1] = gains[i] * ratio;
        }

        for (int i = ref_index; i > 0; i--) {
            float ratio = computeOverlapRatio(i - 1, i);
            if (ratio > 0) gains[i - 1] = gains[i] / ratio;
        }

        for (int i = 0; i < n; i++) {
            images[i].gain = max(0.5f, min(2.0f, gains[i]));
            cout << "  Изображение " << i << " коэффициент усиления: " << images[i].gain << endl;
        }
    }

    float computeOverlapRatio(int idx1, int idx2) {
        Mat H = global_homographies[idx1].inv() * global_homographies[idx2];
        double sum1 = 0, sum2 = 0;
        int count = 0;

        Point2f corners[4] = {
            Point2f(0, 0), Point2f((float)images[idx2].color_image.cols, 0),
            Point2f((float)images[idx2].color_image.cols, (float)images[idx2].color_image.rows),
            Point2f(0, (float)images[idx2].color_image.rows)
        };

        float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
        for (int j = 0; j < 4; j++) {
            Point2f pt = transform(corners[j], H);
            min_x = min(min_x, pt.x); min_y = min(min_y, pt.y);
            max_x = max(max_x, pt.x); max_y = max(max_y, pt.y);
        }

        int x1 = max(0, (int)min_x), y1 = max(0, (int)min_y);
        int x2 = min(images[idx1].color_image.cols - 1, (int)max_x);
        int y2 = min(images[idx1].color_image.rows - 1, (int)max_y);

        if (x1 >= x2 || y1 >= y2) return 0;

        Mat H_inv = H.inv();
        for (int y = y1; y < y2; y += 5) {
            for (int x = x1; x < x2; x += 5) {
                Point2f pt2 = transform(Point2f((float)x, (float)y), H_inv);
                if (pt2.x >= 0 && pt2.x < images[idx2].color_image.cols &&
                    pt2.y >= 0 && pt2.y < images[idx2].color_image.rows) {
                    Vec3b c1 = images[idx1].color_image.at<Vec3b>(y, x);
                    Vec3b c2 = images[idx2].color_image.at<Vec3b>((int)pt2.y, (int)pt2.x);
                    sum1 += (c1[0] + c1[1] + c1[2]) / (3.0 * 255.0);
                    sum2 += (c2[0] + c2[1] + c2[2]) / (3.0 * 255.0);
                    count++;
                }
            }
        }

        if (count < 100) return 0;
        float mean1 = (float)(sum1 / count), mean2 = (float)(sum2 / count);
        if (mean2 < 0.01f) return 0;
        return mean1 / mean2;
    }

    Point2f transform(const Point2f& pt, const Mat& H) {
        Mat p = (Mat_<double>(3, 1) << pt.x, pt.y, 1);
        Mat t = H * p;
        return Point2f((float)(t.at<double>(0) / t.at<double>(2)),
            (float)(t.at<double>(1) / t.at<double>(2)));
    }

    void detectAll() {
        cout << "\n=== Этап 1: Обнаружение ключевых точек на " << images.size() << " изображениях ===" << endl;
        keypoints_count.clear();
        for (size_t i = 0; i < images.size(); i++) {
            cout << "  Изображение " << i << "... " << flush;
            auto start = chrono::high_resolution_clock::now();
            akaze_detector->detectAndCompute(images[i].image, noArray(),
                images[i].keypoints, images[i].descriptors);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            time_detection += elapsed.count();
            keypoints_count.push_back(images[i].keypoints.size());
            cout << images[i].keypoints.size() << " ключевых точек" << endl;
        }
    }

    vector<DMatch> match(const Mat& d1, const Mat& d2) {
        if (d1.empty() || d2.empty()) return {};
        Ptr<BFMatcher> knn = BFMatcher::create(NORM_HAMMING, false);
        vector<vector<DMatch>> km;
        knn->knnMatch(d1, d2, km, 2);
        vector<DMatch> good;
        for (auto& p : km)
            if (p.size() >= 2 && p[0].distance < lowe_ratio * p[1].distance)
                good.push_back(p[0]);
        return good;
    }

    vector<vector<int>> sectors(const vector<KeyPoint>& kp,
        const vector<DMatch>& m, const Size& sz) {
        vector<vector<int>> sec(grid_rows * grid_cols);
        float cw = (float)sz.width / grid_cols, ch = (float)sz.height / grid_rows;
        for (size_t i = 0; i < m.size(); i++) {
            Point2f pt = kp[m[i].queryIdx].pt;
            int c = max(0, min(grid_cols - 1, (int)(pt.x / cw)));
            int r = max(0, min(grid_rows - 1, (int)(pt.y / ch)));
            sec[r * grid_cols + c].push_back((int)i);
        }
        return sec;
    }

    int countInliers(const vector<Point2f>& p1, const vector<Point2f>& p2,
        const Mat& H, float th, vector<uchar>& mask) {
        int n = (int)p1.size(), cnt = 0;
        mask.resize(n, 0);
        for (int i = 0; i < n; i++) {
            if (norm(transform(p1[i], H) - p2[i]) < th) { mask[i] = 1; cnt++; }
        }
        return cnt;
    }

    NormalizationParams normalize(const vector<Point2f>& p1, const vector<Point2f>& p2) {
        NormalizationParams np;
        np.centroid1 = np.centroid2 = Point2f(0, 0);
        for (size_t i = 0; i < p1.size(); i++) { np.centroid1 += p1[i]; np.centroid2 += p2[i]; }
        np.centroid1.x /= p1.size(); np.centroid1.y /= p1.size();
        np.centroid2.x /= p2.size(); np.centroid2.y /= p2.size();
        float d1 = 0, d2 = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            d1 += norm(p1[i] - np.centroid1); d2 += norm(p2[i] - np.centroid2);
        }
        d1 /= p1.size(); d2 /= p2.size();
        np.scale1 = sqrt(2.0f) / max(d1, 1e-6f);
        np.scale2 = sqrt(2.0f) / max(d2, 1e-6f);
        np.T1 = (Mat_<double>(3, 3) << np.scale1, 0, -np.scale1 * np.centroid1.x,
            0, np.scale1, -np.scale1 * np.centroid1.y, 0, 0, 1);
        np.T2 = (Mat_<double>(3, 3) << np.scale2, 0, -np.scale2 * np.centroid2.x,
            0, np.scale2, -np.scale2 * np.centroid2.y, 0, 0, 1);
        return np;
    }

    Mat solveLS(const vector<Point2f>& p1, const vector<Point2f>& p2) {
        int n = (int)p1.size();
        MatrixXd A(2 * n, 9);
        for (int i = 0; i < n; i++) {
            double x1 = p1[i].x, y1 = p1[i].y, x2 = p2[i].x, y2 = p2[i].y;
            A(2 * i, 0) = x1; A(2 * i, 1) = y1; A(2 * i, 2) = 1;
            A(2 * i, 3) = 0; A(2 * i, 4) = 0; A(2 * i, 5) = 0;
            A(2 * i, 6) = -x2 * x1; A(2 * i, 7) = -x2 * y1; A(2 * i, 8) = -x2;
            A(2 * i + 1, 0) = 0; A(2 * i + 1, 1) = 0; A(2 * i + 1, 2) = 0;
            A(2 * i + 1, 3) = x1; A(2 * i + 1, 4) = y1; A(2 * i + 1, 5) = 1;
            A(2 * i + 1, 6) = -y2 * x1; A(2 * i + 1, 7) = -y2 * y1; A(2 * i + 1, 8) = -y2;
        }
        JacobiSVD<MatrixXd> svd(A, ComputeFullV);
        VectorXd h = svd.matrixV().col(8);
        return (Mat_<double>(3, 3) << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8));
    }

    Mat refine(const vector<Point2f>& p1, const vector<Point2f>& p2, const vector<uchar>& mask) {
        vector<Point2f> in1, in2;
        for (size_t i = 0; i < mask.size(); i++)
            if (mask[i]) { in1.push_back(p1[i]); in2.push_back(p2[i]); }
        if (in1.size() < 4) return Mat();
        NormalizationParams np = normalize(in1, in2);
        vector<Point2f> np1, np2;
        for (size_t i = 0; i < in1.size(); i++) {
            Mat a = (Mat_<double>(3, 1) << in1[i].x, in1[i].y, 1);
            Mat b = (Mat_<double>(3, 1) << in2[i].x, in2[i].y, 1);
            Mat na = np.T1 * a, nb = np.T2 * b;
            np1.push_back(Point2f((float)(na.at<double>(0) / na.at<double>(2)),
                (float)(na.at<double>(1) / na.at<double>(2))));
            np2.push_back(Point2f((float)(nb.at<double>(0) / nb.at<double>(2)),
                (float)(nb.at<double>(1) / nb.at<double>(2))));
        }
        return np.T2.inv() * solveLS(np1, np2) * np.T1;
    }

    Mat ransac(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
        const vector<DMatch>& matches, const Size& sz, int& inl_cnt) {
        int n = (int)matches.size();
        vector<Point2f> ap1(n), ap2(n);
        for (int i = 0; i < n; i++) {
            ap1[i] = kp1[matches[i].queryIdx].pt;
            ap2[i] = kp2[matches[i].trainIdx].pt;
        }
        auto sec = sectors(kp1, matches, sz);
        vector<int> nonempty;
        for (int i = 0; i < (int)sec.size(); i++)
            if (!sec[i].empty()) nonempty.push_back(i);
        if (nonempty.size() < 4) {
            Mat mask;
            Mat H = findHomography(ap1, ap2, RANSAC, ransac_threshold, mask);
            inl_cnt = countNonZero(mask);
            return H;
        }
        mt19937 rng(random_device{}());
        Mat bestH; vector<uchar> bestMask; int bestCnt = 0;
        for (int iter = 0; iter < min(ransac_iterations, 5000); iter++) {
            vector<int> avail = nonempty, chosen;
            for (int j = 0; j < 4 && !avail.empty(); j++) {
                uniform_int_distribution<int> d(0, (int)avail.size() - 1);
                int idx = d(rng); chosen.push_back(avail[idx]);
                avail.erase(avail.begin() + idx);
            }
            if (chosen.size() < 4) continue;
            vector<Point2f> sp1, sp2;
            for (int s : chosen) {
                auto& cell = sec[s];
                int mi = cell[uniform_int_distribution<int>(0, (int)cell.size() - 1)(rng)];
                sp1.push_back(ap1[mi]); sp2.push_back(ap2[mi]);
            }
            Mat H = findHomography(sp1, sp2, 0);
            if (H.empty()) continue;
            vector<uchar> mask; int cnt = countInliers(ap1, ap2, H, ransac_threshold, mask);
            if (cnt > bestCnt) {
                bestCnt = cnt; bestH = H.clone(); bestMask = mask;
                if (bestCnt > n * ransac_confidence) break;
            }
        }
        inl_cnt = bestCnt;
        if (bestCnt >= 4) {
            Mat ref = refine(ap1, ap2, bestMask);
            if (!ref.empty()) return ref;
        }
        return bestH;
    }

    Vec3b bilinear(const Mat& img, float x, float y) {
        int x0 = (int)floor(x), y0 = (int)floor(y), x1 = x0 + 1, y1 = y0 + 1;
        float dx = x - x0, dy = y - y0;
        x0 = max(0, min(x0, img.cols - 1)); x1 = max(0, min(x1, img.cols - 1));
        y0 = max(0, min(y0, img.rows - 1)); y1 = max(0, min(y1, img.rows - 1));
        Vec3b p00 = img.at<Vec3b>(y0, x0), p10 = img.at<Vec3b>(y0, x1);
        Vec3b p01 = img.at<Vec3b>(y1, x0), p11 = img.at<Vec3b>(y1, x1);
        Vec3b res;
        for (int c = 0; c < 3; c++) {
            float v = (1 - dx) * (1 - dy) * p00[c] + dx * (1 - dy) * p10[c] + (1 - dx) * dy * p01[c] + dx * dy * p11[c];
            res[c] = (uchar)max(0.0f, min(255.0f, v));
        }
        return res;
    }

    float bilinearWeight(const Mat& wm, float x, float y) {
        int x0 = (int)floor(x), y0 = (int)floor(y), x1 = x0 + 1, y1 = y0 + 1;
        float dx = x - x0, dy = y - y0;
        x0 = max(0, min(x0, wm.cols - 1)); x1 = max(0, min(x1, wm.cols - 1));
        y0 = max(0, min(y0, wm.rows - 1)); y1 = max(0, min(y1, wm.rows - 1));
        float p00 = wm.at<float>(y0, x0), p10 = wm.at<float>(y0, x1);
        float p01 = wm.at<float>(y1, x0), p11 = wm.at<float>(y1, x1);
        return (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p10 + (1 - dx) * dy * p01 + dx * dy * p11;
    }

    Mat stitchFast() {
        cout << "\n=== БЫСТРЫЙ РЕЖИМ ===" << endl;
        int n = (int)images.size();
        if (n < 2) return Mat();

        detectAll();

        int ref = n / 2;
        ref_index = ref;
        cout << "Опорное изображение: " << ref << endl;

        vector<Mat> local_forward(n - 1);
        matches_after_ratio.clear();
        inlier_ratios.clear();

        auto matching_start = chrono::high_resolution_clock::now();

        for (int i = 0; i < n - 1; i++) {
            auto matches = match(images[i].descriptors, images[i + 1].descriptors);
            matches_after_ratio.push_back((int)matches.size());
            int maxm = min(max_points_for_homography * 5, (int)matches.size());
            vector<DMatch> sel(matches.begin(), matches.begin() + maxm);
            int inl;
            local_forward[i] = ransac(images[i].keypoints, images[i + 1].keypoints, sel, images[i].image.size(), inl);
            float inlier_ratio = (float)inl / matches.size() * 100.0f;
            inlier_ratios.push_back(inlier_ratio);
            cout << "  Пара " << i << "->" << i + 1 << ": " << inl << " inliers (доля: " << inlier_ratio << "%)" << endl;
        }

        auto matching_end = chrono::high_resolution_clock::now();
        chrono::duration<double> matching_elapsed = matching_end - matching_start;
        time_matching_homography += matching_elapsed.count();

        global_homographies.resize(n);
        inverse_global.resize(n);
        Mat I = Mat::eye(3, 3, CV_64F);
        global_homographies[ref] = I.clone();
        inverse_global[ref] = I.clone();

        for (int i = ref + 1; i < n; i++) {
            global_homographies[i] = global_homographies[i - 1] * local_forward[i - 1].inv();
            inverse_global[i] = global_homographies[i].inv();
        }
        for (int i = ref - 1; i >= 0; i--) {
            global_homographies[i] = global_homographies[i + 1] * local_forward[i];
            inverse_global[i] = global_homographies[i].inv();
        }

        computeExposureCompensation();

        float min_x = 0, min_y = 0, max_x = images[ref].color_image.cols, max_y = images[ref].color_image.rows;
        for (int i = 0; i < n; i++) {
            Point2f c[4] = { {0,0}, {(float)images[i].color_image.cols,0},
                {(float)images[i].color_image.cols,(float)images[i].color_image.rows},
                {0,(float)images[i].color_image.rows} };
            for (int j = 0; j < 4; j++) {
                Point2f pt = transform(c[j], global_homographies[i]);
                min_x = min(min_x, pt.x); min_y = min(min_y, pt.y);
                max_x = max(max_x, pt.x); max_y = max(max_y, pt.y);
            }
        }

        int off_x = (int)floor(max(0.0f, -min_x)), off_y = (int)floor(max(0.0f, -min_y));
        int pw = (int)ceil(max_x - min_x), ph = (int)ceil(max_y - min_y);

        cout << "Размер панорамы: " << pw << "x" << ph << endl;

        Mat T = (Mat_<double>(3, 3) << 1, 0, off_x, 0, 1, off_y, 0, 0, 1);

        auto stitch_start = chrono::high_resolution_clock::now();

        Mat result;
        warpPerspective(images[0].color_image, result, T * global_homographies[0], Size(pw, ph));

        for (int i = 1; i < n; i++) {
            Mat warped;
            warpPerspective(images[i].color_image, warped, T * global_homographies[i], Size(pw, ph));
            Mat gray, mask;
            cvtColor(warped, gray, COLOR_BGR2GRAY);
            mask = gray > 0;
            warped.copyTo(result, mask);
        }

        auto stitch_end = chrono::high_resolution_clock::now();
        chrono::duration<double> stitch_elapsed = stitch_end - stitch_start;
        time_stitching += stitch_elapsed.count();

        return result;
    }



    Mat stitchQuality() {
        cout << "\n=== КАЧЕСТВЕННЫЙ РЕЖИМ ===" << endl;
        int n = (int)images.size();
        if (n < 2) return Mat();

        detectAll();

        int ref = n / 2;
        ref_index = ref;
        cout << "Опорное изображение " << ref << endl;

        vector<Mat> local_forward(n - 1);
        for (int i = 0; i < n - 1; i++) {
            auto matches = match(images[i].descriptors, images[i + 1].descriptors);
            int maxm = min(max_points_for_homography * 5, (int)matches.size());
            vector<DMatch> sel(matches.begin(), matches.begin() + maxm);
            int inl;
            local_forward[i] = ransac(images[i].keypoints, images[i + 1].keypoints, sel, images[i].image.size(), inl);
            cout << "  Пары " << i << "->" << i + 1 << ": " << inl << " inliers" << endl;
        }

        global_homographies.resize(n);
        inverse_global.resize(n);
        Mat I = Mat::eye(3, 3, CV_64F);
        global_homographies[ref] = I.clone();
        inverse_global[ref] = I.clone();
        for (int i = ref + 1; i < n; i++) {
            global_homographies[i] = global_homographies[i - 1] * local_forward[i - 1].inv();
            inverse_global[i] = global_homographies[i].inv();
        }
        for (int i = ref - 1; i >= 0; i--) {
            global_homographies[i] = global_homographies[i + 1] * local_forward[i];
            inverse_global[i] = global_homographies[i].inv();
        }

        computeExposureCompensation();


        float min_x = 0, min_y = 0, max_x = images[ref].color_image.cols, max_y = images[ref].color_image.rows;
        for (int i = 0; i < n; i++) {
            Point2f c[4] = { {0,0}, {(float)images[i].color_image.cols,0},
                {(float)images[i].color_image.cols,(float)images[i].color_image.rows},
                {0,(float)images[i].color_image.rows} };
            for (int j = 0; j < 4; j++) {
                Point2f pt = transform(c[j], global_homographies[i]);
                min_x = min(min_x, pt.x); min_y = min(min_y, pt.y);
                max_x = max(max_x, pt.x); max_y = max(max_y, pt.y);
            }
        }

        int off_x = (int)floor(max(0.0f, -min_x)), off_y = (int)floor(max(0.0f, -min_y));
        int pw = (int)ceil(max_x - min_x), ph = (int)ceil(max_y - min_y);
        cout << "Панорама: " << pw << "x" << ph << endl;


        Mat accum_r = Mat::zeros(ph, pw, CV_32F);
        Mat accum_g = Mat::zeros(ph, pw, CV_32F);
        Mat accum_b = Mat::zeros(ph, pw, CV_32F);
        Mat accum_w = Mat::zeros(ph, pw, CV_32F);

        for (int i = 0; i < n; i++) {
            for (int y = 0; y < ph; y++) {
                float* r_row = accum_r.ptr<float>(y);
                float* g_row = accum_g.ptr<float>(y);
                float* b_row = accum_b.ptr<float>(y);
                float* w_row = accum_w.ptr<float>(y);

                for (int x = 0; x < pw; x++) {
                    Point2f pt_ref((float)(x - off_x), (float)(y - off_y));
                    Point2f pt_img = transform(pt_ref, inverse_global[i]);

                    if (pt_img.x >= 0 && pt_img.x < images[i].color_image.cols - 1 &&
                        pt_img.y >= 0 && pt_img.y < images[i].color_image.rows - 1) {


                        float dist_left = pt_img.x;
                        float dist_right = images[i].color_image.cols - 1.0f - pt_img.x;
                        float dist_top = pt_img.y;
                        float dist_bottom = images[i].color_image.rows - 1.0f - pt_img.y;
                        float min_dist = min({ dist_left, dist_right, dist_top, dist_bottom });
                        float weight = min(1.0f, min_dist / blend_width);


                        Vec3b color = bilinear(images[i].color_image, pt_img.x, pt_img.y);
                        float gain = images[i].gain;


                        r_row[x] += color[2] * weight * gain;
                        g_row[x] += color[1] * weight * gain;
                        b_row[x] += color[0] * weight * gain;
                        w_row[x] += weight;
                    }
                }
            }

        }


        Mat result(ph, pw, CV_8UC3);
        for (int y = 0; y < ph; y++) {
            Vec3b* row = result.ptr<Vec3b>(y);
            float* r_row = accum_r.ptr<float>(y);
            float* g_row = accum_g.ptr<float>(y);
            float* b_row = accum_b.ptr<float>(y);
            float* w_row = accum_w.ptr<float>(y);

            for (int x = 0; x < pw; x++) {
                if (w_row[x] > 0) {
                    row[x] = Vec3b(
                        (uchar)min(255.0f, b_row[x] / w_row[x]),
                        (uchar)min(255.0f, g_row[x] / w_row[x]),
                        (uchar)min(255.0f, r_row[x] / w_row[x]));
                }
            }
        }

        cout << "Панорама создана" << endl;
        return result;
    }



    Mat stitch() {
        if (images.size() < 2) {
            cerr << "Необходимо как минимум 2 изображения" << endl;
            return Mat();
        }

        time_detection = 0;
        time_matching_homography = 0;
        time_stitching = 0;
        keypoints_count.clear();
        matches_after_ratio.clear();
        inlier_ratios.clear();

        auto total_start = chrono::high_resolution_clock::now();

        cout << "\n========== СШИВКА " << images.size() << " ИЗОБРАЖЕНИЙ ==========" << endl;
        cout << "Режим: " << (mode == StitchingMode::FAST ? "БЫСТРЫЙ" : "КАЧЕСТВЕННЫЙ") << endl;

        if (mode == StitchingMode::FAST) {
            panorama = stitchFast();
        }
        else {
            panorama = stitchQuality();
        }

        auto total_end = chrono::high_resolution_clock::now();
        chrono::duration<double> total_elapsed = total_end - total_start;


        cout << "\n========== СТАТИСТИКА ==========" << endl;
        cout << "Время детектирования: " << time_detection << " сек" << endl;
        cout << "Время сопоставления и создания гомографий: " << time_matching_homography << " сек" << endl;
        cout << "Время сшивания: " << time_stitching << " сек" << endl;
        cout << "Общее время: " << total_elapsed.count() << " сек" << endl;


        if (!keypoints_count.empty()) {
            double avg_keypoints = accumulate(keypoints_count.begin(), keypoints_count.end(), 0.0) / keypoints_count.size();
            cout << "\nСреднее количество ключевых точек: " << avg_keypoints << endl;
        }

        if (!matches_after_ratio.empty()) {
            double avg_matches = accumulate(matches_after_ratio.begin(), matches_after_ratio.end(), 0.0) / matches_after_ratio.size();
            cout << "Среднее соответствий после ratio test: " << avg_matches << endl;
        }

        if (!inlier_ratios.empty()) {
            double avg_inlier_ratio = accumulate(inlier_ratios.begin(), inlier_ratios.end(), 0.0) / inlier_ratios.size();
            cout << "Средняя доля inliers: " << avg_inlier_ratio << "%" << endl;
        }

        cout << "================================" << endl;

        is_panorama_computed = true;
        return panorama;
    }

    bool savePanorama(const string& filename, int quality = 95) {
        if (panorama.empty()) { cerr << "Нет панорамы для сохранения" << endl; return false; }
        vector<int> params = { IMWRITE_JPEG_QUALITY, quality };
        bool ok = imwrite(filename, panorama, params);
        if (ok) cout << "Сохранено: " << filename << " (" << panorama.cols << "x" << panorama.rows << ")" << endl;
        return ok;
    }

    Mat getPanorama() const { return panorama; }
    int getImageCount() const { return (int)images.size(); }
};

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");

    if (argc < 3) {
        cout << "Использование: " << argv[0] << " <изображения...>\n"
            << "  Пример: " << argv[0] << " img1.jpg img2.jpg img3.jpg\n";
        system("pause");
        return -1;
    }

    PanoramaStitcher stitcher(
        0.002f,     // threshold 
        3,          // octaves 
        3,          // octave_layers
        0.8f,       // lowe_ratio
        4, 4,       // grid 
        4, 200,     // min/max points
        20.0f,      // blend_width
        300,        // ransac_iterations 
        5.0f,       // ransac_threshold
        0.95,       // ransac_confidence 
        StitchingMode::FAST
    );


    for (int i = 1; i < argc; i++) {
        stitcher.addImage(argv[i]);
    }

    if (stitcher.getImageCount() < 2) {
        cout << "Необходимо как минимум 2 изображения" << endl;
        return -1;
    }

    Mat panorama = stitcher.stitch();

    if (!panorama.empty()) {
        stitcher.savePanorama("panorama_result.jpg", 95);
        namedWindow("Панорама", WINDOW_NORMAL);
        resizeWindow("Панорама", 1200, 800);
        imshow("Панорама", panorama);
        waitKey(0);
    }

    return 0;
}
