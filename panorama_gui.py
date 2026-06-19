import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QComboBox,
    QGroupBox, QProgressBar, QTextEdit, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QFont

class StitcherWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(str, bool)

    def __init__(self, exe_path, mode, image_paths, output_path):
        super().__init__()
        self.exe_path = exe_path
        self.mode = mode
        self.image_paths = image_paths
        self.output_path = output_path

    def run(self):
        try:
            self.status.emit(f"Запуск сшивки в режиме {self.mode.upper()}...")
            
            process = QProcess()
            process.setProcessChannelMode(QProcess.MergedChannels)
            process.start(self.exe_path, [self.mode] + self.image_paths + [self.output_path])
            process.waitForFinished(-1)
            
            if process.exitCode() == 0 and os.path.exists(self.output_path):
                self.finished.emit(self.output_path, True)
            else:
                self.finished.emit("", False)
                
        except Exception as e:
            self.status.emit(f"Ошибка: {str(e)}")
            self.finished.emit("", False)

class PanoramaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.exe_path = ""
        self.output_path = ""
        self.worker = None
        self.init_ui()
        self.find_exe()
        
    def find_exe(self):
        """Поиск EXE файла"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_paths = [
            os.path.join(script_dir, "x64", "Release", "panoram.exe"),
            os.path.join(script_dir, "Release", "panoram.exe"),
            os.path.join(script_dir, "panoram", "x64", "Release", "panoram.exe"),
        ]
        
        for exe_path in possible_paths:
            if os.path.exists(exe_path):
                self.exe_path = exe_path
                self.exe_label.setText(f"{os.path.basename(exe_path)}")
                self.exe_label.setStyleSheet("color: green; font-weight: bold;")
                self.log(f"Найден EXE: {exe_path}")
                self.update_stitch_button()
                return
        
        self.exe_path = ""
        self.exe_label.setText("panoram.exe не найден")
        self.exe_label.setStyleSheet("color: red; font-weight: bold;")
        self.log("EXE не найден! Соберите проект в Visual Studio.")
        
    def init_ui(self):
        self.setWindowTitle("Сшивка панорам")
        self.setGeometry(300, 200, 500, 450)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        # EXE
        row = QHBoxLayout()
        row.addWidget(QLabel("EXE:"))
        self.exe_label = QLabel("Поиск...")
        row.addWidget(self.exe_label)
        row.addStretch()
        layout.addLayout(row)
        
        # Изображения
        group = QGroupBox("Изображения")
        g_layout = QVBoxLayout()
        
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Добавить")
        load_btn.clicked.connect(self.load_images)
        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.clear_images)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(clear_btn)
        g_layout.addLayout(btn_row)
        
        self.images_list = QListWidget()
        self.images_list.setMaximumHeight(120)
        g_layout.addWidget(self.images_list)
        
        group.setLayout(g_layout)
        layout.addWidget(group)
        
        # Режим
        row = QHBoxLayout()
        row.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Быстрый", "Качественный"])
        row.addWidget(self.mode_combo)
        row.addStretch()
        layout.addLayout(row)
        
        # Сохранение
        group = QGroupBox("Сохранение")
        g_layout = QVBoxLayout()
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Папка:"))
        self.dir_label = QLabel("(не выбрана)")
        row.addWidget(self.dir_label)
        dir_btn = QPushButton("Выбрать")
        dir_btn.clicked.connect(self.select_directory)
        row.addWidget(dir_btn)
        g_layout.addLayout(row)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Имя файла:"))
        self.save_edit = QLineEdit("panorama_result.jpg")
        self.save_edit.textChanged.connect(self.update_output_path)
        row.addWidget(self.save_edit)
        g_layout.addLayout(row)
        
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #666; font-size: 9px;")
        g_layout.addWidget(self.path_label)
        
        group.setLayout(g_layout)
        layout.addWidget(group)
        
        # Кнопка сшивки
        self.stitch_btn = QPushButton("СШИТЬ ПАНОРАМУ")
        self.stitch_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.stitch_btn.clicked.connect(self.start_stitching)
        self.stitch_btn.setEnabled(False)
        layout.addWidget(self.stitch_btn)
        
        # Прогресс
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Лог
        group = QGroupBox("Лог")
        g_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Courier New", 9))
        g_layout.addWidget(self.log_text)
        group.setLayout(g_layout)
        layout.addWidget(group)
        
        self.log("Приложение запущено")
        
    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.ensureCursorVisible()
        
    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*.*)"
        )
        if files:
            for f in files:
                if f not in self.image_paths:
                    self.image_paths.append(f)
                    self.images_list.addItem(os.path.basename(f))
            self.log(f"Загружено {len(files)} изображений")
            self.update_stitch_button()
            
    def clear_images(self):
        self.image_paths.clear()
        self.images_list.clear()
        self.update_stitch_button()
        
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения"
        )
        if dir_path:
            self.output_dir = dir_path
            self.dir_label.setText(dir_path)
            self.update_output_path()
            
    def update_output_path(self):
        if hasattr(self, 'output_dir') and self.output_dir and self.save_edit.text():
            self.output_path = os.path.join(self.output_dir, self.save_edit.text())
            if not self.output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.output_path += '.jpg'
            self.path_label.setText(f"{self.output_path}")
        else:
            self.output_path = ""
            self.path_label.setText("")
        self.update_stitch_button()
        
    def update_stitch_button(self):
        enabled = (
            os.path.exists(self.exe_path) and
            len(self.image_paths) >= 2 and
            bool(self.output_path)
        )
        self.stitch_btn.setEnabled(enabled)
        
    def start_stitching(self):
        if not self.output_path:
            return
            
        mode = "fast" if self.mode_combo.currentIndex() == 0 else "quality"
        
        self.worker = StitcherWorker(
            self.exe_path, mode, self.image_paths, self.output_path
        )
        self.worker.status.connect(self.log)
        self.worker.finished.connect(self.stitching_finished)
        
        self.stitch_btn.setEnabled(False)
        self.stitch_btn.setText("СШИВАНИЕ...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
        self.log(f"Сшивка в режиме {mode.upper()}")
        self.worker.start()
        
    def stitching_finished(self, output_path, success):
        self.progress.setVisible(False)
        self.stitch_btn.setEnabled(True)
        self.stitch_btn.setText("СШИТЬ ПАНОРАМУ")
        
        if success:
            self.log(f"Панорама сохранена: {output_path}")
            QMessageBox.information(self, "Готово", f"Панорама сохранена:\n{output_path}")
        else:
            self.log("Ошибка создания панорамы")
            QMessageBox.warning(self, "Ошибка", "Не удалось создать панораму")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PanoramaGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()