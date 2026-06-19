# Panorama Stitcher

Приложение для автоматической сшивки изображений в панораму с графическим интерфейсом на PyQt5 и высокопроизводительным бэкендом на C++.

## Описание

Panorama Stitcher позволяет объединять несколько перекрывающихся изображений в единую панораму. Программа использует алгоритмы компьютерного зрения для обнаружения ключевых точек, сопоставления изображений и их бесшовного объединения.



## Установка и запуск
```bash
git clone https://github.com/yourusername/panoram.git
cd panoram
git checkout feature/python-gui
pip install -r requirements.txt
```

Сборка C++ части (если EXE отсутствует)
Откройте panoram.sln в Visual Studio

Выберите конфигурацию Release и платформу x64

Соберите проект (Ctrl+Shift+B)

EXE файл появится в папке Release/panoram.exe

python panorama_gui.py