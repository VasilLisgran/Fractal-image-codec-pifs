import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from Codec import compress_color, decompress_color
from SaveAndLoad import save_pifs, load_pifs

IMG_SIZE = (512, 512)
PREVIEW_SIZE = (180, 180)

QUALITY_PRESETS = {
    "Высокое (медленнее)":     (0.0002, 0.0004),
    "Среднее":                  (0.001,  0.002),
    "Быстрое (меньше файл)":    (0.003,  0.005),
}
DEFAULT_QUALITY = "Высокое (медленнее)"

SCALE_PRESETS = {
    "1x (Оригинальное)": (1.0, 20),
    "2x (Увеличенное 2x)": (2.0, 25),
    "4x (Увеличенное 4x)": (4.0, 30),
}
DEFAULT_SCALE = "1x (Оригинальное)"

IMAGE_FILETYPES = [
    ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"),
    ("Все файлы", "*.*"),
]
PIFS_FILETYPES = [
    ("PIFS файлы", "*.pifs"),
    ("Все файлы", "*.*"),
]
