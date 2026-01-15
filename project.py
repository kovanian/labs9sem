import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Цифровая обработка изображений")
        self.root.geometry("1400x950")

        # Переменные состояния
        self.original_image_pil = None
        self.original_image_arr = None
        self.processed_image_arr = None
        self.is_grayscale = False

        # --- Настройка интерфейса ---

        # Левая панель управления (Scrollable canvas)
        control_frame = tk.Frame(root, width=320, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        control_frame.pack_propagate(False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()