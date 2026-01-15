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

        # Кнопки ввода/вывода
        tk.Label(control_frame, text="Файл", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(0, 5))
        tk.Button(control_frame, text="Загрузить изображение", command=self.load_image, width=25).pack(pady=2)
        tk.Button(control_frame, text="Сохранить результат", command=self.save_image, width=25).pack(pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if not file_path:
            return
        try:
            img = Image.open(file_path)
            self.original_image_arr = np.array(img)
            self.processed_image_arr = self.original_image_arr.copy()
            self.display_image(self.original_image_arr, self.lbl_original)
            self.display_image(self.processed_image_arr, self.lbl_processed)
            self.draw_histograms()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    def display_image(self, img_arr, label_widget):
        h, w = img_arr.shape[:2]
        max_size = 500
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        img = Image.fromarray(img_arr)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def save_image(self):
        if self.processed_image_arr is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if file_path:
            Image.fromarray(self.processed_image_arr).save(file_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()