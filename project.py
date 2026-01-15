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

        # Центральная часть (Отображение изображений)
        image_frame = tk.Frame(root)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Холст для изображений
        self.canvas_frame = tk.Frame(image_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Левый лейбл (Оригинал)
        self.lbl_original = tk.Label(self.canvas_frame, text="Оригинал")
        self.lbl_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Правый лейбл (Результат)
        self.lbl_processed = tk.Label(self.canvas_frame, text="Обработка")
        self.lbl_processed.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Нижняя часть (Гистограммы)
        self.hist_frame = tk.Frame(image_frame, height=300)
        self.hist_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # --- УТИЛИТЫ ОБРАБОТКИ ---

    def to_grayscale_manual(self, img_arr):
        if len(img_arr.shape) == 2:
            return img_arr
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)

    def calculate_histogram_data(self, img_arr):
        if len(img_arr.shape) == 3:
            work_img = self.to_grayscale_manual(img_arr)
        else:
            work_img = img_arr
        flat = work_img.flatten()
        counts = np.bincount(flat, minlength=256)
        return counts

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

    def draw_histograms(self):
        if self.original_image_arr is None: return
        for widget in self.hist_frame.winfo_children(): widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), dpi=80)

        hist_data_orig = self.calculate_histogram_data(self.original_image_arr)
        ax1.bar(range(256), hist_data_orig, color='gray', width=1.0)
        ax1.set_title("Гистограмма оригинала")
        ax1.set_xlim([0, 255])

        hist_data_proc = self.calculate_histogram_data(self.processed_image_arr)
        ax2.bar(range(256), hist_data_proc, color='blue', width=1.0)
        ax2.set_title("Гистограмма результата")
        ax2.set_xlim([0, 255])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def reset_image(self):
        if self.original_image_arr is not None:
            self.processed_image_arr = self.original_image_arr.copy()
            self.display_image(self.processed_image_arr, self.lbl_processed)
            self.draw_histograms()




if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()