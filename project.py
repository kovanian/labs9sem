import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЦОСИИ - Лаба 1")
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

        tk.Frame(control_frame, height=15, bg="#f0f0f0").pack()  # Отступ

        # 1. Пороговая обработка (Бинаризация)
        tk.Label(control_frame, text="1. Порог (Бинаризация)", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(
            anchor="w")
        thresh_frame = tk.Frame(control_frame, bg="#f0f0f0")
        thresh_frame.pack(fill=tk.X, pady=2)
        tk.Label(thresh_frame, text="Порог (0-255):", bg="#f0f0f0").pack(side=tk.LEFT)
        self.threshold_entry = tk.Entry(thresh_frame, width=8)
        self.threshold_entry.insert(0, "128")
        self.threshold_entry.pack(side=tk.RIGHT)
        tk.Button(control_frame, text="Применить порог", command=self.apply_threshold_action).pack(fill=tk.X, pady=2)

        tk.Frame(control_frame, height=15, bg="#f0f0f0").pack()

        # 2. Яркостный срез
        tk.Label(control_frame, text="2. Яркостный срез", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w")

        slice_min_frame = tk.Frame(control_frame, bg="#f0f0f0")
        slice_min_frame.pack(fill=tk.X, pady=1)
        tk.Label(slice_min_frame, text="Мин. граница:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.slice_min_entry = tk.Entry(slice_min_frame, width=8)
        self.slice_min_entry.insert(0, "100")
        self.slice_min_entry.pack(side=tk.RIGHT)

        slice_max_frame = tk.Frame(control_frame, bg="#f0f0f0")
        slice_max_frame.pack(fill=tk.X, pady=1)
        tk.Label(slice_max_frame, text="Макс. граница:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.slice_max_entry = tk.Entry(slice_max_frame, width=8)
        self.slice_max_entry.insert(0, "200")
        self.slice_max_entry.pack(side=tk.RIGHT)

        tk.Button(control_frame, text="Применить срез", command=self.apply_slice_action).pack(fill=tk.X, pady=2)

        tk.Frame(control_frame, height=15, bg="#f0f0f0").pack()

        # 4. Фильтрация (Оператор Превитта)
        tk.Label(control_frame, text="4. Оператор Превитта", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w")
        tk.Button(control_frame, text="Выполнить фильтрацию", command=self.apply_prewitt_action).pack(fill=tk.X, pady=2)
        tk.Frame(control_frame, height=15, bg="#f0f0f0").pack()

        # 3. Гамма-коррекция (НОВОЕ)
        tk.Label(control_frame, text="3. Гамма-коррекция", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w")
        tk.Label(control_frame, text="s = c * r^γ", font=("Arial", 8, "italic"), bg="#f0f0f0").pack(anchor="w")

        gamma_c_frame = tk.Frame(control_frame, bg="#f0f0f0")
        gamma_c_frame.pack(fill=tk.X, pady=1)
        tk.Label(gamma_c_frame, text="Параметр c:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.gamma_c_entry = tk.Entry(gamma_c_frame, width=8)
        self.gamma_c_entry.insert(0, "1.0")
        self.gamma_c_entry.pack(side=tk.RIGHT)

        gamma_g_frame = tk.Frame(control_frame, bg="#f0f0f0")
        gamma_g_frame.pack(fill=tk.X, pady=1)
        tk.Label(gamma_g_frame, text="Гамма (γ):", bg="#f0f0f0").pack(side=tk.LEFT)
        self.gamma_val_entry = tk.Entry(gamma_g_frame, width=8)
        self.gamma_val_entry.insert(0, "2.2")
        self.gamma_val_entry.pack(side=tk.RIGHT)

        tk.Button(control_frame, text="Применить гамму", command=self.apply_gamma_action).pack(fill=tk.X, pady=2)

        tk.Frame(control_frame, height=20, bg="#f0f0f0").pack()
        tk.Button(control_frame, text="Сбросить все", command=self.reset_image, bg="#ffcccb").pack(fill=tk.X)

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

    # --- АЛГОРИТМЫ ---

    def threshold_algorithm(self, img_arr, threshold):
        gray = self.to_grayscale_manual(img_arr)
        result = np.zeros_like(gray)
        result[gray >= threshold] = 255
        result[gray < threshold] = 0
        return result

    def intensity_slice_algorithm(self, img_arr, min_v, max_v):
        gray = self.to_grayscale_manual(img_arr)
        result = np.zeros_like(gray)
        mask = (gray >= min_v) & (gray <= max_v)
        result[mask] = 255
        return result

    def prewitt_operator(self, img_arr):
        gray = self.to_grayscale_manual(img_arr).astype(np.float32)
        h, w = gray.shape
        out_img = np.zeros((h, w), dtype=np.float32)

        top_left = gray[:-2, :-2]
        top_mid = gray[:-2, 1:-1]
        top_right = gray[:-2, 2:]
        mid_left = gray[1:-1, :-2]
        mid_right = gray[1:-1, 2:]
        bot_left = gray[2:, :-2]
        bot_mid = gray[2:, 1:-1]
        bot_right = gray[2:, 2:]

        gx = (top_right + mid_right + bot_right) - (top_left + mid_left + bot_left)
        gy = (bot_left + bot_mid + bot_right) - (top_left + top_mid + top_right)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        if np.max(magnitude) != 0:
            magnitude = (magnitude / np.max(magnitude)) * 255.0

        out_img[1:-1, 1:-1] = magnitude
        return out_img.astype(np.uint8)

    def gamma_correction_algorithm(self, img_arr, c, gamma):
        """
        Гамма-коррекция: s = c * r^gamma
        """
        # Нормализация 0..255 -> 0..1
        img_float = img_arr.astype(float) / 255.0

        # Вычисление
        result = c * np.power(img_float, gamma)

        # Возврат к 0..255
        result = result * 255.0
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    # --- GUI ЛОГИКА ---

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

    # --- ВАЛИДАЦИЯ ВВОДА ---

    def _get_int_from_entry(self, entry_widget, field_name):
        try:
            val = int(entry_widget.get())
            if val < 0 or val > 255: raise ValueError
            return val
        except ValueError:
            messagebox.showerror("Ошибка", f"'{field_name}' должно быть целым числом (0-255).")
            return None

    def _get_float_from_entry(self, entry_widget, field_name):
        try:
            val = float(entry_widget.get())
            if val < 0: raise ValueError
            return val
        except ValueError:
            messagebox.showerror("Ошибка", f"'{field_name}' должно быть положительным числом.")
            return None

    # --- ДЕЙСТВИЯ КНОПОК ---

    def apply_threshold_action(self):
        if self.original_image_arr is None: return
        t = self._get_int_from_entry(self.threshold_entry, "Порог")
        if t is None: return
        self.processed_image_arr = self.threshold_algorithm(self.original_image_arr, t)
        self.display_image(self.processed_image_arr, self.lbl_processed)
        self.draw_histograms()

    def apply_slice_action(self):
        if self.original_image_arr is None: return
        min_v = self._get_int_from_entry(self.slice_min_entry, "Мин. граница")
        max_v = self._get_int_from_entry(self.slice_max_entry, "Макс. граница")
        if min_v is None or max_v is None: return
        if min_v > max_v:
            messagebox.showwarning("Внимание", "Мин. граница не может быть больше Макс.")
            return
        self.processed_image_arr = self.intensity_slice_algorithm(self.original_image_arr, min_v, max_v)
        self.display_image(self.processed_image_arr, self.lbl_processed)
        self.draw_histograms()

    def apply_prewitt_action(self):
        if self.original_image_arr is None: return
        self.processed_image_arr = self.prewitt_operator(self.original_image_arr)
        self.display_image(self.processed_image_arr, self.lbl_processed)
        self.draw_histograms()

    def apply_gamma_action(self):
        if self.original_image_arr is None: return
        c = self._get_float_from_entry(self.gamma_c_entry, "Параметр c")
        gamma = self._get_float_from_entry(self.gamma_val_entry, "Гамма")
        if c is None or gamma is None: return

        self.processed_image_arr = self.gamma_correction_algorithm(self.original_image_arr, c, gamma)
        self.display_image(self.processed_image_arr, self.lbl_processed)
        self.draw_histograms()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()