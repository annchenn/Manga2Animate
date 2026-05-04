import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageClipper:
    def __init__(self, root):
        self.root = root
        self.root.title("不規則圖片切分工具 (滾輪縮放 / 右鍵拖曳 / 左鍵連點)")
        
        # 狀態變數
        self.image_path = None
        self.original_cv_img = None
        self.display_img = None
        self.points = []  # 存儲選取的點 (原始坐標)
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # UI 佈局
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_open = tk.Button(self.toolbar, text="開啟圖片", command=self.load_image)
        btn_open.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_save = tk.Button(self.toolbar, text="存取選取區", command=self.save_crop)
        btn_save.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_clear = tk.Button(self.toolbar, text="清除路徑", command=self.clear_points)
        btn_clear.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas = tk.Canvas(root, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 綁定事件
        self.canvas.bind("<Button-1>", self.add_point)        # 左鍵：選點
        self.canvas.bind("<Button-3>", self.start_pan)       # 右鍵：開始拖移
        self.canvas.bind("<B3-Motion>", self.panning)        # 右鍵拖曳：移動
        self.canvas.bind("<MouseWheel>", self.zoom)          # 滾輪：縮放 (Windows)
        self.canvas.bind("<Button-4>", self.zoom)            # 滾輪：縮放 (Linux 上滾)
        self.canvas.bind("<Button-5>", self.zoom)            # 滾輪：縮放 (Linux 下滾)
        
        self.pan_start_x = 0
        self.pan_start_y = 0

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.webp")])
        if not file_path:
            return
        
        self.image_path = file_path
        # 讀取圖片 (支援中文路徑)
        self.original_cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        # 如果是 3 通道則轉為 4 通道 (RGBA) 方便處理透明
        # 確保圖片是 4 通道 (BGRA)
        if len(self.original_cv_img.shape) == 2:
            # 灰階 → BGRA
            self.original_cv_img = cv2.cvtColor(self.original_cv_img, cv2.COLOR_GRAY2BGRA)

        elif self.original_cv_img.shape[2] == 3:
            # BGR → BGRA
            self.original_cv_img = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2BGRA)

        elif self.original_cv_img.shape[2] == 4:
            # 已經是 BGRA，不用動
            pass
            
        self.points = []
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.redraw()

    def redraw(self):
        if self.original_cv_img is None:
            return

        # 計算縮放後的尺寸
        h, w = self.original_cv_img.shape[:2]
        new_w, new_h = int(w * self.zoom_scale), int(h * self.zoom_scale)
        
        # 縮放影像用於顯示
        resized = cv2.resize(self.original_cv_img, (new_w, new_h))
        self.display_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)))
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.display_img)
        
        # 繪製已選取的多邊形線段
        if len(self.points) > 0:
            scaled_points = []
            for p in self.points:
                sx = p[0] * self.zoom_scale + self.offset_x
                sy = p[1] * self.zoom_scale + self.offset_y
                scaled_points.append((sx, sy))
                # 繪製小圓點
                self.canvas.create_oval(sx-3, sy-3, sx+3, sy+3, fill="red", outline="white")
            
            if len(scaled_points) > 1:
                self.canvas.create_polyline(scaled_points, fill="", outline="cyan", width=2)
                # 連接最後一點到第一點
                self.canvas.create_line(scaled_points[-1], scaled_points[0], fill="yellow", dash=(4, 4))

    def add_point(self, event):
        if self.original_cv_img is None: return
        # 將畫布坐標轉回原始圖片坐標
        raw_x = (event.x - self.offset_x) / self.zoom_scale
        raw_y = (event.y - self.offset_y) / self.zoom_scale
        self.points.append((int(raw_x), int(raw_y)))
        self.redraw()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def panning(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.redraw()

    def zoom(self, event):
        if self.original_cv_img is None: return
        # 處理不同平台的縮放值
        if event.num == 4 or event.delta > 0:
            scale_factor = 1.1
        else:
            scale_factor = 0.9
            
        self.zoom_scale *= scale_factor
        self.redraw()

    def clear_points(self):
        self.points = []
        self.redraw()

    def save_crop(self):
        if len(self.points) < 3:
            messagebox.showwarning("提示", "請至少選取三個點來組成區域。")
            return
        
        # 1. 建立遮罩 (Mask)
        h, w = self.original_cv_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # 2. 提取不規則區域
        res = self.original_cv_img.copy()
        # 將遮罩外的地方設為透明 (Alpha=0)
        res[mask == 0] = [0, 0, 0, 0]
        
        # 3. 計算最小邊界矩形 (Bounding Box)
        x, y, w_box, h_box = cv2.boundingRect(pts)
        cropped_res = res[y:y+h_box, x:x+w_box]
        
        # 4. 儲存
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            cv2.imencode('.png', cropped_res)[1].tofile(save_path)
            messagebox.showinfo("成功", f"圖片已存至: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = ImageClipper(root)
    root.mainloop()