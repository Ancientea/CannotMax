import csv
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from ..config.paths import DATA_DIR, IMAGES_DIR, PROJECT_ROOT


class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Data Viewer")
        self.root.geometry("1000x700")

        self.BG_COLOR = "#333333"
        self.FG_COLOR = "#ffffff"
        self.root.configure(bg=self.BG_COLOR)

        # 绑定快捷键
        self.root.bind("<Left>", lambda event: self.show_prev_row())  # 小键盘左键
        self.root.bind("<Right>", lambda event: self.show_next_row())  # 小键盘右键
        self.root.bind("<Delete>", lambda event: self.delete_current_row())  # 删除键

        # 创建顶部和底部框架
        self.top_frame = tk.Frame(root, bg=self.BG_COLOR)
        self.top_frame.pack(pady=10)
        self.bottom_frame = tk.Frame(root, bg=self.BG_COLOR)
        self.bottom_frame.pack(pady=10)

        # 创建按钮
        btn_style = {
            "bg": "#555555",
            "fg": self.FG_COLOR,
            "activebackground": "#777777",
            "activeforeground": self.FG_COLOR,
        }
        self.next_button = tk.Button(
            root, text="下一个", command=self.show_next_row, **btn_style
        )
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.prev_button = tk.Button(
            root, text="上一个", command=self.show_prev_row, **btn_style
        )
        self.prev_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.delete_button = tk.Button(
            root, text="删除数据", command=self.delete_current_row, **btn_style
        )
        self.delete_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # 添加行号显示和跳转功能
        self.row_label = tk.Label(
            root, text="当前行号: 0", bg=self.BG_COLOR, fg=self.FG_COLOR
        )
        self.row_label.pack(side=tk.LEFT, padx=10)

        self.row_entry = tk.Entry(
            root,
            width=5,
            bg="#555555",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
        )
        self.row_entry.pack(side=tk.LEFT, padx=5)

        self.jump_button = tk.Button(
            root, text="跳转", command=self.jump_to_row, **btn_style
        )
        self.jump_button.pack(side=tk.LEFT, padx=5)

        # 初始化数据
        self.data = self.read_csv(DATA_DIR / "arknights.csv")
        self.current_row_index = 0

        # 加载图片
        self.images = self.load_all_images()

        # 显示第一行数据
        self.show_row(self.current_row_index)

    def read_csv(self, file_path):
        """读取 CSV 文件"""
        data = []
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            self.MONSTER_COUNT = sum(1 for col in header if col.endswith("L"))
            for row in reader:
                data.append(row)
        return data

    def load_all_images(self):
        """加载所有图片"""
        images = {}
        monster_csv_path = PROJECT_ROOT / "monster_greenvine.csv"

        # 单次读取，建立 id -> {primary, alt} 映射
        id_to_names = {}
        try:
            with open(monster_csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = int(row["id"])
                    id_to_names[rid] = {"primary": row["原始名称"], "alt": row["名称"]}
        except Exception as e:
            print(f"Failed to read monster CSV: {e}")

        for i in range(1, self.MONSTER_COUNT + 1):
            names = id_to_names.get(i, {"primary": str(i), "alt": None})

            # 依次尝试主名称和备用名称
            image = None
            for name in [names["primary"], names["alt"]]:
                if name is None:
                    break
                image_path = IMAGES_DIR / f"{name}.png"
                if image_path.exists():
                    image = ImageTk.PhotoImage(Image.open(image_path).resize((80, 80)))
                    break

            images[str(i)] = image
            if image is None:
                print(f"Image for {i} not found.")

        return images

    def show_row(self, row_index):
        """显示指定行的非0列数据"""
        for widget in self.top_frame.winfo_children():
            widget.destroy()  # 清空顶部框架内容

        if row_index >= len(self.data):
            return

        row = self.data[row_index]

        # 显示左方怪物
        for i in range(1, self.MONSTER_COUNT + 1):
            value = row[i - 1]
            try:
                value = float(value)
                if value > 0:  # 仅显示非0值
                    if self.images.get(str(i)):
                        tk.Label(
                            self.top_frame, image=self.images[str(i)], bg=self.BG_COLOR
                        ).grid(row=0, column=i - 1, padx=2)
                    else:
                        tk.Label(
                            self.top_frame,
                            text=f"ID:{i}",
                            font=("Arial", 12),
                            bg=self.BG_COLOR,
                            fg=self.FG_COLOR,
                        ).grid(row=0, column=i - 1, padx=2)
                    tk.Label(
                        self.top_frame,
                        text=str(int(value)),
                        font=("Arial", 16, "bold"),
                        bg=self.BG_COLOR,
                        fg=self.FG_COLOR,
                    ).grid(row=1, column=i - 1)
            except ValueError:
                print(f"Skipping invalid value: {row[i - 1]} at column {i - 1}")

        # 插入空白间隔
        gap_column = self.MONSTER_COUNT - 1  # 间隔列索引
        tk.Label(self.top_frame, text="", bg=self.BG_COLOR).grid(
            row=0, column=gap_column, padx=50
        )  # 添加水平间距

        # 显示右方怪物
        for i in range(self.MONSTER_COUNT + 1, self.MONSTER_COUNT * 2 + 1):
            value = row[i - 1]
            try:
                value = float(value)
                if value > 0:  # 仅显示非0值
                    img_idx = str(i - self.MONSTER_COUNT)
                    if self.images.get(img_idx):
                        tk.Label(
                            self.top_frame, image=self.images[img_idx], bg=self.BG_COLOR
                        ).grid(row=0, column=i - 1, padx=2)
                    else:
                        tk.Label(
                            self.top_frame,
                            text=f"ID:{img_idx}",
                            font=("Arial", 12),
                            bg=self.BG_COLOR,
                            fg=self.FG_COLOR,
                        ).grid(row=0, column=i - 1, padx=2)
                    tk.Label(
                        self.top_frame,
                        text=str(int(value)),
                        font=("Arial", 16, "bold"),
                        bg=self.BG_COLOR,
                        fg=self.FG_COLOR,
                    ).grid(row=1, column=i - 1)
            except ValueError:
                print(f"Skipping invalid value: {row[i - 1]} at column {i - 1}")

        # 清空底部框架内容
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

        # 获取图片路径
        base_dir = Path(__file__).parent.parent.parent.parent  # 项目根目录
        img_name = row[-1]

        orig_path = IMAGES_DIR / f"{img_name}.jpg"
        res_path = IMAGES_DIR / f"{img_name}_result.jpg"

        if not orig_path.exists():
            orig_path = base_dir / "images" / f"{img_name}.png"
            if not orig_path.exists():
                orig_path = base_dir / f"{img_name}.jpg"
                if not orig_path.exists():
                    orig_path = None

        if not res_path.exists():
            res_path = base_dir / "images" / f"{img_name}_result.png"
            if not res_path.exists():
                res_path = None

        self.bottom_frame.images = []  # 防止图片被垃圾回收

        # 1. 裁剪并显示原图（上方）
        if orig_path:
            orig_image = Image.open(orig_path)
            w, h = orig_image.size
            # 左右裁掉1/5（保留中间3/5），保留下方1/5（即裁掉上方4/5）
            left = w * 0.2
            top = h * 0.8
            right = w * 0.8
            bottom = h
            cropped_image = orig_image.crop(
                (int(left), int(top), int(right), int(bottom))
            )

            # 如果需要限制大小，可解除下面注释
            # max_size = (800, 200)
            # cropped_image.thumbnail(max_size, Image.Resampling.LANCZOS)

            cropped_image_tk = ImageTk.PhotoImage(cropped_image)
            tk.Label(self.bottom_frame, image=cropped_image_tk, bg=self.BG_COLOR).pack(
                side=tk.TOP, pady=5
            )
            self.bottom_frame.images.append(cropped_image_tk)
        else:
            tk.Label(
                self.bottom_frame,
                text=f"找不到原图: {img_name}",
                bg=self.BG_COLOR,
                fg=self.FG_COLOR,
            ).pack(side=tk.TOP, pady=5)

        # 2. 显示 L/R 文字和色条（中间）
        result_text = row[self.MONSTER_COUNT * 2].strip()
        tk.Label(
            self.bottom_frame,
            text=result_text,
            font=("Arial", 20, "bold"),
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
        ).pack(side=tk.TOP, pady=2)

        # 添加色彩长条 Canvas
        bar_width = 200
        bar_height = 15
        color_bar = tk.Canvas(
            self.bottom_frame,
            width=bar_width,
            height=bar_height,
            highlightthickness=0,
            bg=self.BG_COLOR,
        )
        color_bar.pack(side=tk.TOP, pady=5)

        if result_text.upper() == "L":
            color_bar.create_rectangle(
                0, 0, bar_width // 2, bar_height, fill="yellow", outline=""
            )
            color_bar.create_rectangle(
                bar_width // 2, 0, bar_width, bar_height, fill="gray", outline=""
            )
        elif result_text.upper() == "R":
            color_bar.create_rectangle(
                0, 0, bar_width // 2, bar_height, fill="gray", outline=""
            )
            color_bar.create_rectangle(
                bar_width // 2, 0, bar_width, bar_height, fill="yellow", outline=""
            )
        else:
            # 默认颜色（异常数据时）
            color_bar.create_rectangle(
                0, 0, bar_width, bar_height, fill="gray", outline=""
            )

        # 3. 显示结果图（下方）
        if res_path:
            res_image = Image.open(res_path)
            max_size = (800, 400)
            res_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            res_image_tk = ImageTk.PhotoImage(res_image)
            tk.Label(self.bottom_frame, image=res_image_tk, bg=self.BG_COLOR).pack(
                side=tk.TOP, pady=5
            )
            self.bottom_frame.images.append(res_image_tk)
        else:
            tk.Label(
                self.bottom_frame,
                text="找不到结果图",
                bg=self.BG_COLOR,
                fg=self.FG_COLOR,
            ).pack(side=tk.TOP, pady=5)

        # 更新行号显示
        self.row_label.config(text=f"当前行号: {row_index + 1}")

    def jump_to_row(self):
        """跳转到指定行"""
        try:
            row_index = int(self.row_entry.get()) - 1  # 转换为索引
            if 0 <= row_index < len(self.data):
                self.current_row_index = row_index
                self.show_row(self.current_row_index)
            else:
                print("行号超出范围")
        except ValueError:
            print("请输入有效的行号")

    def show_prev_row(self):
        """显示上一行数据"""
        if self.current_row_index > 0:
            self.current_row_index -= 1
        else:
            self.current_row_index = len(self.data) - 1  # 跳转到最后一行
        self.show_row(self.current_row_index)

    def show_next_row(self):
        """显示下一行数据"""
        if self.current_row_index < len(self.data) - 1:
            self.current_row_index += 1
        else:
            self.current_row_index = 0  # 跳转到第一行
        self.show_row(self.current_row_index)

    def delete_current_row(self):
        """删除当前行数据"""
        if self.current_row_index < len(self.data):
            # 获取当前行最后一列的图片路径
            image_name = self.data[self.current_row_index][-1]
            base_dir = Path(__file__).resolve().parent

            # 删除相关的图片文件（如果存在）
            possible_exts = [".jpg", "_result.jpg", ".png", ""]
            for ext in possible_exts:
                p = base_dir / "images" / f"{image_name}{ext}"
                if p.exists():
                    p.unlink()
                    print(f"已删除图片文件: {p}")

            del self.data[self.current_row_index]  # 从内存中删除当前行
            # 将修改后的数据写回 CSV 文件
            with open("arknights.csv", "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [f"{i}L" for i in range(1, self.MONSTER_COUNT + 1)]
                    + [f"{i}R" for i in range(1, self.MONSTER_COUNT + 1)]
                    + ["Result", "ImgPath"]
                )
                writer.writerows(self.data)
            # 更新显示
            if self.current_row_index >= len(self.data):
                self.current_row_index -= 1  # 防止越界
            self.show_row(self.current_row_index)


# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = ArknightsApp(root)
    root.mainloop()
