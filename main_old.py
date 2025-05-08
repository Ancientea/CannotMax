import logging
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import math
from PIL import Image, ImageTk
from predict import CannotModel
import loadData
import recognize
from train import UnitAwareTransformer
from recognize import MONSTER_COUNT, intelligent_workers_debug
from sklearn.metrics.pairwise import cosine_similarity
from similar_history_match import HistoryMatch
from auto_fetch import AutoFetch

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",)
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

loadData.connect()

class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Neural Network")
        self.history_match = HistoryMatch()

        self.main_panel = tk.Frame(self.root)
        self.main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 鼠标滚轮滚动历史面板
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        # 运行
        self.no_region = True
        self.first_recognize = True
        # 用户选项
        self.is_invest = tk.BooleanVar(value=False)  # 添加投资状态变量
        self.game_mode = tk.StringVar(value="单人")  # 添加游戏模式变量，默认单人模式
        self.device_serial = tk.StringVar(value=loadData.manual_serial)  # 添加设备序列号变量

        # 数据缓存
        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}
        self.progress_var = tk.StringVar()

        self.load_images()
        self.create_widgets()

        # 怪物识别模块
        self.recognizer = recognize.RecognizeMonster()

        # 历史对局面板
        self.history_visible = False
        self.history_container = tk.Frame(self.root, bd=1, relief="sunken")

        # Canvas & Scrollbars
        self.history_canvas = tk.Canvas(self.history_container, bg="white")
        self.history_vscroll = tk.Scrollbar(
            self.history_container, orient="vertical", command=self.history_canvas.yview
        )
        self.history_hscroll = tk.Scrollbar(
            self.history_container, orient="horizontal", command=self.history_canvas.xview
        )

        self.history_canvas.configure(
            yscrollcommand=self.history_vscroll.set, xscrollcommand=self.history_hscroll.set
        )

        # 真正放内容的 Frame
        self.history_frame = tk.Frame(self.history_canvas)
        self.history_canvas.create_window((0, 0), window=self.history_frame, anchor="nw")

        # 更新 scroll region
        self.history_frame.bind(
            "<Configure>",
            lambda e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")),
        )

        # Canvas + 两条滚动条在 history_container 里排版
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_vscroll.grid(row=0, column=1, sticky="ns")
        self.history_hscroll.grid(row=1, column=0, sticky="ew")

        # 让 Canvas 单元格可伸缩
        self.history_container.grid_rowconfigure(0, weight=1)
        self.history_container.grid_columnconfigure(0, weight=1)

        # 模型相关属性
        self.cannot_model = CannotModel()

    def _on_mousewheel(self, event):
        """滑动鼠标滚轮 → 垂直滚动错题本面板"""
        self.history_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        """按住 Shift + 滚轮 → 水平滚动错题本面板"""
        self.history_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_images(self):
        # 获取系统缩放因子
        scaling_factor = self.root.tk.call("tk", "scaling")
        base_size = 30
        icon_size = int(base_size * scaling_factor)  # 动态计算图标大小

        for i in range(1, MONSTER_COUNT + 1):
            # 使用PIL打开图像并缩放
            img = Image.open(f"images/{i}.png")
            width, height = img.size

            # 计算缩放比例，保持宽高比且不超过目标尺寸
            ratio = min(icon_size / width, icon_size / height)
            new_size = (int(width * ratio), int(height * ratio))

            # 高质量缩放
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为Tkinter兼容格式
            photo_img = ImageTk.PhotoImage(img_resized)
            self.images[str(i)] = photo_img

    def create_widgets(self):
        # 创建顶层容器
        self.top_container = tk.Frame(self.main_panel)
        self.bottom_container = tk.Frame(self.main_panel)

        # 顶部容器布局（填充整个水平空间）
        self.top_container.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.bottom_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        # 创建居中容器用于放置左右怪物框
        self.monster_center = tk.Frame(self.top_container)
        self.monster_center.pack(side=tk.TOP, anchor="center")

        # 创建左右怪物容器（添加边框和背景色）
        self.left_frame = tk.Frame(
            self.monster_center, borderwidth=2, relief="groove", padx=5, pady=5
        )
        self.right_frame = tk.Frame(
            self.monster_center, borderwidth=2, relief="groove", padx=5, pady=5
        )

        # 添加左右标题
        tk.Label(self.left_frame, text="左侧怪物", font=("Helvetica", 10, "bold")).grid(
            row=0, columnspan=10
        )
        tk.Label(self.right_frame, text="右侧怪物", font=("Helvetica", 10, "bold")).grid(
            row=0, columnspan=10
        )

        # 左右布局（添加显式间距并居中）
        self.left_frame.pack(side=tk.LEFT, padx=10, anchor="n", pady=5)
        self.right_frame.pack(side=tk.RIGHT, padx=10, anchor="n", pady=5)

        for side, frame, monsters in [
            ("left", self.left_frame, self.left_monsters),
            ("right", self.right_frame, self.right_monsters),
        ]:
            row_n = 6
            monsters_per_row = math.ceil(MONSTER_COUNT / row_n)
            for row in range(row_n):
                start = row * monsters_per_row + 1
                end = min((row + 1) * monsters_per_row + 1, MONSTER_COUNT + 1)
                for i in range(start, end):
                    # 图片标签（缩小尺寸）
                    tk.Label(frame, image=self.images[str(i)], padx=1, pady=1).grid(
                        row=row * 2, column=i - start, sticky="ew"
                    )
                    # 输入框（保持宽度5）
                    monsters[str(i)] = tk.Entry(frame, width=5)
                    monsters[str(i)].grid(
                        row=row * 2 + 1, column=i - start, pady=(0, 1)  # 减小底部间距
                    )

            # 调整列权重使布局更紧凑
            for col in range(monsters_per_row):
                frame.grid_columnconfigure(col, weight=1, minsize=25)  # 适当调整最小列宽

        # 结果显示区域（增加边框）
        self.result_frame = tk.Frame(self.bottom_container, relief="ridge", borderwidth=1)
        self.result_frame.pack(fill=tk.X, pady=5)

        # 使用更醒目的字体
        self.result_label = tk.Label(
            self.result_frame, text="Prediction: ", font=("Helvetica", 16, "bold"), fg="blue"
        )
        self.result_label.pack(pady=3)
        self.stats_label = tk.Label(self.result_frame, text="", font=("Helvetica", 12), fg="green")
        self.stats_label.pack(pady=3)

        # 按钮区域容器（增加边框和背景）
        self.button_frame = tk.Frame(
            self.bottom_container, relief="groove", borderwidth=2, padx=10, pady=10
        )
        self.button_frame.pack(fill=tk.BOTH, expand=True)

        # 按钮布局（分左右两列布局）
        left_buttons = tk.Frame(self.button_frame)
        center_buttons = tk.Frame(self.button_frame)  # 新增中间按钮容器
        right_buttons = tk.Frame(self.button_frame)

        # 使用grid布局实现均匀分布
        left_buttons.grid(row=0, column=0, sticky="ew")
        center_buttons.grid(row=0, column=1, sticky="ew")  # 中间列
        right_buttons.grid(row=0, column=2, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1, 2), weight=1)  # 均匀分布三列

        # 左侧按钮列（控制选项）
        control_col = tk.Frame(left_buttons)
        control_col.pack(anchor="center", expand=True)

        # 时长输入组
        duration_frame = tk.Frame(control_col)
        duration_frame.pack(pady=2)
        tk.Label(duration_frame, text="训练时长:").pack(side=tk.LEFT)
        self.duration_entry = tk.Entry(duration_frame, width=6)
        self.duration_entry.insert(0, "-1")
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        # 模式选择组
        mode_frame = tk.Frame(control_col)
        mode_frame.pack(pady=2)
        self.mode_menu = tk.OptionMenu(mode_frame, self.game_mode, "单人", "30人")
        self.mode_menu.pack(side=tk.LEFT)
        self.invest_checkbox = tk.Checkbutton(mode_frame, text="投资", variable=self.is_invest)
        self.invest_checkbox.pack(side=tk.LEFT, padx=5)

        # 中间按钮列（核心操作）
        action_col = tk.Frame(center_buttons)
        action_col.pack(anchor="center", expand=True)

        # 核心操作按钮
        action_buttons = [("自动获取数据", self.toggle_auto_fetch)]
        # 单独处理自动获取数据按钮
        for text, cmd in action_buttons:
            btn = tk.Button(action_col, text=text, command=cmd, width=14)  # 加宽按钮
            btn.pack(pady=5, ipadx=5)
            if text == "自动获取数据":
                self.auto_fetch_button = btn
            btn.pack(pady=5, ipadx=5)

        # 右侧按钮列（功能按钮）
        func_col = tk.Frame(right_buttons)
        func_col.pack(anchor="center", expand=True)

        # 预测功能组
        predict_frame = tk.Frame(func_col)
        predict_frame.pack(pady=2)
        self.predict_button = tk.Button(
            predict_frame, text="预测", command=self.predict, width=8, bg="#FFE4B5"
        )
        self.predict_button.pack(side=tk.LEFT, padx=2)

        self.recognize_button = tk.Button(
            predict_frame, text="识别并预测", command=self.recognize, width=10, bg="#98FB98"
        )
        self.recognize_button.pack(side=tk.LEFT, padx=2)

        self.reset_button = tk.Button(
            predict_frame, text="归零", command=self.reset_entries, width=6
        )
        self.reset_button.pack(side=tk.LEFT, padx=2)

        # 设备序列号组（独立行）
        serial_frame = tk.Frame(func_col)
        serial_frame.pack(pady=5)

        self.reselect_button = tk.Button(
            serial_frame, text="选择范围", command=self.reselect_roi, width=10
        )
        self.reselect_button.pack(side=tk.LEFT)

        tk.Label(serial_frame, text="设备号:").pack(side=tk.LEFT)
        self.serial_entry = tk.Entry(serial_frame, textvariable=self.device_serial, width=15)
        self.serial_entry.pack(side=tk.LEFT, padx=3)

        self.serial_button = tk.Button(
            serial_frame, text="更新", command=self.update_device_serial, width=6
        )
        self.serial_button.pack(side=tk.LEFT)

        # 错题本开关
        self.history_button = tk.Button(
            func_col, text="显示错题本", command=self.toggle_history_panel, width=10
        )
        self.history_button.pack(pady=4)  # 可以 side=tk.TOP / BOTTOM 都行

    def toggle_history_panel(self):
        if not self.history_visible:
            self.history_container.pack(side="right", fill="both", padx=5, pady=5)
            self.history_button.config(text="隐藏错题本")
            for w in self.history_frame.winfo_children():
                w.destroy()
            self.render_history(self.history_frame)
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        else:
            self.history_container.pack_forget()
            self.history_button.config(text="显示错题本")
        self.history_visible = not self.history_visible

    def render_history(self, parent):
        # 准备输入数据（完全匹配ArknightsDataset的处理方式）
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        # 从界面获取数据（空值处理为0）
        for name, entry in self.left_monsters.items():
            value = entry.get()
            left_counts[int(name) - 1] = int(value) if value.isdigit() else 0
        for name, entry in self.right_monsters.items():
            value = entry.get()
            right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

        self.history_match.render_similar_matches(left_counts, right_counts)
        try:
            left_rate = self.history_match.left_rate
            right_rate = self.history_match.right_rate
            cur_left = self.history_match.cur_left
            cur_right = self.history_match.cur_right
            sims = self.history_match.sims
            swap = self.history_match.swap
            top20_idx = self.history_match.top20_idx
            # 清空旧内容
            for w in parent.winfo_children():
                w.destroy()

            # 标题
            head = tk.Frame(parent)
            head.pack(fill="x", pady=4)
            fgL, fgR = ("#E23F25", "#666") if left_rate > right_rate else ("#666", "#25ace2")
            tk.Label(head, text="近5条左右胜率：", font=("Helvetica", 12, "bold")).pack(side="left")
            tk.Label(
                head, text=f"左边 {left_rate:.2%}  ", fg=fgL, font=("Helvetica", 12, "bold")
            ).pack(side="left")
            tk.Label(
                head, text=f"右边 {right_rate:.2%}", fg=fgR, font=("Helvetica", 12, "bold")
            ).pack(side="left")

            # 错题本主体渲染
            self._history_parent = parent
            self._top20 = top20_idx.tolist()
            self._sims = sims
            self._swap = swap
            self._batch_idx = 0

            # 调整Canvas宽度
            self.history_canvas.config(width=700)  # 增加Canvas宽度
            self.history_frame.config(width=700)  # 增加Frame宽度

            parent.after(0, lambda: self._render_batch(batch_size=5))

        except Exception as e:
            print("[渲染错题本失败]", e)

    def _render_batch(self, batch_size=5):
        start = self._batch_idx * batch_size
        end = start + batch_size
        history_match = self.history_match
        parent = self._history_parent
        top20 = self._top20
        sims = self._sims
        swap = self._swap

        for rank, idx in enumerate(top20[start:end], start + 1):
            sims_val = sims[idx]
            swapped = swap[idx]
            Lh, Rh = (history_match.past_left if not swapped else history_match.past_right)[idx], (
                history_match.past_right if not swapped else history_match.past_left
            )[idx]
            lab = history_match.labels[idx]
            if swapped:
                lab = "L" if lab == "R" else "R"
            winL, winR = (lab == "L"), (lab == "R")

            # csv中的行数=局数
            real_no = idx + 2

            row = tk.Frame(parent, pady=6)
            row.pack(fill="x")

            # 左侧信息区域
            info_frame = tk.Frame(row)
            info_frame.pack(side="left", fill="y", padx=5)

            # 局数
            tk.Label(
                info_frame,
                text=f"第 {real_no} 局",
                font=("Helvetica", 10),
            ).pack(anchor="w")

            # 相似度
            tk.Label(
                info_frame, text=f"{rank}. 相似度 {sims_val:.2f}", font=("Helvetica", 10, "bold")
            ).pack(anchor="w")

            # 右侧阵容区域
            roster_frame = tk.Frame(row)
            roster_frame.pack(side="right", fill="both", expand=True)

            # 左右阵容渲染（修改为水平排列）
            for side_name, is_left, win, bg_color, fg_color in [
                ("左", True, winL, "#ffe5e5", "#E23F25"),
                ("右", False, winR, "#e5e5ff", "#25ace2"),
            ]:
                pane = tk.Frame(
                    roster_frame,
                    bd=2,
                    relief="solid",
                    bg=bg_color,
                    highlightbackground="red" if winL and is_left or winR and not is_left else "#aaa",
                    highlightthickness=2,
                )
                pane.pack(side="left" if is_left else "right", fill="both", expand=True, padx=5)

                # 侧边阵容
                tk.Label(
                    pane,
                    text=f"{side_name}边",
                    fg=fg_color if win else "#666",
                    bg=bg_color if win else "#f0f0f0",
                    font=("Helvetica", 9, "bold"),
                ).pack(anchor="w", padx=4)
                inner = tk.Frame(pane, bg=bg_color if win else "#f0f0f0")
                inner.pack(fill="x", padx=4, pady=2)
                data = Lh if is_left else Rh
                for i, count in enumerate(data):
                    if count > 0:
                        img = self.images[str(i + 1)]
                        tk.Label(inner, image=img, bg=bg_color if win else "#f0f0f0").pack(
                            side="left", padx=2
                        )
                        tk.Label(
                            inner, text=f"×{int(count)}", bg=bg_color if win else "#f0f0f0"
                        ).pack(side="left", padx=(0, 6))
        self._batch_idx += 1
        # 更新滚动区域
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        if end < len(top20):
            parent.after(50, lambda: self._render_batch(batch_size))

    def reset_entries(self):
        for entry in self.left_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        for entry in self.right_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        self.result_label.config(text="Prediction: ")

    def get_prediction(self):
        # 准备输入数据（完全匹配ArknightsDataset的处理方式）
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        # 从界面获取数据（空值处理为0）
        for name, entry in self.left_monsters.items():
            value = entry.get()
            left_counts[int(name) - 1] = int(value) if value.isdigit() else 0
        for name, entry in self.right_monsters.items():
            value = entry.get()
            right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

        try:
            prediction = self.cannot_model.get_prediction(left_counts, right_counts)
            return prediction
        except FileNotFoundError:
            messagebox.showerror("错误", "未找到模型文件，请先点击「训练」按钮")
            return 0.5
        except RuntimeError as e:
            if "size mismatch" in str(e):
                messagebox.showerror("错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            return 0.5
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字（0或正整数）")
            return 0.5
        except Exception as e:
            messagebox.showerror("错误", f"预测时发生错误: {str(e)}")
            return 0.5

    def predictText(self, prediction):
        # 结果解释（注意：prediction直接对应标签'R'的概率）
        right_win_prob = prediction  # 模型输出的是右方胜率
        left_win_prob = 1 - right_win_prob

        # 格式化输出
        result_text = (
            f"预测结果:\n\n左方胜率: {left_win_prob:.2%}\t\t" f"右方胜率: {right_win_prob:.2%}"
        )

        # 根据胜率设置颜色（保持与之前一致）
        self.result_label.config(text=result_text)
        if left_win_prob > 0.7:
            self.result_label.config(fg="#E23F25", font=("Helvetica", 12, "bold"))  # red
        elif left_win_prob > 0.6:
            self.result_label.config(fg="#E23F25", font=("Helvetica", 12, "bold"))
        elif right_win_prob > 0.7:
            self.result_label.config(fg="#25ace2", font=("Helvetica", 12, "bold"))  # blue
        elif right_win_prob > 0.6:
            self.result_label.config(fg="#25ace2", font=("Helvetica", 12, "bold"))
        else:
            self.result_label.config(fg="black", font=("Helvetica", 12, "bold"))

    def predict(self):
        # 保存当前预测结果用于后续数据收集
        self.current_prediction = self.get_prediction()
        self.predictText(self.current_prediction)

        if self.history_visible:
            for w in self.history_frame.winfo_children():
                w.destroy()
            self.render_history(self.history_frame)
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def recognize(self):
        # 如果正在进行自动获取数据，从adb加载截图
        if hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
        else:
            screenshot = None

        if self.no_region:  # 如果尚未选择区域，从adb获取截图
            if self.first_recognize:  # 首次识别时，尝试连接adb
                self.recognizer.main_roi = [
                    (int(0.2479 * loadData.screen_width), int(0.8410 * loadData.screen_height)),
                    (int(0.7526 * loadData.screen_width), int(0.9510 * loadData.screen_height)),
                ]
                adb_path = loadData.adb_path  # 从loadData获取adb路径
                device_serial = loadData.device_serial  # 从loadData获取设备号
                subprocess.run(f"{adb_path} connect {device_serial}", shell=True, check=True)
                self.first_recognize = False
            screenshot = loadData.capture_screenshot()

        results = self.recognizer.process_regions(screenshot=screenshot)
        self.reset_entries()

        # 处理结果
        for res in results:
            region_id = res["region_id"]
            if "error" not in res:
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.delete(0, tk.END)
                    entry.insert(0, number)
                    # Highlight the image if the entry already has data
                    if entry.get():
                        entry.config(bg="yellow")
            else:
                if "matched_id" in res:
                    matched_id = res["matched_id"]
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.delete(0, tk.END)
                    entry.config(bg="red")
                    entry.insert(0, "Error")

        self.predict()
        return self.current_prediction, results, screenshot

    def reselect_roi(self):
        self.recognizer.main_roi = self.recognizer.select_roi()
        self.no_region = False

    def start_training(self):
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        # Update progress
        self.root.update_idletasks()

        # Simulate training process
        subprocess.run(["python", "train.py"])
        self.root.update_idletasks()

        messagebox.showinfo("Info", "Model trained successfully")

    def start_callback(self):
        self.auto_fetch_button.config(text="停止自动获取数据")
        pass

    def stop_callback(self):
        self.auto_fetch_button.config(text="自动获取数据")
        pass

    def update_statistics(self):
        elapsed_time = time.time() - self.auto_fetch.start_time if self.auto_fetch.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.auto_fetch.total_fill_count} ，    "
            f"填写×次数: {self.auto_fetch.incorrect_fill_count}，    "
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟"
        )
        self.stats_label.config(text=stats_text)

    def toggle_auto_fetch(self):
        if not (hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running):
            self.auto_fetch = AutoFetch(
                self.game_mode,
                self.is_invest,
                reset=self.reset_entries,
                recognizer=self.recognize,
                updater=self.update_statistics,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
                training_duration=float(self.duration_entry.get()) * 3600,  # 获取训练时长
            )
            self.auto_fetch.start_auto_fetch()
        else:
            self.auto_fetch.stop_auto_fetch()

    def update_device_serial(self):
        """更新设备序列号"""
        new_serial = self.device_serial.get()
        loadData.set_device_serial(new_serial)
        # 重新初始化设备连接
        loadData.device_serial = None  # 重置device_serial
        loadData.get_device_serial()  # 重新获取设备序列号
        messagebox.showinfo("提示", f"已更新模拟器序列号为: {new_serial}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ArknightsApp(root)
    root.mainloop()
