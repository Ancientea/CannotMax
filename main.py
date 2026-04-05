import csv
import os
import tkinter as tk
from tkinter import messagebox
import time

import cv2
import keyboard
import numpy as np
import subprocess
import loadData
import threading
import torch
from train import UnitAwareTransformer
import recognize

class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Neural Network")
        self.auto_fetch_running = False

        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}
        self.progress_var = tk.StringVar()
        self.main_roi = None

        # 添加统计信息的变量
        self.total_fill_count = 0
        self.incorrect_fill_count = 0
        self.start_time = None

        self.load_images()
        self.create_widgets()

    def load_images(self):
        for i in range(1, 27):
            original_image = tk.PhotoImage(file=f"images/{i}.png")
            self.images[str(i)] = original_image.subsample(1, 1)  # Reduce size by 50%

    def create_widgets(self):
        # Create frames
        self.top_frame = tk.Frame(self.root)
        self.bottom_frame = tk.Frame(self.root)
        self.button_frame = tk.Frame(self.root)
        self.result_frame = tk.Frame(self.root)

        self.top_frame.pack(side=tk.TOP, padx=10, pady=10)
        self.bottom_frame.pack(side=tk.TOP, padx=10, pady=10)
        self.button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)
        self.result_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Create labels and entries for top and bottom monsters
        for i in range(1, 14):
            tk.Label(self.top_frame, image=self.images[str(i)]).grid(row=0, column=i-1)
            self.left_monsters[str(i)] = tk.Entry(self.top_frame, width=10)
            self.left_monsters[str(i)].grid(row=1, column=i-1)

        for i in range(14, 27):
            tk.Label(self.top_frame, image=self.images[str(i)]).grid(row=2, column=i-14)
            self.left_monsters[str(i)] = tk.Entry(self.top_frame, width=10)
            self.left_monsters[str(i)].grid(row=3, column=i-14)

        for i in range(1, 14):
            tk.Label(self.bottom_frame, image=self.images[str(i)]).grid(row=0, column=i-1)
            self.right_monsters[str(i)] = tk.Entry(self.bottom_frame, width=10)
            self.right_monsters[str(i)].grid(row=1, column=i-1)

        for i in range(14, 27):
            tk.Label(self.bottom_frame, image=self.images[str(i)]).grid(row=2, column=i-14)
            self.right_monsters[str(i)] = tk.Entry(self.bottom_frame, width=10)
            self.right_monsters[str(i)].grid(row=3, column=i-14)

        # Create buttons
        # 添加当次训练时长输入框
        self.duration_label = tk.Label(self.button_frame, text="当次训练时长(小时):")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        self.duration_entry = tk.Entry(self.button_frame, width=4)
        self.duration_entry.insert(0, "799325")  # 默认值为799325表示无限训练时间
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        # self.train_button = tk.Button(self.button_frame, text="训练", command=self.train_model)
        # self.train_button.pack(side=tk.LEFT, padx=5)
        self.auto_fetch_button = tk.Button(self.button_frame, text="自动获取数据", command=self.toggle_auto_fetch)
        self.auto_fetch_button.pack(side=tk.LEFT, padx=5)

        self.fill_correct_button = tk.Button(self.button_frame, text="填写√", command=self.fill_data_correct)
        self.fill_correct_button.pack(side=tk.LEFT, padx=5)

        self.fill_incorrect_button = tk.Button(self.button_frame, text="填写×", command=self.fill_data_incorrect)
        self.fill_incorrect_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="归零", command=self.reset_entries)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.predict_button = tk.Button(self.button_frame, text="{----预测----}", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.recognize_button = tk.Button(self.button_frame, text="识别", command=self.recognize)
        self.recognize_button.pack(side=tk.LEFT, padx=5)

        self.reselect_button = tk.Button(self.button_frame, text="重选范围", command=self.reselect_roi)
        self.reselect_button.pack(side=tk.LEFT, padx=5)

        # Create result label
        self.result_label = tk.Label(self.result_frame, text="Prediction: ", font=("Helvetica", 16))
        self.result_label.pack()

        # Create statistics label
        self.stats_label = tk.Label(self.result_frame, text="", font=("Helvetica", 12))
        self.stats_label.pack()

    def reset_entries(self):
        for entry in self.left_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        for entry in self.right_monsters.values():
            entry.delete(0, tk.END)
            entry.config(bg="white")  # Reset color
        self.result_label.config(text="Prediction: ")

    def fill_data_correct(self):
        result = 'R' if self.current_prediction > 0.5 else 'L'
        self.fill_data(result)
        self.total_fill_count += 1  # 更新总填写次数
        self.update_statistics()  # 更新统计信息

    def fill_data_incorrect(self):
        result = 'L' if self.current_prediction > 0.5 else 'R'
        self.fill_data(result)
        self.total_fill_count += 1  # 更新总填写次数
        self.incorrect_fill_count += 1  # 更新填写×次数
        self.update_statistics()  # 更新统计信息

    def fill_data(self, result):
        image_data = np.zeros((1, 52))
        for name, entry in self.left_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) - 1] = int(value)
        for name, entry in self.right_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) + 26 - 1] = int(value)
        image_data = np.append(image_data, result)
        with open('arknights.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(image_data)
        #messagebox.showinfo("Info", "Data filled successfully")

    def get_prediction(self):
        try:
            # 检查模型文件是否存在
            if not os.path.exists('best_model.pth'):
                raise FileNotFoundError("未找到训练好的模型文件 'best_model.pth'，请先训练模型")

            # 初始化模型（与train.py中的配置完全一致）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = UnitAwareTransformer(
                num_units=26,
                embed_dim=128,
                num_heads=8,
                num_layers=4  # 注意：train.py中config['n_layers']=4
            ).to(device)

            # 加载模型权重（使用strict=False以兼容可能的微小差异）
            model.load_state_dict(torch.load('best_model.pth', map_location=device), strict=False)
            model.eval()

            # 准备输入数据（完全匹配ArknightsDataset的处理方式）
            left_counts = np.zeros(26, dtype=np.float32)
            right_counts = np.zeros(26, dtype=np.float32)

            # 从界面获取数据（空值处理为0）
            for name, entry in self.left_monsters.items():
                value = entry.get()
                left_counts[int(name) - 1] = float(value) if value.isdigit() else 0.0

            for name, entry in self.right_monsters.items():
                value = entry.get()
                right_counts[int(name) - 1] = float(value) if value.isdigit() else 0.0

            # 转换为张量并处理符号和绝对值（关键步骤）
            left_signs = torch.sign(torch.tensor(left_counts, dtype=torch.float32)).unsqueeze(0).to(device)
            left_counts = torch.abs(torch.tensor(left_counts, dtype=torch.float32)).unsqueeze(0).to(device)
            right_signs = torch.sign(torch.tensor(right_counts, dtype=torch.float32)).unsqueeze(0).to(device)
            right_counts = torch.abs(torch.tensor(right_counts, dtype=torch.float32)).unsqueeze(0).to(device)

            # 预测流程（与evaluate函数一致）
            with torch.no_grad():
                # 模型输出已经是sigmoid后的概率值
                prediction = model(left_signs, left_counts, right_signs, right_counts).item()
        except FileNotFoundError:
            messagebox.showerror("错误", "未找到模型文件，请先点击「训练」按钮")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                messagebox.showerror("错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字（0或正整数）")
        except Exception as e:
            messagebox.showerror("错误", f"预测时发生错误: {str(e)}")
        return prediction

    def predictText(self, prediction):
        # 结果解释（注意：prediction直接对应标签'R'的概率）
        right_win_prob = prediction  # 模型输出的是右方胜率
        left_win_prob = 1 - right_win_prob

        # 格式化输出
        result_text = (f"预测结果:\n"
                       f"左方胜率: {left_win_prob:.2%}\n"
                       f"右方胜率: {right_win_prob:.2%}")

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
        prediction = self.get_prediction()
        self.predictText(prediction)
        # 保存当前预测结果用于后续数据收集
        self.current_prediction = prediction

    def recognize(self):
        self.reset_entries()
        if self.main_roi is None:
            #如果当前未执行自动获取数据，则提示用户选择区域
            if not self.auto_fetch_running:
                self.main_roi = recognize.select_roi()
            else:
                self.main_roi = [
                    (int(0.3339 * loadData.screen_width), int(0.7787 * loadData.screen_height)),
                    (int(0.8995 * loadData.screen_width), int(0.9111 * loadData.screen_height))
                ]
        ref_images = recognize.load_ref_images()
        # 如果正在进行自动获取数据，从文件加载截图
        if self.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
        else:
            screenshot = None

        results = recognize.process_regions(self.main_roi, ref_images, screenshot)
        # 输出结果
        for region in self.main_roi:
            print(f"Region: {region}")
        # 处理结果
        for res in results:
            if 'error' not in res:
                region_id = res['region_id']
                matched_id = res['matched_id']
                number = res['number']
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

    def reselect_roi(self):
        self.main_roi = recognize.select_roi()

    def start_training(self):
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        # Update progress
        self.root.update_idletasks()

        # Simulate training process
        subprocess.run(["python", "train.py"])
        self.root.update_idletasks()

        messagebox.showinfo("Info", "Model trained successfully")

    def calculate_average_green(self, image):#计算钱来钱去区域
        if image is None:
            print(f"Failed to load image")
            return None
        height, width, _ = image.shape
        x1, y1, x2, y2 = (0.3406, 0.5759, 0.4182, 0.6194)
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

        region = image[y1:y2, x1:x2]
        green_channel = region[:, :, 1]
        average_green = np.mean(green_channel)
        return average_green

    def save_statistics_to_log(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (f"总共填写次数: {self.total_fill_count}\n"
                      f"填写×次数: {self.incorrect_fill_count}\n"
                      f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n")
        with open("log.txt", "a") as log_file:
            log_file.write(stats_text)

    def toggle_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch_running = True
            self.auto_fetch_button.config(text="停止自动获取数据")
            self.start_time = time.time()  # 记录开始时间
            self.total_fill_count = 0  # 重置总填写次数
            self.incorrect_fill_count = 0  # 重置填写×次数
            self.update_statistics()  # 更新统计信息
            self.training_duration = float(self.duration_entry.get()) * 3600  # 获取训练时长（小时转秒）
            threading.Thread(target=self.auto_fetch_loop).start()
        else:
            self.auto_fetch_running = False
            self.auto_fetch_button.config(text="自动获取数据")
            self.update_statistics()  # 更新统计信息

    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            self.auto_fetch_data()
            self.update_statistics()  # 更新统计信息
            elapsed_time = time.time() - self.start_time
            if self.training_duration != -1 and elapsed_time >= self.training_duration:
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                self.save_statistics_to_log()  # 保存统计信息到log.txt
                break
            time.sleep(2)
            if keyboard.is_pressed('esc'):
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                break

    def auto_fetch_data(self):
        relative_points = [
            (0.453, 0.95),  # 投资左
            (0.779, 0.95),  # 投资右
            (0.322, 0.88),  # 省点饭钱
            (0.864, 0.88),  # 敬请见证
            (0.864, 0.90)  # 下一轮
        ]
        screenshot = loadData.capture_screenshot()
        if screenshot is not None:
            results = loadData.match_images(screenshot, loadData.process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            #print("匹配结果：", results[0])
            for idx, score in results:
                if score > 0.5:
                    if idx == 0:
                        #归零
                        self.reset_entries()
                        # 识别怪物类型数量，导入模型进行预测
                        self.recognize()
                        prediction = self.get_prediction()
                        self.predictText(prediction)
                        # 保存当前预测结果用于后续数据收集
                        self.current_prediction = prediction
                        # 根据预测结果点击投资左/右
                        if prediction > 0.5:
                            loadData.click(relative_points[1])  # 投资右
                            print("投资右")
                        else:
                            loadData.click(relative_points[0])  # 投资左
                            print("投资左")
                    elif idx in [1, 5]:
                        loadData.click(relative_points[2])  # 点击省点饭钱
                        print("点击省点饭钱")
                    elif idx == 2:
                        loadData.click(relative_points[3])  # 点击敬请见证
                        print("点击敬请见证")
                    elif idx in [3, 4]:
                        #80值添加数据√
                        #62值添加数据×
                        # 计算平均绿色值
                        average_green = self.calculate_average_green(screenshot)
                        if average_green > 70:
                            # 填写数据√
                            self.fill_data_correct()
                            print("填写数据√")
                        else:
                            # 填写数据×
                            self.fill_data_incorrect()
                            print("填写数据×")
                        loadData.click(relative_points[4])  # 点击下一轮
                        print("点击下一轮")
                    # elif idx == 6:
                    #     print("等待战斗结束")
                    elif idx == 7:
                        loadData.click(relative_points[4])  # 点击下一轮
                        print("点击返回主页")
                    elif idx == 8:
                        loadData.click(relative_points[4])  # 点击下一轮
                        print("下一局")
                    break  # 匹配到第一个结果后退出
        pass

    # 更新统计信息
    def update_statistics(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (f"总共填写次数: {self.total_fill_count} ，    "
                      f"填写×次数: {self.incorrect_fill_count}，    "
                      f"当次运行时长: {int(hours)}小时{int(minutes)}分钟")
        self.stats_label.config(text=stats_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ArknightsApp(root)
    root.mainloop()
