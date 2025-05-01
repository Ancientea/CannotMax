import csv
import os
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox
import keyboard
import numpy as np
import torch
import loadData
import recognize
import train
from train import UnitAwareTransformer
from recognize import MONSTER_COUNT


class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Neural Network")
        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = tk.BooleanVar(value=False)  # 添加投资状态变量
        self.game_mode = tk.StringVar(value="单人")  # 添加游戏模式变量，默认单人模式
        self.device_serial = tk.StringVar(value=loadData.manual_serial)  # 添加设备序列号变量

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

        # 模型相关属性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # 模型实例
        self.load_model()  # 初始化时加载模型

    def load_images(self):
        for i in range(1, MONSTER_COUNT + 1):
            original_image = tk.PhotoImage(file=f"images/{i}.png")
            # 计算合适的缩放比例使图片显示为60*60像素
            width = original_image.width()
            height = original_image.height()
            width_ratio = width / 60
            height_ratio = height / 60
            # 使用较大的比例确保图片不超过60*60
            ratio = max(width_ratio, height_ratio)
            if ratio > 0:
                self.images[str(i)] = original_image.subsample(int(ratio), int(ratio))
            else:
                self.images[str(i)] = original_image

    def load_model(self):
        """初始化时加载模型"""
        try:
            if not os.path.exists('models/best_model_full.pth'):
                raise FileNotFoundError("未找到训练好的模型文件 'models/best_model_full.pth'，请先训练模型")

            # 初始化模型结构
            #model = torch.load('models/best_model_full.pth', map_location=self.device,weights_only=False)
            #model = torch.load('models/best_model_full.pth', map_location=self.device) if __
            try:
                model = torch.load('models/best_model_full.pth', map_location=self.device, weights_only=False)
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load('models/best_model_full.pth', map_location=self.device)
            model.eval()
            self.model = model.to(self.device)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            messagebox.showerror("严重错误", error_msg)
            self.root.destroy()  # 无法继续运行，退出程序

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

        # 使用嵌套循环创建左侧和右侧怪物输入框
        for side, frame, monsters in [("left", self.top_frame, self.left_monsters),
                                      ("right", self.bottom_frame, self.right_monsters)]:
            for row in range(3):  # 分为3行
                start = row * (MONSTER_COUNT // 3) + 1
                end = (row + 1) * (MONSTER_COUNT // 3) + 1
                for i in range(start, end):
                    tk.Label(frame, image=self.images[str(i)]).grid(row=row * 2, column=i - start)
                    monsters[str(i)] = tk.Entry(frame, width=8)
                    monsters[str(i)].grid(row=row * 2 + 1, column=i - start)

        # Create buttons
        # 添加当次训练时长输入框
        self.duration_label = tk.Label(self.button_frame, text="当次训练时长(小时):")
        self.duration_label.pack(side=tk.LEFT, padx=5)
        self.duration_entry = tk.Entry(self.button_frame, width=4)
        self.duration_entry.insert(0, "-1")  # 默认值为-1表示无限训练时间
        self.duration_entry.pack(side=tk.LEFT, padx=5)

        # self.train_button = tk.Button(self.button_frame, text="训练", command=self.train_model)
        # self.train_button.pack(side=tk.LEFT, padx=5)
        self.auto_fetch_button = tk.Button(self.button_frame, text="自动获取数据", command=self.toggle_auto_fetch)
        self.auto_fetch_button.pack(side=tk.LEFT, padx=5)

        # 添加游戏模式下拉菜单
        self.mode_menu = tk.OptionMenu(self.button_frame, self.game_mode, "单人", "30人")
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        # 添加投资复选框
        self.invest_checkbox = tk.Checkbutton(self.button_frame, text="投资", variable=self.is_invest)
        self.invest_checkbox.pack(side=tk.LEFT, padx=5)

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

        self.reselect_button = tk.Button(self.button_frame, text="选择范围", command=self.reselect_roi)
        self.reselect_button.pack(side=tk.LEFT, padx=5)

        # 添加设备序列号输入框
        self.serial_label = tk.Label(self.button_frame, text="模拟器序列号:")
        self.serial_label.pack(side=tk.LEFT, padx=5)
        self.serial_entry = tk.Entry(self.button_frame, textvariable=self.device_serial, width=15)
        self.serial_entry.pack(side=tk.LEFT, padx=5)
        self.serial_button = tk.Button(self.button_frame, text="更新", command=self.update_device_serial)
        self.serial_button.pack(side=tk.LEFT, padx=5)

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
        image_data = np.zeros((1, MONSTER_COUNT * 2))
        for name, entry in self.left_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) - 1] = int(value)
        for name, entry in self.right_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) + MONSTER_COUNT - 1] = int(value)
        image_data = np.append(image_data, result)
        image_data = np.nan_to_num(image_data, nan=-1)  # 替换所有NaN为-1

        with open('arknights.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(image_data)
        # messagebox.showinfo("Info", "Data filled successfully")

    def get_prediction(self):
        try:
            if self.model is None:
                raise RuntimeError("模型未正确初始化")

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

            # 转换为张量并处理符号和绝对值
            left_signs = torch.sign(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            left_counts = torch.abs(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_signs = torch.sign(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
            right_counts = torch.abs(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)

            # 预测流程
            with torch.no_grad():
                # 使用修改后的模型前向传播流程
                prediction = self.model(left_signs, left_counts, right_signs, right_counts).item()

                # 确保预测值在有效范围内
                if np.isnan(prediction) or np.isinf(prediction):
                    print("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                    prediction = 0.5

                # 检查预测结果是否在[0,1]范围内
                if prediction < 0 or prediction > 1:
                    prediction = max(0, min(1, prediction))

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
        # 如果正在进行自动获取数据，从adb加载截图
        if self.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
        else:
            screenshot = None

        if self.no_region: # 如果尚未选择区域，从adb获取截图
            if self.first_recognize: # 首次识别时，尝试连接adb
                self.main_roi = [
                    (int(0.2479 * loadData.screen_width), int(0.8410 * loadData.screen_height)),
                    (int(0.7526 * loadData.screen_width), int(0.9510 * loadData.screen_height))
                ]
                adb_path = loadData.adb_path # 从loadData获取adb路径
                device_serial = loadData.device_serial # 从loadData获取设备号
                subprocess.run(f'{adb_path} connect {device_serial}', shell=True, check=True)
                self.first_recognize = False
            screenshot = loadData.capture_screenshot()

        results = recognize.process_regions(self.main_roi,screenshot=screenshot)
        self.reset_entries()

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

    def calculate_average_yellow(self, image):  # 检测左上角一点是否为黄色
        if image is None:
            print(f"图像加载失败")
            return None
        height, width, _ = image.shape
        # 取左上角(0,0)点
        point_color = image[0, 0]
        # 提取BGR通道值
        blue, green, red = point_color
        # 判断是否为黄色 (黄色RGB值大致为R高、G高、B低)
        is_yellow = (red > 150 and green > 150 and blue < 100)
        return is_yellow

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
            self.save_statistics_to_log()  # 保存统计信息到log.txt

    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            try:
                self.auto_fetch_data()
                self.update_statistics()  # 更新统计信息
                elapsed_time = time.time() - self.start_time
                if self.training_duration != -1 and elapsed_time >= self.training_duration:
                    self.auto_fetch_running = False
                    self.auto_fetch_button.config(text="自动获取数据")
                    self.save_statistics_to_log()  # 保存统计信息到log.txt
                    break

                # 检测一次间隔时间——————————————————————————————————
                time.sleep(0.5)
                if keyboard.is_pressed('esc'):
                    self.auto_fetch_running = False
                    self.auto_fetch_button.config(text="自动获取数据")
                    self.save_statistics_to_log()  # 保存统计信息到log.txt
                    break
            except Exception as e:
                print(f"自动获取数据出错: {str(e)}")
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                self.save_statistics_to_log()  # 保存统计信息到log.txt
                break
            #time.sleep(2)
            if keyboard.is_pressed('esc'):
                self.auto_fetch_running = False
                self.auto_fetch_button.config(text="自动获取数据")
                break

    def auto_fetch_data(self):
        relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        screenshot = loadData.capture_screenshot()
        if screenshot is not None:
            results = loadData.match_images(screenshot, loadData.process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            #print("匹配结果：", results[0])
            for idx, score in results:
                if score > 0.5:
                    if idx == 0:
                        loadData.click(relative_points[0])
                        print("加入赛事")
                    elif idx == 1:
                        if self.game_mode.get() == "30人":
                            loadData.click(relative_points[1])
                            print("竞猜对决30人")
                            time.sleep(2)
                            loadData.click(relative_points[0])
                            print("开始游戏")
                        else:
                            loadData.click(relative_points[2])
                            print("自娱自乐")
                    elif idx == 2:
                        loadData.click(relative_points[0])
                        print("开始游戏")
                    elif idx in [3, 4, 5, 15]:
                        time.sleep(1)
                        #归零
                        self.reset_entries()
                        #识别怪物类型数量
                        self.recognize()
                        #预测
                        prediction = self.get_prediction()
                        self.predictText(prediction)
                        self.current_prediction = prediction
                        #点击下一轮
                        if self.is_invest.get():#投资
                            # 根据预测结果点击投资左/右
                            if prediction > 0.5:
                                if idx == 4:
                                    loadData.click(relative_points[0])
                                else:
                                    loadData.click(relative_points[2])
                                print("投资右")
                            else:
                                if idx == 4:
                                    loadData.click(relative_points[1])
                                else:
                                    loadData.click(relative_points[3])
                                print("投资左")
                            if self.game_mode.get() == "30人":
                                time.sleep(20)#30人模式下，投资后需要等待20秒
                        else:#不投资
                            loadData.click(relative_points[4])
                            print("本轮观望")
                            
                    elif idx in [8, 9, 10, 11]:
                        #判断本次是否填写错误
                        if self.calculate_average_yellow(screenshot):
                            self.fill_data('L')
                            if self.current_prediction > 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据左赢")
                        else:
                            self.fill_data('R')
                            if self.current_prediction < 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据右赢")
                        self.total_fill_count += 1  # 更新总填写次数
                        self.update_statistics()  # 更新统计信息
                        print("下一轮")
                        # 为填写数据操作设置冷却期
                        time.sleep(10)
                    elif idx in [6, 7, 14]:
                        print("等待战斗结束")
                    elif idx in [12, 13]:  #返回主页
                        loadData.click(relative_points[0])
                        print("返回主页")
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
