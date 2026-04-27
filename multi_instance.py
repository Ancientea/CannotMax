import sys
import subprocess
import threading
import time
import logging
import ctypes
import win32gui
import win32con
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSpinBox, QTextEdit, QPlainTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 获取屏幕分辨率
def get_screen_resolution():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

# 查找指定标题的窗口
def find_window_by_title(title):
    """查找指定标题的窗口"""
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title in window_title:
                extra.append(hwnd)
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

class MainWorker(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, adb_port):
        super().__init__()
        self.adb_port = adb_port
        self.adb_serial = f"127.0.0.1:{adb_port}"
        self.running = True
        self.process = None
    
    def run(self):
        try:
            # 构建启动main.py的命令，传递ADB序列号
            # 使用start命令创建新的命令行窗口
            command = f'start "明日方舟-端口{self.adb_port}" cmd /k python main.py --adb {self.adb_serial}'
            
            # 添加调试信息
            self.output_signal.emit(f"准备启动实例: 端口{self.adb_port} -> ADB地址: {self.adb_serial}")
            self.output_signal.emit(f"执行命令: {command}")
            
            # 使用start命令创建新的命令行窗口
            # /k参数保持窗口打开，/c参数会在命令执行后关闭窗口
            self.process = subprocess.Popen(
                command, 
                shell=True
            )
            
            self.output_signal.emit(f"端口{self.adb_port} 实例已启动，请查看对应的命令行窗口")
            
            # 等待start命令完成（不是等待main.py完成）
            self.process.wait()
            
            # 由于使用start命令，start命令会立即返回，我们需要跟踪实际的python进程
            # 这里我们假设进程启动成功
            self.output_signal.emit(f"端口{self.adb_port} 启动命令已执行完成")
            
            # 由于无法直接跟踪start命令创建的子进程，我们设置一个标志
            # 表示进程已经启动，但无法直接监控其状态
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(f"启动实例 端口{self.adb_port} 时出错: {str(e)}")
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False
        # 由于使用start命令启动，我们需要通过端口找到对应的python进程并终止
        try:
            # 查找包含特定ADB端口的python进程
            find_cmd = f'tasklist /FI "IMAGENAME eq python.exe" /V | findstr "{self.adb_serial}"'
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and self.adb_serial in result.stdout:
                # 找到了对应的进程，终止它
                kill_cmd = f'taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq 明日方舟-端口{self.adb_port}*"'
                subprocess.run(kill_cmd, shell=True, capture_output=True)
                self.output_signal.emit(f"已尝试终止端口{self.adb_port}的进程")
            else:
                # 如果找不到特定进程，尝试通过窗口标题终止
                kill_cmd = f'taskkill /F /FI "WINDOWTITLE eq 明日方舟-端口{self.adb_port}*"'
                subprocess.run(kill_cmd, shell=True, capture_output=True)
                self.output_signal.emit(f"已尝试通过窗口标题终止端口{self.adb_port}的进程")
        except Exception as e:
            # 如果终止失败，记录错误但不抛出异常
            pass

class MultiInstanceManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("铁鲨鱼一键启动工具")
        self.setGeometry(100, 100, 500, 300)
        
        # 初始化变量
        self.active_workers = []
        self.instance_count = 1
        self.auto_fetch_running = False
        
        # 创建UI
        self.create_ui()
    
    def closeEvent(self, event):
        """窗口关闭时的处理，确保所有子进程都被终止"""
        self.stop_all_instances()
        event.accept()
    
    def create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # 顶部控制区域
        control_layout = QHBoxLayout()
        
        # 实例数量设置
        count_layout = QHBoxLayout()
        count_label = QLabel("启动数量:")
        self.count_spinbox = QSpinBox()
        self.count_spinbox.setRange(1, 10)
        self.count_spinbox.setValue(1)
        self.count_spinbox.valueChanged.connect(self.on_count_changed)
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_spinbox)
        
        # 启动按钮
        start_button = QPushButton("启动多实例")
        start_button.clicked.connect(self.start_multi_instance)
        
        # 停止按钮
        stop_button = QPushButton("停止所有")
        stop_button.clicked.connect(self.stop_all_instances)
        
        # 自动排列按钮
        arrange_button = QPushButton("自动排列")
        arrange_button.clicked.connect(self.auto_arrange)
        
        # 一键自动获取/停止按钮
        auto_fetch_button = QPushButton("一键自动获取")
        auto_fetch_button.clicked.connect(self.toggle_auto_fetch)
        self.auto_fetch_button = auto_fetch_button
        
        control_layout.addLayout(count_layout)
        control_layout.addWidget(start_button)
        control_layout.addWidget(stop_button)
        control_layout.addWidget(auto_fetch_button)
        control_layout.addWidget(arrange_button)
        
        # ADB端口输入区域
        adb_label = QLabel("ADB端口 (每行一个，只输入端口号):")
        self.adb_input = QPlainTextEdit()
        self.adb_input.setPlaceholderText("例如:\n5555\n5557\n5559")
        self.adb_input.setMinimumHeight(150)
        
        # 输出区域
        output_label = QLabel("运行输出:")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(250)
        
        main_layout.addLayout(control_layout)
        main_layout.addWidget(adb_label)
        main_layout.addWidget(self.adb_input)
        main_layout.addWidget(output_label)
        main_layout.addWidget(self.output_text)
        
        central_widget.setLayout(main_layout)
    
    def on_count_changed(self, value):
        self.instance_count = value
    
    def start_multi_instance(self):
        # 获取ADB端口列表
        adb_ports = [port.strip() for port in self.adb_input.toPlainText().split('\n') if port.strip()]
        
        self.output_text.append(f"解析到的端口列表: {adb_ports}")
        
        if not adb_ports:
            QMessageBox.warning(self, "警告", "请输入ADB端口")
            return
        
        if len(adb_ports) < self.instance_count:
            QMessageBox.warning(self, "警告", f"ADB端口数量不足，需要 {self.instance_count} 个，只提供了 {len(adb_ports)} 个")
            return
        
        # 停止之前的所有工作线程
        self.stop_all_instances()
        
        self.output_text.append(f"开始启动 {self.instance_count} 个实例...")
        
        # 为每个实例启动一个Main worker
        for i in range(self.instance_count):
            adb_port = adb_ports[i]
            self.output_text.append(f"正在创建第 {i+1} 个worker，使用端口: {adb_port}")
            
            # 启动main.py实例
            worker = MainWorker(adb_port)
            worker.output_signal.connect(self.append_output)
            worker.error_signal.connect(self.append_output)
            # 使用默认参数捕获当前的adb_port值，避免lambda闭包问题
            worker.finished_signal.connect(lambda p=adb_port: self.on_worker_finished(p))
            worker.start()
            
            self.active_workers.append(worker)
            
            # 等待一段时间，避免同时启动导致的问题
            time.sleep(1)
        
        self.output_text.append("多实例启动完成")
    
    def stop_all_instances(self):
        # 停止所有工作线程
        for worker in self.active_workers:
            worker.stop()
            worker.wait()
        
        # 清空工作线程列表
        self.active_workers.clear()
        self.output_text.append("已停止所有实例")
        self.auto_fetch_running = False
        self.auto_fetch_button.setText("一键自动获取")
    
    def auto_arrange(self):
        """自动排列所有启动的客户端窗口"""
        # 获取ADB端口列表
        adb_ports = [port.strip() for port in self.adb_input.toPlainText().split('\n') if port.strip()]
        
        if not adb_ports:
            QMessageBox.warning(self, "警告", "请输入ADB端口")
            return
        
        # 限制排列数量
        arrange_count = min(len(adb_ports), self.instance_count)
        
        if arrange_count == 0:
            QMessageBox.warning(self, "警告", "没有可排列的实例")
            return
        
        # 获取屏幕分辨率
        screen_width, screen_height = get_screen_resolution()
        
        # 计算每个窗口的大小和位置
        # 下半部分高度
        half_height = screen_height // 2
        # 每个窗口宽度
        window_width = screen_width // arrange_count
        window_height = half_height
        
        self.output_text.append(f"开始自动排列 {arrange_count} 个客户端窗口...")
        self.output_text.append(f"屏幕分辨率: {screen_width}x{screen_height}")
        self.output_text.append(f"每个窗口大小: {window_width}x{window_height}")
        
        # 查找所有可能的窗口
        all_windows = []
        
        # 查找标题包含以下关键字的窗口
        keywords = ["明日方舟", "铁鲨鱼", "Python", "main.py"]
        
        for keyword in keywords:
            windows = find_window_by_title(keyword)
            all_windows.extend(windows)
        
        # 去重
        all_windows = list(set(all_windows))
        
        # 过滤出可见窗口，排除多开工具窗口
        visible_windows = []
        for hwnd in all_windows:
            if win32gui.IsWindowVisible(hwnd):
                # 排除多开工具窗口
                try:
                    title = win32gui.GetWindowText(hwnd)
                    if "铁鲨鱼一键启动工具" not in title:
                        visible_windows.append(hwnd)
                except:
                    pass
        
        all_windows = visible_windows
        
        if not all_windows:
            self.output_text.append("未找到任何窗口")
            return
        
        self.output_text.append(f"找到 {len(all_windows)} 个窗口")
        
        # 显示找到的窗口标题
        for hwnd in all_windows:
            try:
                title = win32gui.GetWindowText(hwnd)
                self.output_text.append(f"找到窗口: {title}")
            except:
                pass
        
        # 区分命令行窗口和程序窗口
        cmd_windows = []
        app_windows = []
        
        # 为窗口添加端口号信息
        cmd_windows_with_port = []
        app_windows_with_port = []
        
        for hwnd in all_windows:
            try:
                title = win32gui.GetWindowText(hwnd)
                
                # 提取端口号
                port = None
                if "端口" in title:
                    # 尝试从标题中提取端口号
                    import re
                    match = re.search(r'端口(\d+)', title)
                    if match:
                        port = int(match.group(1))
                
                # 识别命令行窗口
                if any(keyword in title for keyword in ["Python", "cmd", "命令提示符", "端口"]):
                    cmd_windows_with_port.append((hwnd, port))
                # 识别程序窗口（排除多开工具）
                elif "铁鲨鱼_Arknights Neural Network" in title or "明日方舟" in title:
                    app_windows_with_port.append((hwnd, port))
            except:
                pass
        
        # 按端口号排序
        cmd_windows_with_port.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
        app_windows_with_port.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
        
        # 提取排序后的窗口
        cmd_windows = [hwnd for hwnd, port in cmd_windows_with_port]
        app_windows = [hwnd for hwnd, port in app_windows_with_port]
        
        self.output_text.append(f"命令行窗口: {len(cmd_windows)} 个")
        self.output_text.append(f"程序窗口: {len(app_windows)} 个")
        
        # 先排列命令行窗口
        for i, hwnd in enumerate(cmd_windows[:arrange_count]):
            # 计算窗口位置（下半部分）
            x = i * window_width
            y = screen_height - half_height  # 下半部分
            
            try:
                # 获取窗口标题
                window_title = win32gui.GetWindowText(hwnd)
                self.output_text.append(f"正在排列命令行窗口: {window_title}")
                
                # 移动和调整窗口大小
                win32gui.MoveWindow(hwnd, x, y, window_width, window_height, True)
                
                # 激活窗口
                win32gui.SetForegroundWindow(hwnd)
                
                self.output_text.append(f"已排列命令行窗口到位置 ({x}, {y})")
            except Exception as e:
                self.output_text.append(f"排列命令行窗口时出错: {str(e)}")
            
            # 等待一段时间，避免同时操作导致的问题
            time.sleep(0.5)
        
        # 再排列程序窗口
        for i, hwnd in enumerate(app_windows[:arrange_count]):
            # 计算窗口位置（下半部分）
            x = i * window_width
            y = screen_height - half_height  # 下半部分
            
            try:
                # 获取窗口标题
                window_title = win32gui.GetWindowText(hwnd)
                self.output_text.append(f"正在排列程序窗口: {window_title}")
                
                # 移动和调整窗口大小
                win32gui.MoveWindow(hwnd, x, y, window_width, window_height, True)
                
                # 激活窗口
                win32gui.SetForegroundWindow(hwnd)
                
                self.output_text.append(f"已排列程序窗口到位置 ({x}, {y})")
            except Exception as e:
                self.output_text.append(f"排列程序窗口时出错: {str(e)}")
            
            # 等待一段时间，避免同时操作导致的问题
            time.sleep(0.5)
        
        self.output_text.append("自动排列完成")
    
    def toggle_auto_fetch(self):
        """一键控制所有实例的自动获取功能"""
        # 获取ADB端口列表
        adb_ports = [port.strip() for port in self.adb_input.toPlainText().split('\n') if port.strip()]
        
        if not adb_ports:
            QMessageBox.warning(self, "警告", "请输入ADB端口")
            return
        
        # 限制操作数量
        operate_count = min(len(adb_ports), self.instance_count)
        
        if operate_count == 0:
            QMessageBox.warning(self, "警告", "没有可操作的实例")
            return
        
        if not self.auto_fetch_running:
            # 启动自动获取
            self.output_text.append(f"开始一键启动 {operate_count} 个实例的自动获取...")
            
            # 向每个实例发送自动获取命令
            for i in range(operate_count):
                adb_port = adb_ports[i]
                adb_serial = f"127.0.0.1:{adb_port}"
                
                try:
                    # 创建命令文件
                    command_file = f"command_{adb_serial.replace(':', '_')}.txt"
                    with open(command_file, 'w') as f:
                        f.write("start_auto_fetch")
                    
                    self.output_text.append(f"已向端口{adb_port}发送自动获取命令")
                except Exception as e:
                    self.output_text.append(f"向端口{adb_port}发送命令时出错: {str(e)}")
            
            self.auto_fetch_running = True
            self.auto_fetch_button.setText("一键停止获取")
            self.output_text.append("一键启动自动获取完成")
        else:
            # 停止自动获取
            self.output_text.append(f"开始一键停止 {operate_count} 个实例的自动获取...")
            
            # 向每个实例发送停止获取命令
            for i in range(operate_count):
                adb_port = adb_ports[i]
                adb_serial = f"127.0.0.1:{adb_port}"
                
                try:
                    # 创建命令文件
                    command_file = f"command_{adb_serial.replace(':', '_')}.txt"
                    with open(command_file, 'w') as f:
                        f.write("stop_auto_fetch")
                    
                    self.output_text.append(f"已向端口{adb_port}发送停止获取命令")
                except Exception as e:
                    self.output_text.append(f"向端口{adb_port}发送命令时出错: {str(e)}")
            
            self.auto_fetch_running = False
            self.auto_fetch_button.setText("一键自动获取")
            self.output_text.append("一键停止自动获取完成")
    
    def append_output(self, text):
        self.output_text.append(text)
        # 自动滚动到底部
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
    
    def on_worker_finished(self, port):
        self.output_text.append(f"[端口{port}] 实例已结束")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiInstanceManager()
    window.show()
    sys.exit(app.exec())
