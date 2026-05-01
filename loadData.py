import subprocess
import time
import cv2
import numpy as np
import logging
import gzip
import win32gui
import win32api
import win32con
import sys
import os
from pathlib import Path
from winrt_capture import WinRTScreenCapture

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class AdbConnector:
    def __init__(self, adb_serial=None):
        self.adb_path = r".\platform-tools\adb.exe"
        self.screen_width = 0
        self.screen_height = 0
        self.device_serial = adb_serial if adb_serial else ""
        self.is_connected = False

    def connect(self):
        # 初始化设备序列号
        try:
            # 如果已经有序列号，则尝试更新该序列号；否则使用默认值
            target_serial = self.device_serial if self.device_serial else "127.0.0.1:5555"
            self.update_device_serial(target_serial)
            logger.info(f"最终使用设备: {self.device_serial}")
        except RuntimeError as e:
            logger.exception(f"初始化设备序列号错误: ", e)
            self.is_connected = False
            return

        if self.device_serial:
            # 获取屏幕分辨率
            self.screen_width, self.screen_height = self.get_window_size()
            self.is_connected = True
        else:
            logger.warning(f"连接模拟器失败，使用默认分辨率1920x1080。")
            self.screen_width, self.screen_height = 1920, 1080
            self.is_connected = False

    def connect_to_emulator(self):
        try:
            # 使用绝对路径连接到雷电模拟器
            connect_cmd = f"{self.adb_path} connect {self.device_serial}"
            subprocess.run(connect_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.exception(f"ADB connect command failed: {e}")
        except FileNotFoundError as e:
            logger.exception(
                f"Error: {e}. Please ensure adb is installed and added to the system PATH."
            )

    def get_window_size(self):
        try:
            # 执行ADB命令获取分辨率
            size_cmd = f"{self.adb_path} -s {self.device_serial} shell wm size"
            result = subprocess.run(
                size_cmd, shell=True, capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()

            # 解析分辨率输出
            if "Physical size:" in output:
                res_str = output.split("Physical size: ")[1]
            elif "Override size:" in output:
                res_str = output.split("Override size: ")[1]
            else:
                raise ValueError("无法解析分辨率输出格式")

            # 分割分辨率并转换为整数
            width, height = map(int, res_str.split("x"))
            if width > height:
                global screen_width, screen_height
                screen_width = width
                screen_height = height
            else:
                screen_width = height
                screen_height = width
            logger.info(f"成功获取模拟器分辨率: {screen_width}x{screen_height}")
        except Exception as e:  # 否则使用默认分辨率
            logger.exception(f"获取分辨率失败，使用默认分辨率1920x1080。错误: {e}")
            screen_width = 1920
            screen_height = 1080
        return screen_width, screen_height

    def get_device_list(self):
        try:
            device_cmd = f"{self.adb_path} devices"
            result = subprocess.run(
                device_cmd, shell=True, capture_output=True, text=True, timeout=5
            )
            devices: list[str] = []
            for line in result.stdout.split("\n"):
                if "\tdevice" in line:
                    dev = line.split("\t")[0]
                    devices.append(dev)
            return devices
        except Exception as e:
            logger.exception(f"获取设备列表失败", e)
            return []

    def update_device_serial(self, serial):
        try:
            if serial == "":
                logger.error(f"当前serial为空")
                serial = "127.0.0.1:5555"
            connect_cmd = f"{self.adb_path} connect {serial}"
            subprocess.run(connect_cmd, shell=True, check=True)

            # 检查手动设备是否在线
            device_cmd = f"{self.adb_path} devices"
            result = subprocess.run(
                device_cmd, shell=True, capture_output=True, text=True, timeout=5
            )
            # 只在调试模式下输出完整设备列表
            logger.debug(f"ADB devices输出:\n{result.stdout}")

            devices: list[str] = []
            for line in result.stdout.split("\n"):
                if "\tdevice" in line:
                    dev = line.split("\t")[0]
                    devices.append(dev)
                    if dev == serial:
                        logger.info(f"使用手动指定设备: {dev}")
                        self.device_serial = serial
                        return dev

            # 只使用指定的设备，不要自动选择其他设备
            logger.error(f"未找到指定的设备: {serial}，当前在线设备: {devices}")
            self.device_serial = ""
            return ""

        except Exception as e:
            logger.exception(f"设备检测失败", e)
            self.device_serial = ""
            return ""

    def capture_screenshot(self):
        if not self.is_connected:
            return None
        return self.capture_screenshot_raw_gzip()

    def capture_screenshot_png(self):
        try:
            ta = time.time()
            # 获取二进制图像数据
            get_png_cmd = f"{self.adb_path} -s {self.device_serial} exec-out screencap -p"
            screenshot_data = subprocess.check_output(get_png_cmd, shell=True)
            # 将二进制数据转换为numpy数组
            img_array = np.frombuffer(screenshot_data, dtype=np.uint8)
            # 使用OpenCV解码图像
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图像数据")
            # logger.debug(f"获取图片用时{time.time()-ta:.3f}s")
            return img
        except subprocess.CalledProcessError as e:
            logger.exception(f"Screenshot capture failed: {e}")
            return None
        except Exception as e:
            logger.exception(f"Image processing error: {e}")
            return None

    def decode_raw(self, data: bytes):
        if len(data) < 8:
            raise RuntimeError("RAW image is empty")
        width = data[0] << 0 | data[1] << 8 | data[2] << 16 | data[3] << 24
        height = data[4] << 0 | data[5] << 8 | data[6] << 16 | data[7] << 24
        if width != self.screen_width or height != self.screen_height:
            logger.error(
                f"width: {width} height: {height} != screen_width: {self.screen_width} screen_height: {self.screen_height}"
            )
            raise RuntimeError(f"RAW图像分辨率与屏幕分辨率不符")
        # 12 or 16. ref: https://android.googlesource.com/platform/frameworks/base/+/26a2b97dbe48ee45e9ae70110714048f2f360f97%5E%21/cmds/screencap/screencap.cpp
        std_size = 4 * width * height
        header_size = len(data) - std_size
        # 将二进制数据转换为numpy数组
        argb_array = np.frombuffer(data, dtype=np.uint8)[header_size:]

        # 确保数据长度正确（实际屏幕分辨率，4通道）
        expected_length = self.screen_width * self.screen_height * 4
        if len(argb_array) != expected_length:
            raise ValueError(f"Invalid data length for {self.screen_width}x{self.screen_height} ARGB image")

        # 转换为正确的形状 (高度, 宽度, 通道)
        argb_array = argb_array.reshape((self.screen_height, self.screen_width, 4))

        # 分离Alpha通道（如果需要保留Alpha，可以去掉这步）
        # 这里将ARGB转换为BGR（OpenCV默认格式）
        # 通过切片操作 [:, :, [2, 1, 0]] 实现通道交换
        bgr_array = argb_array[:, :, [2, 1, 0]]  # 交换R和B通道

        # 转换为OpenCV可用的连续数组（某些OpenCV操作需要）
        image = np.ascontiguousarray(bgr_array)
        return image

    def decode_raw_with_gzip(self, data: bytes):
        try:
            decompressed_data = gzip.decompress(data)
            image = self.decode_raw(decompressed_data)
            return image
        except Exception as e:
            logger.exception("Gzip decompression or image decoding failed: %s", e)
            return None

    def capture_screenshot_raw_gzip(self):
        get_raw_gzip_cmd = (
            rf'{self.adb_path} -s {self.device_serial} exec-out "screencap | gzip -1"'
        )
        ta = time.time()
        try:
            # 获取经过gzip压缩的二进制图像数据
            screenshot_raw_gzip = subprocess.check_output(get_raw_gzip_cmd, shell=True)
            image = self.decode_raw_with_gzip(screenshot_raw_gzip)
            if image is None:
                raise RuntimeError("OpenCV failed to decode image")
        except subprocess.CalledProcessError as e:
            logger.exception("Screenshot capture failed (ADB error): %s", e)
            return None
        except gzip.BadGzipFile as e:
            logger.exception("Gzip decompression failed: %s", e)
            return None
        except Exception as e:
            logger.exception("Image processing error: %s", e)
            return None
        # logger.debug(f"获取图片用时{time.time()-ta:.3f}s")
        return image

    def click(self, point):
        x, y = point
        x_coord = int(x * self.screen_width)
        y_coord = int(y * self.screen_height)
        logger.info(f"点击坐标: ({x_coord}, {y_coord})")
        click_cmd = f"{self.adb_path} -s {self.device_serial} shell input tap {x_coord} {y_coord}"
        subprocess.run(click_cmd, shell=True)


class PcConnector:
    def __init__(self):
        self.window_name = "明日方舟"
        self.hwnd = None
        self.screen_width = 0
        self.screen_height = 0
        self.capture = None
        self.is_connected = False
        self.maa_ctrl = None

    def connect(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd:
            self.hwnd = hwnd
            rect = win32gui.GetClientRect(self.hwnd)
            self.screen_width = rect[2] - rect[0]
            self.screen_height = rect[3] - rect[1]
            
            try:
                from maa_adb_connector import resolve_maafw_path
                binary_path = resolve_maafw_path()
                if binary_path:
                    os.environ["MAAFW_BINARY_PATH"] = binary_path
                
                from maa.toolkit import Toolkit
                from maa.controller import Win32Controller, MaaWin32ScreencapMethodEnum, MaaWin32InputMethodEnum
                Toolkit.init_option(str(Path.cwd()))
                
                # 既然纯 SendMessage 被引擎无视，我们退一步使用 SendMessageWithCursorPos
                # 它会在瞬间把鼠标光标移动到目标位置发送消息，再瞬间移回原位。这种方式可能不会强制将游戏窗口调回前台。
                self.maa_ctrl = Win32Controller(
                    self.hwnd,
                    screencap_method=MaaWin32ScreencapMethodEnum.FramePool,
                    mouse_method=MaaWin32InputMethodEnum.SendMessageWithCursorPos,
                    keyboard_method=MaaWin32InputMethodEnum.SendMessageWithCursorPos
                )
                self.maa_ctrl.post_connection().wait()
                
                # 设置截图使用原始大小，防止因为MAA默认缩放导致外部坐标及图像切割计算出错
                self.maa_ctrl.set_screenshot_use_raw_size(True)
                
                logger.info(f"已成功通过 MaaFramework 接管 PC 窗口 (支持后台操作)")
            except Exception as e:
                logger.warning(f"MaaFramework 初始化失败，退回原有前台实现: {e}")
                self.maa_ctrl = None
                self.capture = WinRTScreenCapture(window_name=self.window_name)
                self.capture.start()

            self.is_connected = True
            logger.info(f"成功连接到PC端窗口: {self.window_name}, 分辨率: {self.screen_width}x{self.screen_height}")
        else:
            logger.warning(f"未找到PC端窗口: {self.window_name}")
            self.is_connected = False

    def capture_screenshot(self):
        if not self.is_connected:
            return None
        
        if self.maa_ctrl:
            try:
                self.maa_ctrl.post_screencap().wait()
                return self.maa_ctrl.cached_image
            except Exception as e:
                logger.error(f"MAA 截图失败: {e}")
                return None
        else:
            if not self.capture:
                return None
            frame = self.capture.snapshot()
            return frame

    def click(self, point):
        if not self.hwnd:
            return
        
        try:
            # PC端点击需要重新获取一次ClientRect以防窗口大小改变
            rect = win32gui.GetClientRect(self.hwnd)
            self.screen_width = rect[2] - rect[0]
            self.screen_height = rect[3] - rect[1]

            x, y = point
            x_coord = int(x * self.screen_width)
            y_coord = int(y * self.screen_height)
            
            if self.maa_ctrl:
                logger.info(f"MAA后台点击坐标: ({x_coord}, {y_coord})")
                self.maa_ctrl.post_click(x_coord, y_coord).wait()
                return

            client_left, client_top = win32gui.ClientToScreen(self.hwnd, (0, 0))
            screen_x = client_left + x_coord
            screen_y = client_top + y_coord
            
            logger.info(f"PC端点击坐标: 窗口内({x_coord}, {y_coord}) -> 屏幕({screen_x}, {screen_y})")

            # 尝试将窗口置于前台，忽略可能的错误
            try:
                foreground_hwnd = win32gui.GetForegroundWindow()
                if foreground_hwnd != self.hwnd:
                    win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(self.hwnd)
                    time.sleep(0.1)
            except Exception as e:
                logger.debug(f"置于前台失败，可能已经是前台或被阻止: {e}")

            # 使用底层 SendInput 模拟鼠标事件，支持多显示器 (VIRTUALDESK) 并且不会在权限不足时抛出异常
            import ctypes
            
            PUL = ctypes.POINTER(ctypes.c_ulong)
            class KeyBdInput(ctypes.Structure):
                _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
            class HardwareInput(ctypes.Structure):
                _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]
            class MouseInput(ctypes.Structure):
                _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
            class Input_I(ctypes.Union):
                _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
            class Input(ctypes.Structure):
                _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

            SM_XVIRTUALSCREEN = 76
            SM_YVIRTUALSCREEN = 77
            SM_CXVIRTUALSCREEN = 78
            SM_CYVIRTUALSCREEN = 79
            
            vscreen_x = ctypes.windll.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vscreen_y = ctypes.windll.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vscreen_w = ctypes.windll.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vscreen_h = ctypes.windll.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

            if vscreen_w == 0 or vscreen_h == 0:
                vscreen_w = 1920
                vscreen_h = 1080

            # 转换为 0-65535 虚拟桌面绝对坐标
            dx = int((screen_x - vscreen_x) * 65535 / vscreen_w)
            dy = int((screen_y - vscreen_y) * 65535 / vscreen_h)

            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_ABSOLUTE = 0x8000
            MOUSEEVENTF_VIRTUALDESK = 0x4000
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004

            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            
            # 移动鼠标
            ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, ctypes.pointer(extra))
            cmd = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
            
            time.sleep(0.05)
            
            # 按下左键
            ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, ctypes.pointer(extra))
            cmd = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
            
            time.sleep(0.05)
            
            # 抬起左键
            ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, ctypes.pointer(extra))
            cmd = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))

        except Exception as e:
            logger.exception(f"PC端点击出错: {e}")

    def get_device_list(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd:
            return [f"PC: {self.window_name}"]
        return []

    def update_device_serial(self, serial):
        pass


relative_points = [
    (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
    (0.0713, 0.8833),  # 左ALL
    (0.8281, 0.8833),  # 右礼物、自娱自乐
    (0.1640, 0.8833),  # 左礼物
    (0.4979, 0.6324),  # 本轮观望
]


"""
def operation_simple(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx == 0:  # 加入赛事
                click(relative_points[0])
                logger.info("加入赛事")
            elif idx == 1:  # 自娱自乐
                click(relative_points[2])
                logger.info("自娱自乐")
            elif idx == 2:  # 开始游戏
                click(relative_points[0])
                logger.info("开始游戏")
            elif idx in [3, 4, 5]:  # 本轮观望
                click(relative_points[4])
                logger.info("本轮观望")
            elif idx in [10, 11]:
                logger.info("下一轮")
            elif idx in [6, 7]:
                logger.info("等待战斗结束")
            elif idx == 12:  # 返回主页
                click(relative_points[0])
                logger.info("返回主页")
            break  # 匹配到第一个结果后退出


def operation(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx in [3, 4, 5]:
                # 识别怪物类型数量，导入模型进行预测
                prediction = 0.6
                # 根据预测结果点击投资左/右
                if prediction > 0.5:
                    click(relative_points[1])  # 投资右
                    logger.info("投资右")
                else:
                    click(relative_points[0])  # 投资左
                    logger.info("投资左")
            elif idx in [1, 5]:
                click(relative_points[2])  # 点击省点饭钱
                logger.info("点击省点饭钱")
            elif idx == 2:
                click(relative_points[3])  # 点击敬请见证
                logger.info("点击敬请见证")
            elif idx in [3, 4]:
                # 保存数据
                click(relative_points[4])  # 点击下一轮
                logger.info("点击下一轮")
            elif idx == 6:
                logger.info("等待战斗结束")
            break  # 匹配到第一个结果后退出
"""


# def main():
#     while True:
#         screenshot = capture_screenshot()
#         if screenshot is not None:
#             results = match_images(screenshot, process_images)
#             results = sorted(results, key=lambda x: x[1], reverse=True)
#             print("匹配结果：", results[0])
#             operation(results)
#         time.sleep(2)


# if __name__ == "__main__":
#     main()
