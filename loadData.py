import subprocess
import time
import cv2
import numpy as np

adb_path = r".\platform-tools\adb.exe"
# 默认设备序列号，可以在main.py中修改
# manual_serial = '127.0.0.1:5555'
manual_serial = '127.0.0.1:16384'

def set_device_serial(serial):
    global manual_serial
    manual_serial = serial

def get_device_serial():
    global device_serial
    try:
        # 使用当前的manual_serial值
        subprocess.run(f'{adb_path} connect {manual_serial}', shell=True, check=True)

        # 检查手动设备是否在线
        result = subprocess.run(
            f'{adb_path} devices',
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )

        devices = []
        for line in result.stdout.split('\n'):
            if '\tdevice' in line:
                dev = line.split('\t')[0]
                devices.append(dev)
                if dev == manual_serial:
                    device_serial = dev
                    return dev

        # 自动选择第一个可用设备
        if devices:
            device_serial = devices[0]
            print(f"自动选择设备: {device_serial}")
            return device_serial

        print("未找到连接的Android设备")
        return None

    except Exception as e:
        print(f"设备检测失败: {str(e)}")
        return None

# 初始化设备序列号
try:
    device_serial = get_device_serial()
    print(f"最终使用设备: {device_serial}")
except RuntimeError as e:
    print(f"错误: {str(e)}")
    exit(1)

process_images = [cv2.imread(f'images/process/{i}.png') for i in range(17)]#16个模板

# 屏幕分辨率
screen_width = 1920
screen_height = 1080

relative_points = [
    (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
    (0.0713, 0.8833),  # 左ALL
    (0.8281, 0.8833),  # 右礼物、自娱自乐
    (0.1640, 0.8833),  # 左礼物
    (0.4979, 0.6324),  # 本轮观望
]


def connect_to_emulator():
    try:
        # 使用绝对路径连接到雷电模拟器
        subprocess.run(f'{adb_path} connect {device_serial}', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ADB connect command failed: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure adb is installed and added to the system PATH.")


connect_to_emulator()


def capture_screenshot():
    try:
        # 获取二进制图像数据
        screenshot_data = subprocess.check_output(
            f'{adb_path} -s {device_serial} exec-out screencap -p',
            shell=True
        )

        # 将二进制数据转换为numpy数组
        img_array = np.frombuffer(screenshot_data, dtype=np.uint8)

        # 使用OpenCV解码图像
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img
    except subprocess.CalledProcessError as e:
        print(f"Screenshot capture failed: {e}")
        return None
    except Exception as e:
        print(f"Image processing error: {e}")
        return None


def match_images(screenshot, templates):
    screenshot_quarter = screenshot[int(screenshot.shape[0] * 3 / 4):, :]
    results = []
    for idx, template in enumerate(templates):
        template_quarter = template[int(template.shape[0] * 3 / 4):, :]
        res = cv2.matchTemplate(screenshot_quarter, template_quarter, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        results.append((idx, max_val))
    return results


def click(point):
    x, y = point
    x_coord = int(x * screen_width)
    y_coord = int(y * screen_height)
    print(f"点击坐标: ({x_coord}, {y_coord})")
    subprocess.run(f'{adb_path} -s {device_serial} shell input tap {x_coord} {y_coord}', shell=True)


def operation_simple(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx == 0:  # 加入赛事
                click(relative_points[0])
                print("加入赛事")
            elif idx == 1:  # 自娱自乐
                click(relative_points[2])
                print("自娱自乐")
            elif idx == 2:  # 开始游戏
                click(relative_points[0])
                print("开始游戏")
            elif idx in [3, 4, 5]:  # 本轮观望
                click(relative_points[4])
                print("本轮观望")
            elif idx in [10, 11]:
                print("下一轮")
            elif idx in [6, 7]:
                print("等待战斗结束")
            elif idx == 12:  # 返回主页
                click(relative_points[0])
                print("返回主页")
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
                    print("投资右")
                else:
                    click(relative_points[0])  # 投资左
                    print("投资左")
            elif idx in [1, 5]:
                click(relative_points[2])  # 点击省点饭钱
                print("点击省点饭钱")
            elif idx == 2:
                click(relative_points[3])  # 点击敬请见证
                print("点击敬请见证")
            elif idx in [3, 4]:
                # 保存数据
                click(relative_points[4])  # 点击下一轮
                print("点击下一轮")
            elif idx == 6:
                print("等待战斗结束")
            break  # 匹配到第一个结果后退出

def main():
    while True:
        screenshot = capture_screenshot()
        if screenshot is not None:
            results = match_images(screenshot, process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            print("匹配结果：", results[0])
            operation(results)
        time.sleep(2)


if __name__ == "__main__":
    main()
