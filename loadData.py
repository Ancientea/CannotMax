import cv2
import subprocess
import time
import gzip
import numpy as np


# 全局变量
adb_path = r'platform-tools\adb.exe'
device_serial = '127.0.0.1:5555'  # 指定设备序列号
process_images = [cv2.imread(f'images/process/{i}.png') for i in range(9)]

#屏幕分辨率
screen_width = 1920
screen_height = 1080

relative_points = [
    (0.453, 0.95),  # 投资左
    (0.779, 0.95),  # 投资右
    (0.322, 0.88),  # 省点饭钱
    (0.864, 0.88),  # 敬请见证
    (0.864, 0.90)   # 下一轮
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
    """
    参考新分支
    """
    try:
        get_png_cmd = f'{adb_path} -s {device_serial} exec-out "screencap -p | gzip -1"'
        compressed_data = subprocess.check_output(get_png_cmd, shell=True)
        screenshot_data = gzip.decompress(compressed_data)
        img_array = np.frombuffer(screenshot_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("警告: 无法解码图像数据")
            return None

        return img
    except subprocess.CalledProcessError as e:
        print(f"Screenshot capture failed (ADB 错误): {e}")
        return None
    except Exception as e:
        print(f"Image processing error (解码错误): {e}")
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

def operation(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx == 0:
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