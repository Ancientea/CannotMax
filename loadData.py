import subprocess
import time
import cv2
import numpy as np
import logging
import gzip

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

adb_path = r".\platform-tools\adb.exe"
# 默认设备序列号，可以在main.py中修改
manual_serial = "127.0.0.1:5555"

screen_width = 0
screen_height = 0
process_images = []


def set_device_serial(serial):
    global manual_serial
    manual_serial = serial


def get_device_serial():
    global device_serial
    try:
        # 使用当前的manual_serial值
        subprocess.run(f"{adb_path} connect {manual_serial}", shell=True, check=True)

        # 检查手动设备是否在线
        result = subprocess.run(
            f"{adb_path} devices", shell=True, capture_output=True, text=True, timeout=5
        )

        devices = []
        for line in result.stdout.split("\n"):
            if "\tdevice" in line:
                dev = line.split("\t")[0]
                devices.append(dev)
                if dev == manual_serial:
                    device_serial = dev
                    return dev

        # 自动选择第一个可用设备
        if devices:
            device_serial = devices[0]
            logger.info(f"自动选择设备: {device_serial}")
            return device_serial

        logger.warning("未找到连接的Android设备")
        return None

    except Exception as e:
        logger.exception(f"设备检测失败", e)
        return None


def connect_to_emulator():
    try:
        # 使用绝对路径连接到雷电模拟器
        subprocess.run(f"{adb_path} connect {device_serial}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception(f"ADB connect command failed: {e}")
    except FileNotFoundError as e:
        logger.exception(
            f"Error: {e}. Please ensure adb is installed and added to the system PATH."
        )


def connect():
    global device_serial
    # 初始化设备序列号
    try:
        device_serial = get_device_serial()
        logger.info(f"最终使用设备: {device_serial}")
    except RuntimeError as e:
        logger.exception(f"初始化设备序列号错误: ", e)
        exit(1)

    connect_to_emulator()

    # 获取屏幕分辨率
    try:
        # 执行ADB命令获取分辨率
        result = subprocess.run(
            f"{adb_path} -s {device_serial} shell wm size",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
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
    global process_images
    process_images = [cv2.imread(f"images/process/{i}.png") for i in range(16)]  # 16个模板
    process_images = [cv2.resize(img, (screen_width, screen_height)) for img in process_images]


relative_points = [
    (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
    (0.0713, 0.8833),  # 左ALL
    (0.8281, 0.8833),  # 右礼物、自娱自乐
    (0.1640, 0.8833),  # 左礼物
    (0.4979, 0.6324),  # 本轮观望
]


def capture_screenshot_png():
    try:
        ta = time.time()
        # 获取二进制图像数据
        screenshot_data = subprocess.check_output(
            f"{adb_path} -s {device_serial} exec-out screencap -p", shell=True
        )
        # 将二进制数据转换为numpy数组
        img_array = np.frombuffer(screenshot_data, dtype=np.uint8)
        # 使用OpenCV解码图像
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像数据")
        logger.debug(f"获取图片用时{time.time()-ta:.3f}s")
        return img
    except subprocess.CalledProcessError as e:
        logger.exception(f"Screenshot capture failed: {e}")
        return None
    except Exception as e:
        logger.exception(f"Image processing error: {e}")
        return None


def capture_screenshot_raw_gzip():
    try:
        ta = time.time()
        # 获取经过gzip压缩的二进制图像数据
        screenshot_data = subprocess.check_output(
            rf'{adb_path} -s {device_serial} exec-out "screencap | gzip -1"', shell=True
        )
        # 解压gzip数据
        try:
            decompressed_data = gzip.decompress(screenshot_data)
        except gzip.BadGzipFile as e:
            raise RuntimeError("Gzip decompression failed") from e
        try:
            # 将二进制数据转换为numpy数组
            argb_array = np.frombuffer(decompressed_data, dtype=np.uint8)[16:]

            # 确保数据长度正确（1920x1080分辨率，4通道）
            if len(argb_array) != 1920 * 1080 * 4:
                raise ValueError("Invalid data length for 1920x1080 ARGB image")

            # 转换为正确的形状 (高度, 宽度, 通道)
            argb_array = argb_array.reshape((1080, 1920, 4))

            # 分离Alpha通道（如果需要保留Alpha，可以去掉这步）
            # 这里将ARGB转换为BGR（OpenCV默认格式）
            # 通过切片操作 [:, :, [2, 1, 0]] 实现通道交换
            bgr_array = argb_array[:, :, [2, 1, 0]]  # 交换R和B通道

            # 转换为OpenCV可用的连续数组（某些OpenCV操作需要）
            image = np.ascontiguousarray(bgr_array)
            logger.debug(f"获取图片用时{time.time()-ta:.3f}s")

        except Exception as e:
            raise RuntimeError(f"Image conversion failed: {str(e)}") from e

        if image is None:
            raise RuntimeError("OpenCV failed to decode image")

        return image
    except subprocess.CalledProcessError as e:
        print(f"Screenshot capture failed (ADB error): {e}")
        return None
    except gzip.BadGzipFile as e:
        print(f"Gzip decompression failed: {e}")
        return None
    except Exception as e:
        print(f"Image processing error: {e}")
        return None


def capture_screenshot():
    return capture_screenshot_raw_gzip()


def match_images(screenshot, templates):
    screenshot_quarter = screenshot[int(screenshot.shape[0] * 3 / 4) :, :]
    results = []
    for idx, template in enumerate(templates):
        template_quarter = template[int(template.shape[0] * 3 / 4) :, :]
        res = cv2.matchTemplate(screenshot_quarter, template_quarter, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        results.append((idx, max_val))
    return results


def click(point):
    x, y = point
    x_coord = int(x * screen_width)
    y_coord = int(y * screen_height)
    logger.info(f"点击坐标: ({x_coord}, {y_coord})")
    subprocess.run(f"{adb_path} -s {device_serial} shell input tap {x_coord} {y_coord}", shell=True)


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
