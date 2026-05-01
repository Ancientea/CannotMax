from pathlib import Path
import cv2
import numpy as np
import random


def get_webm_frame(webm_path, frame_number):
    """
    从WEBM文件中提取指定帧。

    参数:
    webm_path (str): 输入的WEBM文件路径。
    frame_number (int): 要提取的帧编号。

    返回:
    numpy.ndarray: 提取的帧图像。
    """
    cap = cv2.VideoCapture(webm_path)

    for i in range(frame_number - 1):
        ret = cap.grab()

    # 读取帧
    ret, frame = cap.read()

    if not ret:
        raise Exception(f"无法读取第 {frame_number} 帧")

    cap.release()
    return frame


def get_webm_frame_count(webm_path):
    """
    获取WEBM文件的总帧数。

    参数:
    webm_path (str): 输入的WEBM文件路径。

    返回:
    int: 总帧数。
    """
    cap = cv2.VideoCapture(webm_path)

    # 手动计算总帧数
    total_frames = 0
    while True:
        ret = cap.grab()  # 使用grab()而不是read()更快
        if not ret:
            break
        total_frames += 1

    print(f"total_frames: {total_frames}")

    # 确保总帧数大于0
    if total_frames <= 0:
        cap.release()
        raise Exception("视频帧数为0")

    cap.release()
    return total_frames


def get_random_png_frame(png_folder):
    # 遍历path下所有png文件
    png_list = list(Path(png_folder).glob("*.png"))
    # 随机选择一个png文件
    random_png = random.choice(png_list)
    print(f"随机选择的PNG文件: {random_png}")
    frame = cv2.imread(str(random_png), cv2.IMREAD_UNCHANGED) # Read with alpha channel
    if frame is None:
        raise Exception(f"无法读取PNG文件: {random_png}")
    return frame


def compose_frame(frame, background, x, y):
    """
    将前景帧合成到背景图像上，包含边界检查和裁切。

    参数:
    frame (numpy.ndarray): 前景帧图像。
    background (numpy.ndarray): 背景图像。
    x (int): 前景帧左上角在背景图像上的x坐标。
    y (int): 前景帧左上角在背景图像上的y坐标。

    返回:
    numpy.ndarray: 合成后的图像。
    """
    # 获取背景和前景尺寸
    bg_height, bg_width = background.shape[:2]
    fr_height, fr_width = frame.shape[:2]

    # 计算有效的ROI区域
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(bg_width, x + fr_width)
    y_end = min(bg_height, y + fr_height)

    # 计算frame需要裁切的区域
    frame_x_start = max(0, -x)
    frame_y_start = max(0, -y)
    frame_x_end = fr_width - max(0, (x + fr_width) - bg_width)
    frame_y_end = fr_height - max(0, (y + fr_height) - bg_height)

    # 如果完全超出边界，直接返回背景
    if x_start >= x_end or y_start >= y_end:
        return background

    # 裁切frame
    frame_roi = frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # 在目标位置创建ROI
    roi = background[y_start:y_end, x_start:x_end]

    # 获取前景的alpha通道
    alpha_channel = frame_roi[:, :, 3] / 255.0
    # 获取前景的颜色通道
    overlay_colors = frame_roi[:, :, :3]

    # 调整alpha通道尺寸以匹配背景ROI
    alpha_channel = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]))
    overlay_colors = cv2.resize(overlay_colors, (roi.shape[1], roi.shape[0]))

    # 进行alpha混合
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + overlay_colors[:, :, c] * alpha_channel

    # 将结果放回原图
    background[y_start:y_end, x_start:x_end] = roi

    return background


def composite_random_frame():
    # 读取战场背景图
    battlefield = cv2.imread("./tools/battlefield_composite/monster_images/IM-1.png")
    battlefield = cv2.resize(battlefield, (1920, 1080))
    if battlefield is None:
        raise Exception("无法读取战场背景图")

    # 获取背景图尺寸
    bg_height, bg_width = battlefield.shape[:2]

    frame_width = 1000 * 0.40
    frame_height = 1000 * 0.40

    frame_list = []

    for i in range(10):
        # 生成随机坐标(确保角色完全在背景图内)
        x = random.randint(int(bg_width * 0.1), int(bg_width * 0.3))
        y = random.randint(0, int(bg_height * 0.7))
        frame_list.append([x, y])

    frame_list.sort(key=lambda x: x[1])

    for i in range(10):
        x, y = frame_list[i]
        frame = get_random_png_frame("./tools/battlefield_composite/monster_images/Arc_Frontliner_Leader-Move")

        # 缩放
        factor = (y / bg_height) * 0.4 + 0.6
        new_frame_width = int(frame_width * factor)
        new_frame_height = int(frame_height * factor)
        print(f"缩放比例: {factor}")
        print(f"缩放后的尺寸: {new_frame_width}x{new_frame_height}")
        print(f"缩放后的坐标: {x}, {y}")
        small_frame = cv2.resize(frame, (new_frame_width, new_frame_height))
        # 裁切到外接矩形
        small_frame, center = crop_to_bounding_box(small_frame)
        x1, y1 = small_frame.shape[1], small_frame.shape[0]
        # 画脚下椭圆
        ellipse_x = int(x + center[0])
        ellipse_y = int(y + center[1] + new_frame_height * 0.30)
        black = (0, 0, 0)
        cv2.ellipse(battlefield, (ellipse_x, ellipse_y),
                    (int(new_frame_width * 0.100), int(new_frame_height * 0.025)),
                    0, 0, 360, black, -1)
        # 画框
        cv2.rectangle(battlefield, (x, y), (x + x1, y + y1), (0, 255, 0), 2)

        # 合成图像
        battlefield = compose_frame(small_frame, battlefield, int(x), int(y))

    # 画面左侧增加从左到右的从黑色到透明的渐变
    # 创建一个与图像相同大小的黑色图像
    black_image = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    # 创建一个与图像相同大小的alpha通道，从左到右从255到0
    alpha_channel = np.zeros((bg_height, bg_width), dtype=np.uint8)
    for i in range(int(bg_width * 0.4)):
        alpha_channel[:, i] = int(192 * (1 - i / int(bg_width * 0.4)))

    # 合并黑色图像和alpha通道
    gradient_mask = np.dstack((black_image, alpha_channel))

    # 将渐变掩码应用到图像上
    battlefield = compose_frame(gradient_mask, battlefield, 0, 0)


    # 保存结果
    cv2.imwrite("./tools/battlefield_composite/battlefield_composite.png", battlefield)


def crop_to_bounding_box(image):
    """
    将图像裁切到非透明像素的外接矩形。

    参数:
    image (numpy.ndarray): 输入图像 (包含alpha通道)。

    返回:
    numpy.ndarray: 裁切后的图像。
    """
    # 检查图像是否包含alpha通道
    if image.shape[2] < 4:
        # 如果没有alpha通道，则无法进行基于透明度的裁切
        print("警告: 图像不包含alpha通道，无法进行基于透明度的裁切。返回原图。")
        return image

    # 获取alpha通道
    alpha_channel = image[:, :, 3]
    alpha_channel = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)[1]

    # 找到非透明像素的坐标
    coords = cv2.findNonZero(alpha_channel)
    # coords = alpha_channel[alpha_channel > 0.1]

    # 如果没有非透明像素，返回原图
    if coords is None:
        print("警告: 图像中没有非透明像素。返回原图。")
        return image

    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(coords)

    # 裁切图像
    cropped_image = image[y:y+h, x:x+w]
    center_x = image.shape[1] / 2 - x
    center_y = image.shape[0] / 2 - y
    center = (center_x, center_y)

    return cropped_image, center


if __name__ == "__main__":
    composite_random_frame()
