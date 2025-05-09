import logging
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 类伽马变换函数
def adjust_quasi_gamma(image):
    c = np.arange(256.0 / 255, step=1.0 / 255)
    table = np.log(30 * c + 1) * 65.5
    table = np.uint8(table)
    return cv2.LUT(image, table)


# 备用类伽马参数
def adjust_quasi_gamma_spare(image):
    c = np.arange(256.0 / 255, step=1.0 / 255)
    table = np.log(51) * 64.9
    table = np.uint8(table)
    return cv2.LUT(image, table)


# 分辨率自适应
def flex_pixel(image):
    height, width, _ = image.shape
    size_big_min = np.round(width * 0.048).astype("int")
    size_big_max = np.round(width * 0.064).astype("int")
    size_small_min = np.round(width * 0.016).astype("int")
    size_small_max = np.round(width * 0.032).astype("int")

    return [size_big_min, size_big_max, size_small_min, size_small_max]


# 预处理，返回切割后图片 6+2
def preprocess(image, blur, spare=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    if spare == 0:
        gray = adjust_quasi_gamma(gray)
    elif spare == 1:
        gray = adjust_quasi_gamma_spare(gray)

    _, width, _ = image.shape

    # 对不同分辨率的自适应
    size1 = (np.round(width * 0.0018) * 2 + 1).astype("int")
    # 高斯模糊（参数已调好）
    gray_blur = cv2.GaussianBlur(gray, (size1, size1), blur)
    thresh = cv2.GaussianBlur(thresh, (size1 + 2, size1 + 2), 6)

    # 划定位置

    divide1 = int(width * 0.1)
    divide2 = int(width * 0.2)
    divide3 = int(width * 0.3)
    divide4 = int(width * 0.4)

    divide5 = int(width * 0.6)
    divide6 = int(width * 0.7)
    divide7 = int(width * 0.8)
    divide8 = int(width * 0.9)

    x_ratio = [0, 0.1, 0.2, 0.6, 0.7, 0.8]
    x_ratio_small = [0, 0.8]

    crop_blur = []
    crop_small = []

    crop_blur.append(gray_blur[:, :divide2])
    crop_blur.append(gray_blur[:, divide1:divide3])
    crop_blur.append(gray_blur[:, divide2:divide4])

    crop_blur.append(gray_blur[:, divide5:divide7])
    crop_blur.append(gray_blur[:, divide6:divide8])
    crop_blur.append(gray_blur[:, divide7:])

    crop_small.append(thresh[:, :divide2])
    crop_small.append(thresh[:, divide7:])

    return crop_blur, crop_small, x_ratio, x_ratio_small


# 两个找圆函数
def find_big(crop, x_ratio, minR, maxR, width, p1=30, p2=35):
    results = []
    for subimage, i in zip(crop, x_ratio):
        circles_big = cv2.HoughCircles(
            subimage,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=maxR - minR,
            param1=p1,
            param2=p2,
            minRadius=minR,
            maxRadius=maxR,
        )  # 参数已调好
        if circles_big is not None:
            circles_big_bias = circles_big[0].copy()
            circles_big_bias[:, 0] = circles_big[0][:, 0] + np.round(i * width)
            for m in circles_big_bias:
                j = np.append(m, x_ratio.index(i))
                results.append(j)
                logger.debug(f"big circle: {j}")
        else:
            logger.warning(f"section{x_ratio.index(i)} big circle not detected")

    results = np.array(results)
    return results


def find_small(crop_small, x_ratio_small, x_ratio, minR, maxR, width):
    results = []
    for subimage, i in zip(crop_small, x_ratio_small):
        circles_small = cv2.HoughCircles(
            subimage,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=maxR - minR,
            param1=25,
            param2=35,
            minRadius=minR,
            maxRadius=maxR,
        )
        if circles_small is not None:
            circles_small_bias = circles_small[0].copy()
            circles_small_bias[:, 0] = circles_small[0][:, 0] + np.round(i * width)
            for m in circles_small_bias:
                j = np.append(m, x_ratio.index(i))
                results.append(j)
                logger.debug(f"small circle: {j}")

        else:
            logger.warning(f"section{x_ratio.index(i)} small circle not detected")

    results = np.array(results)
    return results


# 通用异常坐标筛选
def detect_outliers(coords, threshold=0.1):
    """
    检测并剔除异常坐标点
    :param coords: 坐标列表，格式为 [[x1, y1], [x2, y2], ...]
    :param threshold: 距离阈值（单位：标准差），超过该阈值的点被认为是异常点
    :return: 剔除异常点后的坐标列表
    """
    # 将坐标转换为 NumPy 数组
    coords = np.array(coords)

    # 计算所有点之间的距离矩阵
    distance_matrix = squareform(pdist(coords))

    # 计算每个点的平均距离
    avg_distances = np.mean(distance_matrix, axis=1)

    # 计算平均距离的均值和标准差
    mean_avg_distance = np.mean(avg_distances)
    std_avg_distance = np.std(avg_distances)

    # 找出距离均值超过阈值倍标准差的点
    outliers = np.where(avg_distances > mean_avg_distance + threshold * std_avg_distance)[0]

    # 剔除异常点
    filtered_coords = np.delete(coords, outliers, axis=0)

    return filtered_coords, outliers


# 框架创建
def create_frame(cx, cy, r, high_tol=False):
    k = 1.0401189 #修正因子
    m = 4.7102526 #中部padding
    nums_bias=0.9
    nums_bias_inn=0.3
    nums_y=cy-0.5*r
    
    if high_tol==False:
        t = 0.025 #容差因子
        avatar = np.round(np.array([[cx      -r*t, cy      +r*t, cx+2*k*r+r*t, cy-2*k*r-r*t],
                                    [cx+2*k*r-r*t, cy-2*k*r-r*t, cx+4*k*r+r*t, cy      +r*t],
                                    [cx+4*k*r-r*t, cy      +r*t, cx+6*k*r+r*t, cy-2*k*r-r*t],
                                    [cx+(6 *k+m)*r-r*t, cy      +r*t, cx+(8 *k+m)*r+r*t, cy-2*k*r-r*t],
                                    [cx+(8 *k+m)*r-r*t, cy-2*k*r-r*t, cx+(10*k+m)*r+r*t, cy      +r*t],
                                    [cx+(10*k+m)*r-r*t, cy      +r*t, cx+(12*k+m)*r+r*t, cy-2*k*r-r*t]])).astype("int")
        nums = np.round(np.array([[cx+(nums_bias+0*k)*r-r*t, cy+r*t, cx+(nums_bias_inn+2*k)*r+r*t, nums_y],
                                  [cx+(nums_bias+2*k)*r-r*t, nums_y, cx+(nums_bias_inn+4*k)*r+r*t, cy+r*t],
                                  [cx+(nums_bias+4*k)*r-r*t, cy+r*t, cx+(nums_bias_inn+6*k)*r+r*t, nums_y],
                                  [cx+(-nums_bias_inn+6 *k+m)*r-r*t, cy+r*t, cx+(-nums_bias+8 *k+m)*r+r*t, nums_y],
                                  [cx+(-nums_bias_inn+8 *k+m)*r-r*t, nums_y, cx+(-nums_bias+10*k+m)*r+r*t, cy+r*t],
                                  [cx+(-nums_bias_inn+10*k+m)*r-r*t, cy+r*t, cx+(-nums_bias+12*k+m)*r+r*t, nums_y]])).astype("int")
        
        return avatar,nums

    if high_tol==True:
        t = 0.17 #容差因子
        avatar = np.round(np.array([[cx      -r*t, cy      +r*t, cx+2*k*r+r*t, cy-2*k*r-r*t],
                                    [cx+2*k*r-r*t, cy-2*k*r-r*t, cx+4*k*r+r*t, cy      +r*t],
                                    [cx+4*k*r-r*t, cy      +r*t, cx+6*k*r+r*t, cy-2*k*r-r*t],
                                    [cx+(6 *k+m)*r-r*t, cy      +r*t, cx+(8 *k+m)*r+r*t, cy-2*k*r-r*t],
                                    [cx+(8 *k+m)*r-r*t, cy-2*k*r-r*t, cx+(10*k+m)*r+r*t, cy      +r*t],
                                    [cx+(10*k+m)*r-r*t, cy      +r*t, cx+(12*k+m)*r+r*t, cy-2*k*r-r*t]])).astype("int")
        nums = np.round(np.array([[cx+(nums_bias+0*k)*r-r*t, cy+r*t, cx+(nums_bias_inn+2*k)*r+r*t, nums_y],
                                  [cx+(nums_bias+2*k)*r-r*t, nums_y, cx+(nums_bias_inn+4*k)*r+r*t, cy+r*t],
                                  [cx+(nums_bias+4*k)*r-r*t, cy+r*t, cx+(nums_bias_inn+6*k)*r+r*t, nums_y],
                                  [cx+(-nums_bias_inn+6 *k+m)*r-r*t, cy+r*t, cx+(-nums_bias+8 *k+m)*r+r*t, nums_y],
                                  [cx+(-nums_bias_inn+8 *k+m)*r-r*t, nums_y, cx+(-nums_bias+10*k+m)*r+r*t, cy+r*t],
                                  [cx+(-nums_bias_inn+10*k+m)*r-r*t, cy+r*t, cx+(-nums_bias+12*k+m)*r+r*t, nums_y]])).astype("int")
        
        return avatar,nums
        
def filter(results_big, results_small, height):
    high_tol = 0
    big_key = 0
    # 开始清洗数据
    if results_big.shape == (0,):
        logger.warning("未识别到大圆，自动进入高容差模式")
        high_tol = 1
        big_key = 1
    small_key = 0  # 0代表小圆正常
    if results_small.shape != (0,):
        filtered_small = results_small[results_small[:, 1] >= height / 2]
        if filtered_small.shape[0] != 2:
            if filtered_small.shape[0] == 1:
                small_key = 1  # 1代表只探测到一个小圆，取xy进入最小二乘
            else:
                small_key = 2  # 2代表探测到大于2个小圆，将进入深筛选

        if small_key == 2:  # y坐标深筛选
            mean_y = np.mean(filtered_small[:, 1])
            abs_diff = np.abs(filtered_small[:, 1] - mean_y)
            min_two_indices = np.argsort(filtered_small)[:2]
            filtered_small = filtered_small[min_two_indices]

            if np.count_nonzero(filtered_small[:, -1] == 0) in (0, 2):
                small_key = 3  # 3代表至少有一侧识别失败, 另一侧也不可分辨
            else:
                small_key = 0

    else:
        small_key = 3

    # 小圆正常识别，筛选大圆的流程
    if small_key == 0:
        if big_key == 1:
            filtered_big = []
            logger.warning("仅使用小圆进入最小二乘")
        else:
            r_refer = np.abs(filtered_small[1, 0] - filtered_small[0, 0]) / 16.39  # 大圆的参考半径
            diff_r = np.abs(results_big[:, 2] - r_refer)
            filtered_big = results_big[diff_r <= 0.05 * r_refer]
            # y筛选
            std_y = np.std(filtered_big[:, 1])
            while std_y > 0.02 * r_refer:
                # 如果数组为空，报错
                if filtered_big.size == 0:
                    logger.error(results_big)
                    raise IndexError("std_y筛选出现问题，请检查以上数据输入是否合法")

                mean_value = np.mean(filtered_big[:, 1])
                abs_diff = np.abs(filtered_big[:, 1] - mean_value)
                outlier_index = np.argmax(abs_diff)
                filtered_big = np.delete(filtered_big, outlier_index, axis=0)
                std_y = np.std(filtered_big[:, 1])

            # x筛选
            k = 1.0401189
            p_cx = []  # 参考回溯cx
            for x, y, radius, n in filtered_big:
                if n in [0, 1, 2]:
                    p_cx.append(x - ((2 * n + 1) * k * r_refer))
                elif n in [3, 4, 5]:
                    p_cx.append(x - ((2 * n + 1) * k * r_refer + 4.710 * r_refer))

            filtered_big_p = np.column_stack((filtered_big, p_cx))
            std_x = np.std(filtered_big_p[:, -1])
            while std_x > 0.5 * r_refer:
                # 如果数组为空，报错
                if filtered_big_p.size == 0:
                    logger.error(results_big)
                    raise IndexError("std_x筛选出现问题，请检查以上数据输入是否合法")

                mean_value = np.mean(filtered_big_p[:, -1])
                abs_diff = np.abs(filtered_big_p[:, -1] - mean_value)
                outlier_index = np.argmax(abs_diff)
                filtered_big_p = np.delete(filtered_big_p, outlier_index, axis=0)
                std_x = np.std(filtered_big_p[:, -1])

            filtered_big = filtered_big_p[:, :-1]
            logger.debug(filtered_big)

    # 小圆识别错误的情况
    elif small_key == 3:
        logger.warning("小圆识别异常，将进入高容差模式")
        if big_key == 0:
            if results_big.shape[0] <= 2:  # 大圆数量不够筛选
                logger.warning(
                    "大圆数量不足，直接进入最小二乘"
                )  # 这里最好是抛出一个Error然后回到范围选择
                filtered_big = results_big
                small_key = 4
            else:
                std = []
                for x, y, radius, n in results_big:
                    if n in [0, 1, 2]:
                        a_cx = x - ((2 * n + 1) * k * r_refer)
                    elif n in [3, 4, 5]:
                        a_cx = x - ((2 * n + 1) * k * r_refer + 4.710 * r_refer)
                    a_cy = y + radius
                    std.append([a_cx, a_cy])
                _, out_index = detect_outliers(std, threshold=0.02 * np.mean(results_big[:, 2]))
                filtered_big = np.delete(results_big, out_index, axis=0)

        high_tol = 1

    elif small_key == 1:
        if big_key == 0:
            if results_big.shape[0] <= 2:  # 大圆数量不够筛选
                logger.warning(
                    "警告：大圆数量不足，直接进入最小二乘"
                )  # 这里最好是抛出一个Error然后回到范围选择
                filtered_big = results_big
                small_key = 4
            else:
                std = []
                for x, y, radius, n in results_big:
                    if n in [0, 1, 2]:
                        a_cx = x - ((2 * n + 1) * k * r_refer)
                    elif n in [3, 4, 5]:
                        a_cx = x - ((2 * n + 1) * k * r_refer + 4.710 * r_refer)
                    a_cy = y + radius
                    std.append([a_cx, a_cy])
                _, out_index = detect_outliers(std, threshold=0.02 * np.mean(results_big[:, 2]))
                filtered_big = np.delete(results_big, out_index, axis=0)
        else:
            filtered_big = []
            logger.warning("警告：大圆数量不足，仅以唯一小圆进入框架创建")
            small_key = 4
        high_tol = 1

    if small_key == 4:
        raise RuntimeError("捕捉效果差, 将启用备用参数进行捕捉")
    return filtered_big, filtered_small, high_tol


# 框架切割（参数已封装）
def cutFrame(image, high_tol=False):
    height, width, _ = image.shape

    # 找圆
    R_sets = flex_pixel(image)

    try:
        crop_blur, crop_small, x_ratio, x_ratio_small = preprocess(image, blur=11)
        results_big = find_big(
            crop_blur, x_ratio, minR=R_sets[0], maxR=R_sets[1], width=width, p1=21, p2=28
        )
        results_small = find_small(
            crop_small, x_ratio_small, x_ratio, minR=R_sets[2], maxR=R_sets[3], width=width
        )
        filtered_big, filtered_small, high_tol = filter(results_big, results_small, height)

    except IndexError:
        try:
            crop_blur, crop_small, x_ratio, x_ratio_small = preprocess(image, blur=7, spare=1)
            results_big = find_big(
                crop_blur, x_ratio, minR=R_sets[0], maxR=R_sets[1], width=width, p1=18, p2=24
            )
            results_small = find_small(
                crop_small, x_ratio_small, x_ratio, minR=R_sets[2], maxR=R_sets[3], width=width
            )
            filtered_big, filtered_small, high_tol = filter(results_big, results_small, height)
        except IndexError:
            logger.error("备用参数捕捉失败！请重新框选试试")

    # 开始最小二乘筛选
    def residuals(params, large_circles, small_circles):  # 定义目标函数
        cx, cy, r = params
        k = 1.0401189
        residuals = []

        # 大圆的残差
        for x, y, radius, n in large_circles:
            if n in [0, 1, 2]:
                predicted_x = (2 * n + 1) * k * r + cx
            elif n in [3, 4, 5]:
                predicted_x = (2 * n + 1) * k * r + cx + 4.7102526 * r
            predicted_y = -k * r + cy
            residuals.append(x - predicted_x)
            residuals.append(y - predicted_y)

        # 小圆的残差
        for x, y, radius, n in small_circles:
            if n == 0:
                predicted_x = 0.44576523 * r + cx
                predicted_y = -0.14858841 * r + cy
            elif n == 5:
                predicted_x = 16.745914 * r + cx
                predicted_y = -0.14858841 * r + cy
            residuals.append(x - predicted_x)
            residuals.append(y - predicted_y)

        return residuals

    # 初始猜测
    initial_guess = [0, 200, 60]

    # 使用最小二乘法求解
    result = least_squares(residuals, initial_guess, args=(filtered_big, filtered_small))
    avatar, nums = create_frame(result.x[0], result.x[1], result.x[2], high_tol)

    divisors = np.array([width, height, width, height])
    d_avatar = avatar / divisors
    d_nums = nums / divisors

    return d_avatar, d_nums


if __name__ == "__main__":
    image = cv2.imread("images/tmp/zone1.png")

    height, width, _ = image.shape

    crop_blur, crop_small, x_ratio, x_ratio_small = preprocess(image, blur=11)

    R_sets = flex_pixel(image)
    results_big = find_big(
        crop_blur, x_ratio, minR=R_sets[0], maxR=R_sets[1], width=width, p1=21, p2=32
    )
    results_small = find_small(
        crop_small, x_ratio_small, x_ratio, minR=R_sets[2], maxR=R_sets[3], width=width
    )

    circles = np.round(results_big).astype("int")
    circles_small = np.round(results_small).astype("int")

    for i in circles:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    for i in circles_small:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    d_avatar, d_nums = cutFrame(image)
    divisors = np.array([width, height, width, height])
    avatar = np.round(d_avatar * divisors).astype("int")
    nums = np.round(d_nums * divisors).astype("int")

    for i in avatar:
        x1, y1, x2, y2 = i
        cv2.rectangle(image, (x1, y1), (x2, y2), (225, 0, 225), 2)

    for i in nums:
        x1, y1, x2, y2 = i
        cv2.rectangle(image, (x1, y1), (x2, y2), (225, 225, 0), 2)

    cv2.imshow("Detected circles", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
