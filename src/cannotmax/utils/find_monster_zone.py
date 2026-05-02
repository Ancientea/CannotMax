"""
frame_detector.py
─────────────────
游戏帧定位模块：通过 Hough 圆检测 + 最小二乘拟合，从图像中提取头像框与数字框的归一化坐标。

主要入口：cutFrame(image) → (d_avatar, d_nums)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────
# 全局几何常数
# ──────────────────────────────────────────────────────────

_K = 1.0401189  # 圆心间距比例系数
_M = 4.7102526  # 左右两组之间的中部 padding（单位：r）
_SMALL_DX = 16.39  # 小圆 x 间距与大圆半径的比值（用于估算参考半径）

# ──────────────────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────────────────


def _solve_least_squares(fun, x0, args=()):
    """
    单步数值线性最小二乘求解器（替代 scipy.optimize.least_squares）。
    构造数值雅可比矩阵后调用 numpy.linalg.lstsq 求解。
    """

    class _Result:
        def __init__(self, x):
            self.x = x

    x = np.array(x0, dtype=float)
    r = np.array(fun(x, *args))

    if r.size == 0:
        return _Result(x)

    n = len(x)
    J = np.zeros((len(r), n))
    eps = 1e-6

    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        r_eps = np.array(fun(x_eps, *args))
        J[:, i] = (r_eps - r) / eps

    delta, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)

    return _Result(x + delta)


def _detect_outliers(coords, threshold=0.1):
    """
    基于点间平均距离剔除离群坐标。

    Parameters
    ----------
    coords    : array-like, shape (N, 2)
    threshold : float —— 超过 mean + threshold * std 的点视为离群点

    Returns
    -------
    filtered_coords : np.ndarray  剔除离群点后的坐标
    outlier_indices : np.ndarray  被剔除点的索引
    """
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    avg_distances = np.mean(distance_matrix, axis=1)

    mean_d = np.mean(avg_distances)
    std_d = np.std(avg_distances)
    outliers = np.where(avg_distances > mean_d + threshold * std_d)[0]

    return np.delete(coords, outliers, axis=0), outliers


# ──────────────────────────────────────────────────────────
# 图像预处理
# ──────────────────────────────────────────────────────────


def _apply_quasi_gamma(gray):
    """主用类伽马变换（log 曲线映射）。"""
    c = np.arange(256.0 / 255, step=1.0 / 255)
    table = np.uint8(np.log(30 * c + 1) * 65.5)
    return cv2.LUT(gray, table)


def _apply_quasi_gamma_spare(gray):
    """备用类伽马变换（常数映射）。"""
    c = np.arange(256.0 / 255, step=1.0 / 255)
    table = np.uint8(np.log(51) * 64.9)
    return cv2.LUT(gray, table)


def _get_resolution_radius_range(image):
    """
    根据图像宽度自适应计算大圆/小圆的搜索半径范围。

    Returns
    -------
    (big_min, big_max, small_min, small_max) : tuple[int]
    """
    _, width, _ = image.shape
    big_min = int(np.round(width * 0.048))
    big_max = int(np.round(width * 0.064))
    small_min = int(np.round(width * 0.016))
    small_max = int(np.round(width * 0.032))
    return big_min, big_max, small_min, small_max


def _preprocess(image, blur, spare=0):
    """
    图像预处理：灰度化、类伽马变换、高斯模糊，然后按固定比例切割为子图。

    Parameters
    ----------
    image : BGR 图像
    blur  : 高斯模糊 sigma
    spare : 0 → 主用伽马；1 → 备用伽马

    Returns
    -------
    crop_blur       : list[np.ndarray]  6 块用于大圆检测的模糊子图
    crop_thresh     : list[np.ndarray]  2 块用于小圆检测的二值子图
    x_ratio         : list[float]       大圆子图在原图中的 x 偏移比例
    x_ratio_small   : list[float]       小圆子图在原图中的 x 偏移比例
    """
    _, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化（用于小圆检测）
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 伽马变换（用于大圆检测）
    gray = _apply_quasi_gamma(gray) if spare == 0 else _apply_quasi_gamma_spare(gray)

    # 分辨率自适应核大小
    k1 = int(np.round(width * 0.0018)) * 2 + 1

    gray_blur = cv2.GaussianBlur(gray, (k1, k1), blur)
    thresh = cv2.GaussianBlur(thresh, (k1 + 2, k1 + 2), 6)

    # 分割边界（按宽度比例）
    d1, d2, d3, d4 = [int(width * r) for r in (0.1, 0.2, 0.3, 0.4)]
    d5, d6, d7, d8 = [int(width * r) for r in (0.6, 0.7, 0.8, 0.9)]

    x_ratio = [0, 0.1, 0.2, 0.6, 0.7, 0.8]
    x_ratio_small = [0, 0.8]

    crop_blur = [
        gray_blur[:, :d2],
        gray_blur[:, d1:d3],
        gray_blur[:, d2:d4],
        gray_blur[:, d5:d7],
        gray_blur[:, d6:d8],
        gray_blur[:, d7:],
    ]
    crop_thresh = [
        thresh[:, :d2],
        thresh[:, d7:],
    ]

    return crop_blur, crop_thresh, x_ratio, x_ratio_small


# ──────────────────────────────────────────────────────────
# Hough 圆检测
# ──────────────────────────────────────────────────────────


def _detect_big_circles(crop_list, x_ratio, min_r, max_r, width, p1=30, p2=35):
    """
    在各子图上执行 Hough 大圆检测，将局部坐标转换为全图坐标。

    Returns
    -------
    results : np.ndarray, shape (N, 4)  列为 [x, y, radius, section_index]
    """
    results = []
    for subimage, x_off in zip(crop_list, x_ratio):
        circles = cv2.HoughCircles(
            subimage,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_r - min_r,
            param1=p1,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for c in circles[0]:
                cx_global = c[0] + np.round(x_off * width)
                results.append([cx_global, c[1], c[2], x_ratio.index(x_off)])
                logger.debug(f"big circle: {results[-1]}")
        else:
            logger.warning(f"section {x_ratio.index(x_off)} 大圆未检测到")

    return np.array(results)


def _detect_small_circles(crop_list, x_ratio_small, x_ratio, min_r, max_r, width):
    """
    在各子图上执行 Hough 小圆检测，将局部坐标转换为全图坐标。

    Returns
    -------
    results : np.ndarray, shape (N, 4)  列为 [x, y, radius, section_index]
    """
    results = []
    for subimage, x_off in zip(crop_list, x_ratio_small):
        circles = cv2.HoughCircles(
            subimage,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_r - min_r,
            param1=25,
            param2=35,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for c in circles[0]:
                cx_global = c[0] + np.round(x_off * width)
                results.append([cx_global, c[1], c[2], x_ratio.index(x_off)])
                logger.debug(f"small circle: {results[-1]}")
        else:
            logger.warning(f"section {x_ratio.index(x_off)} 小圆未检测到")

    return np.array(results)


# ──────────────────────────────────────────────────────────
# 圆点数据清洗
# ──────────────────────────────────────────────────────────


def _filter_big_circles_by_radius_and_position(results_big, r_refer):
    """
    按参考半径和 y/x 坐标标准差迭代剔除大圆异常点。

    Parameters
    ----------
    results_big : np.ndarray, shape (N, 4)
    r_refer     : float —— 参考大圆半径

    Returns
    -------
    filtered : np.ndarray
    """
    # 半径筛选
    diff_r = np.abs(results_big[:, 2] - r_refer)
    filtered = results_big[diff_r <= 0.05 * r_refer]

    # y 坐标迭代筛选
    std_y = np.std(filtered[:, 1])
    while std_y > 0.02 * r_refer:
        if filtered.size == 0:
            logger.error(results_big)
            raise IndexError("std_y 筛选出现问题，请检查以上数据输入是否合法")
        mean_y = np.mean(filtered[:, 1])
        outlier_idx = np.argmax(np.abs(filtered[:, 1] - mean_y))
        filtered = np.delete(filtered, outlier_idx, axis=0)
        std_y = np.std(filtered[:, 1])

    # x 坐标（回溯 cx）迭代筛选
    p_cx = [
        x - ((2 * n + 1) * _K * r_refer)
        if n in [0, 1, 2]
        else x - ((2 * n + 1) * _K * r_refer + 4.710 * r_refer)
        for x, y, radius, n in filtered
    ]
    filtered_p = np.column_stack((filtered, p_cx))
    std_x = np.std(filtered_p[:, -1])

    while std_x > 0.5 * r_refer:
        if filtered_p.size == 0:
            logger.error(results_big)
            raise IndexError("std_x 筛选出现问题，请检查以上数据输入是否合法")
        mean_x = np.mean(filtered_p[:, -1])
        outlier_idx = np.argmax(np.abs(filtered_p[:, -1] - mean_x))
        filtered_p = np.delete(filtered_p, outlier_idx, axis=0)
        std_x = np.std(filtered_p[:, -1])

    return filtered_p[:, :-1]  # 去掉辅助列


def _filter_big_circles_by_outlier(results_big):
    """
    当小圆识别失败时，仅用离群点检测清洗大圆（高容差模式）。
    """
    std = []
    for x, y, radius, n in results_big:
        a_cx = (
            x - ((2 * n + 1) * _K * radius)
            if n in [0, 1, 2]
            else x - ((2 * n + 1) * _K * radius + 4.710 * radius)
        )
        std.append([a_cx, y + radius])

    _, out_index = _detect_outliers(std, threshold=0.02 * np.mean(results_big[:, 2]))
    return np.delete(results_big, out_index, axis=0)


def _classify_small_circles(results_small, height):
    """
    对小圆检测结果分级：

    Returns
    -------
    filtered_small : np.ndarray  筛选后的小圆（可能为空）
    small_key      : int
        0 → 正常（2 个小圆）
        1 → 只有 1 个小圆
        2 → 超过 2 个（已内部降级为 0 或 3）
        3 → 完全失效
    """
    if results_small.shape == (0,):
        return results_small, 3

    filtered = results_small[results_small[:, 1] >= height / 2]

    if filtered.shape[0] == 2:
        return filtered, 0

    if filtered.shape[0] == 1:
        return filtered, 1

    # 超过 2 个：按 y 取最近两点
    mean_y = np.mean(filtered[:, 1])
    min_two_indices = np.argsort(np.abs(filtered[:, 1] - mean_y))[:2]
    filtered = filtered[min_two_indices]

    # 检查两侧是否各有一个
    if np.count_nonzero(filtered[:, -1] == 0) in (0, 2):
        return filtered, 3
    return filtered, 0


def _filter_detections(results_big, results_small, height):
    """
    综合大圆与小圆的检测结果，输出清洗后的数据集及容差标志。

    Returns
    -------
    filtered_big   : np.ndarray  清洗后的大圆数组（可能为空列表）
    filtered_small : np.ndarray  清洗后的小圆数组
    high_tol       : int         0 → 正常容差；1 → 高容差
    """
    high_tol = 0
    big_key = int(results_big.shape == (0,))  # 1 → 大圆完全未检测到

    if big_key:
        logger.warning("未识别到大圆，自动进入高容差模式")

    filtered_small, small_key = _classify_small_circles(results_small, height)

    # ── 小圆正常：精确模式 ─────────────────────────────────
    if small_key == 0:
        r_refer = np.abs(filtered_small[1, 0] - filtered_small[0, 0]) / _SMALL_DX
        if big_key:
            logger.warning("仅使用小圆进入最小二乘")
            filtered_big = []
        else:
            filtered_big = _filter_big_circles_by_radius_and_position(
                results_big, r_refer
            )
        return filtered_big, filtered_small, high_tol

    # ── 小圆完全失效 ──────────────────────────────────────
    if small_key == 3:
        logger.warning("小圆识别异常，将进入高容差模式")
        high_tol = 1
        if big_key:
            raise RuntimeError("捕捉效果差, 将启用备用参数进行捕捉")
        if results_big.shape[0] <= 2:
            logger.warning("大圆数量不足，直接进入最小二乘")
            return results_big, filtered_small, high_tol
        filtered_big = _filter_big_circles_by_outlier(results_big)
        return filtered_big, filtered_small, high_tol

    # ── 只有 1 个小圆 ─────────────────────────────────────
    if small_key == 1:
        high_tol = 1
        if big_key:
            logger.warning("警告：大圆数量不足，仅以唯一小圆进入框架创建")
            raise RuntimeError("捕捉效果差, 将启用备用参数进行捕捉")
        if results_big.shape[0] <= 2:
            logger.warning("警告：大圆数量不足，直接进入最小二乘")
            return results_big, filtered_small, high_tol
        filtered_big = _filter_big_circles_by_outlier(results_big)
        return filtered_big, filtered_small, high_tol


# ──────────────────────────────────────────────────────────
# 最小二乘拟合
# ──────────────────────────────────────────────────────────


def _residuals(params, large_circles, small_circles):
    """
    目标函数：返回大圆与小圆检测坐标相对于几何模型的残差向量。

    Parameters
    ----------
    params         : [cx, cy, r]  框架原点与单位半径
    large_circles  : np.ndarray, shape (N, 4)
    small_circles  : np.ndarray, shape (M, 4)

    Returns
    -------
    residuals : list[float]
    """
    cx, cy, r = params
    res = []

    for x, y, radius, n in large_circles:
        pred_x = (2 * n + 1) * _K * r + cx + (_M * r if n in [3, 4, 5] else 0)
        pred_y = -_K * r + cy
        res += [x - pred_x, y - pred_y]

    for x, y, radius, n in small_circles:
        if n == 0:
            pred_x, pred_y = 0.44576523 * r + cx, -0.14858841 * r + cy
        else:  # n == 5
            pred_x, pred_y = 16.745914 * r + cx, -0.14858841 * r + cy
        res += [x - pred_x, y - pred_y]

    return res


def _fit_frame_params(filtered_big, filtered_small):
    """
    以最小二乘法拟合框架几何参数 (cx, cy, r)。

    Returns
    -------
    cx, cy, r : float
    """
    result = _solve_least_squares(
        _residuals,
        x0=[0, 200, 60],
        args=(filtered_big, filtered_small),
    )
    return result.x  # [cx, cy, r]


# ──────────────────────────────────────────────────────────
# 框架坐标生成
# ──────────────────────────────────────────────────────────


def _build_frame_boxes(cx, cy, r, high_tol=False):
    """
    根据拟合参数生成头像框（avatar）与数字框（nums）的像素坐标。

    Parameters
    ----------
    cx, cy, r : float  拟合结果
    high_tol  : bool   False → 容差 2.5%；True → 容差 17%

    Returns
    -------
    avatar : np.ndarray, shape (6, 4)  [x1, y1, x2, y2]
    nums   : np.ndarray, shape (6, 4)  [x1, y1, x2, y2]
    """
    k, m = _K, _M
    t = 0.17 if high_tol else 0.025

    nb = 0.9  # nums outer bias
    nbi = 0.3  # nums inner bias
    ny = cy - 0.5 * r

    avatar = np.round(
        np.array(
            [
                [
                    cx - r * t,
                    cy + r * t,
                    cx + 2 * k * r + r * t,
                    cy - 2 * k * r - r * t,
                ],
                [
                    cx + 2 * k * r - r * t,
                    cy - 2 * k * r - r * t,
                    cx + 4 * k * r + r * t,
                    cy + r * t,
                ],
                [
                    cx + 4 * k * r - r * t,
                    cy + r * t,
                    cx + 6 * k * r + r * t,
                    cy - 2 * k * r - r * t,
                ],
                [
                    cx + (6 * k + m) * r - r * t,
                    cy + r * t,
                    cx + (8 * k + m) * r + r * t,
                    cy - 2 * k * r - r * t,
                ],
                [
                    cx + (8 * k + m) * r - r * t,
                    cy - 2 * k * r - r * t,
                    cx + (10 * k + m) * r + r * t,
                    cy + r * t,
                ],
                [
                    cx + (10 * k + m) * r - r * t,
                    cy + r * t,
                    cx + (12 * k + m) * r + r * t,
                    cy - 2 * k * r - r * t,
                ],
            ]
        )
    ).astype("int")

    nums = np.round(
        np.array(
            [
                [
                    cx + (nb + 0 * k) * r - r * t,
                    cy + r * t,
                    cx + (nbi + 2 * k) * r + r * t,
                    ny,
                ],
                [
                    cx + (nb + 2 * k) * r - r * t,
                    ny,
                    cx + (nbi + 4 * k) * r + r * t,
                    cy + r * t,
                ],
                [
                    cx + (nb + 4 * k) * r - r * t,
                    cy + r * t,
                    cx + (nbi + 6 * k) * r + r * t,
                    ny,
                ],
                [
                    cx + (-nbi + 6 * k + m) * r - r * t,
                    cy + r * t,
                    cx + (-nb + 8 * k + m) * r + r * t,
                    ny,
                ],
                [
                    cx + (-nbi + 8 * k + m) * r - r * t,
                    ny,
                    cx + (-nb + 10 * k + m) * r + r * t,
                    cy + r * t,
                ],
                [
                    cx + (-nbi + 10 * k + m) * r - r * t,
                    cy + r * t,
                    cx + (-nb + 12 * k + m) * r + r * t,
                    ny,
                ],
            ]
        )
    ).astype("int")

    return avatar, nums


# ──────────────────────────────────────────────────────────
# 单次完整检测流程（内部）
# ──────────────────────────────────────────────────────────


def _run_detection(image, blur, spare=0, p1=21, p2=28):
    """
    执行一次完整的圆检测 → 数据清洗流程。

    Returns
    -------
    filtered_big   : np.ndarray
    filtered_small : np.ndarray
    high_tol       : int
    """
    height, width, _ = image.shape
    big_min, big_max, small_min, small_max = _get_resolution_radius_range(image)

    crop_blur, crop_thresh, x_ratio, x_ratio_small = _preprocess(
        image, blur=blur, spare=spare
    )

    results_big = _detect_big_circles(
        crop_blur, x_ratio, big_min, big_max, width, p1=p1, p2=p2
    )
    results_small = _detect_small_circles(
        crop_thresh, x_ratio_small, x_ratio, small_min, small_max, width
    )

    return _filter_detections(results_big, results_small, height)


# ──────────────────────────────────────────────────────────
# 公开主接口
# ──────────────────────────────────────────────────────────


def find_monster_zone(image):
    """
    从图像中提取头像框与数字框的归一化坐标。

    流程
    ────
    1. 主参数圆检测（blur=11, spare=0）
    2. 若失败，自动切换备用参数（blur=7, spare=1）重试
    3. 最小二乘拟合框架几何参数
    4. 生成头像框与数字框坐标，归一化返回

    Parameters
    ----------
    image : np.ndarray  BGR 格式原始图像

    Returns
    -------
    d_avatar : np.ndarray, shape (6, 4)  归一化头像框坐标 [x1,y1,x2,y2]
    d_nums   : np.ndarray, shape (6, 4)  归一化数字框坐标 [x1,y1,x2,y2]
    """
    height, width, _ = image.shape

    # ── 第一次尝试（主参数）───────────────────────────────
    try:
        filtered_big, filtered_small, high_tol = _run_detection(
            image, blur=11, spare=0, p1=21, p2=28
        )
    except (IndexError, RuntimeError):
        # ── 第二次尝试（备用参数）─────────────────────────
        try:
            filtered_big, filtered_small, high_tol = _run_detection(
                image, blur=7, spare=1, p1=18, p2=24
            )
        except (IndexError, RuntimeError):
            logger.error("备用参数捕捉失败！请重新框选试试")
            return None, None

    # ── 最小二乘拟合 ──────────────────────────────────────
    cx, cy, r = _fit_frame_params(filtered_big, filtered_small)

    # ── 生成框坐标并归一化 ────────────────────────────────
    avatar, nums = _build_frame_boxes(cx, cy, r, high_tol=bool(high_tol))

    divisors = np.array([width, height, width, height])
    d_avatar = avatar / divisors
    d_nums = nums / divisors

    return d_avatar, d_nums


# ──────────────────────────────────────────────────────────
# 调试入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    image = cv2.imread("images/tmp/zone1.png")
    height, width, _ = image.shape

    # 独立运行一次检测并可视化
    big_min, big_max, small_min, small_max = _get_resolution_radius_range(image)
    crop_blur, crop_thresh, x_ratio, x_ratio_small = _preprocess(image, blur=11)

    results_big = _detect_big_circles(
        crop_blur, x_ratio, big_min, big_max, width, p1=21, p2=32
    )
    results_small = _detect_small_circles(
        crop_thresh, x_ratio_small, x_ratio, small_min, small_max, width
    )

    for i in np.round(results_big).astype("int"):
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    for i in np.round(results_small).astype("int"):
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    d_avatar, d_nums = find_monster_zone(image)
    if d_avatar is not None:
        divisors = np.array([width, height, width, height])
        for x1, y1, x2, y2 in np.round(d_avatar * divisors).astype("int"):
            cv2.rectangle(image, (x1, y1), (x2, y2), (225, 0, 225), 2)
        for x1, y1, x2, y2 in np.round(d_nums * divisors).astype("int"):
            cv2.rectangle(image, (x1, y1), (x2, y2), (225, 225, 0), 2)

    cv2.imshow("Detected Frames", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
