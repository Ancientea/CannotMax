"""比例描述的 ROI 在坐标系间的快速变换工具。

支持批量处理多个 ROI 坐标的归一化变换。
"""

import numpy as np


def transform_coords(
    coords: np.ndarray,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    clamp: bool = True,
) -> np.ndarray:
    """通用坐标变换：平移 + 缩放。

    变换公式：
        正向: x_new = x_offset + x * scale_x  (归一化→像素)
        反向: x_new = (x - x_offset) * scale_x  (像素→归一化)

    Args:
        coords: 坐标数组，shape (N, 4) 或 (4,)，每行 [x1, y1, x2, y2]
        x_offset: x 方向偏移量
        y_offset: y 方向偏移量
        scale_x: x 方向缩放比
        scale_y: y 方向缩放比
        clamp: 是否将结果 clamp 到 [0, 1] 范围

    Returns:
        变换后的坐标，shape 与输入相同
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    result = np.zeros_like(coords)
    result[:, 0] = (coords[:, 0] - x_offset) * scale_x
    result[:, 1] = (coords[:, 1] - y_offset) * scale_y
    result[:, 2] = (coords[:, 2] - x_offset) * scale_x
    result[:, 3] = (coords[:, 3] - y_offset) * scale_y

    if clamp:
        result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
        result[:, 1] = np.clip(result[:, 1], 0.0, 1.0)
        result[:, 2] = np.clip(result[:, 2], 0.0, 1.0)
        result[:, 3] = np.clip(result[:, 3], 0.0, 1.0)

    return result.squeeze()


def ensure_valid_coords(coords: np.ndarray) -> np.ndarray:
    """确保坐标顺序有效：x1 <= x2, y1 <= y2。

    Args:
        coords: 坐标，shape (N, 4) 或 (4,)，每行 [x1, y1, x2, y2]

    Returns:
        调整后的坐标
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    result = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        x1, y1, x2, y2 = coords[i]
        result[i, 0] = min(x1, x2)
        result[i, 1] = min(y1, y2)
        result[i, 2] = max(x1, x2)
        result[i, 3] = max(y1, y2)

    return result.squeeze()
