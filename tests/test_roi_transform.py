"""Tests for roi_transform utility."""

import numpy as np

from cannotmax.utils.roi_transform import ensure_valid_coords, transform_coords


class TestTransformCoords:
    """Test transform_coords function."""

    def test_single_coord_transform(self):
        """通用变换：x_new = (x - x_offset) * scale_x"""
        coords = np.array([0.1, 0.2, 0.3, 0.4])
        result = transform_coords(
            coords,
            x_offset=0,
            y_offset=0,
            scale_x=100,
            scale_y=50,
            clamp=False,
        )
        expected = np.array([10, 10, 30, 20])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_coord_inverse_transform(self):
        """反向转换：像素→归一化"""
        coords = np.array([200, 150, 400, 350])
        result = transform_coords(
            coords,
            x_offset=100,
            y_offset=50,
            scale_x=1 / 300,
            scale_y=1 / 300,
            clamp=False,
        )
        expected = np.array(
            [(200 - 100) / 300, (150 - 50) / 300, (400 - 100) / 300, (350 - 50) / 300]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_coords(self):
        """批量处理多个坐标"""
        coords = np.array([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
        result = transform_coords(
            coords,
            x_offset=0,
            y_offset=0,
            scale_x=100,
            scale_y=100,
            clamp=False,
        )
        expected = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        np.testing.assert_array_equal(result, expected)

    def test_clamp_to_valid_range(self):
        """Clamp 到 [0,1] 范围"""
        coords = np.array([-0.1, 1.5, 0.5, 0.5])
        result = transform_coords(
            coords,
            x_offset=0,
            y_offset=0,
            scale_x=1,
            scale_y=1,
            clamp=True,
        )
        expected = np.array([0.0, 1.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_clamp_disabled(self):
        """关闭 clamp 时不限制范围"""
        coords = np.array([-0.1, 1.5, 0.5, 0.5])
        result = transform_coords(
            coords,
            x_offset=0,
            y_offset=0,
            scale_x=1,
            scale_y=1,
            clamp=False,
        )
        expected = np.array([-0.1, 1.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_roi_to_screen_transform(self):
        """ROI 归一化坐标 → 全屏像素坐标"""
        roi_x, roi_y, roi_w, roi_h = 100, 50, 800, 400
        d_avatar = np.array([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
        result = transform_coords(
            d_avatar,
            x_offset=-roi_x / roi_w,
            y_offset=-roi_y / roi_h,
            scale_x=roi_w,
            scale_y=roi_h,
            clamp=False,
        )
        expected = np.array(
            [
                [100 + 0.0 * 800, 50 + 0.0 * 400, 100 + 0.1 * 800, 50 + 0.1 * 400],
                [100 + 0.5 * 800, 50 + 0.5 * 400, 100 + 0.6 * 800, 50 + 0.6 * 400],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_screen_to_normalized_transform(self):
        """全屏像素坐标 → 归一化坐标"""
        screen_coords = np.array([[200, 100, 300, 200], [500, 300, 600, 400]])
        crop_x, crop_y, crop_w, crop_h = 100, 50, 500, 350
        result = transform_coords(
            screen_coords,
            x_offset=crop_x,
            y_offset=crop_y,
            scale_x=1.0 / crop_w,
            scale_y=1.0 / crop_h,
            clamp=False,
        )
        expected = np.array(
            [
                [
                    (200 - 100) / 500,
                    (100 - 50) / 350,
                    (300 - 100) / 500,
                    (200 - 50) / 350,
                ],
                [
                    (500 - 100) / 500,
                    (300 - 50) / 350,
                    (600 - 100) / 500,
                    (400 - 50) / 350,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)


class TestEnsureValidCoords:
    """Test ensure_valid_coords function."""

    def test_swap_x_coords(self):
        """交换 x1 > x2 的情况"""
        coords = np.array([0.8, 0.2, 0.3, 0.6])
        result = ensure_valid_coords(coords)
        expected = np.array([0.3, 0.2, 0.8, 0.6])
        np.testing.assert_array_almost_equal(result, expected)

    def test_swap_y_coords(self):
        """交换 y1 > y2 的情况"""
        coords = np.array([0.2, 0.8, 0.5, 0.3])
        result = ensure_valid_coords(coords)
        expected = np.array([0.2, 0.3, 0.5, 0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_both_swapped(self):
        """x 和 y 都交换"""
        coords = np.array([0.8, 0.9, 0.2, 0.1])
        result = ensure_valid_coords(coords)
        expected = np.array([0.2, 0.1, 0.8, 0.9])
        np.testing.assert_array_almost_equal(result, expected)

    def test_already_valid(self):
        """已经是有效坐标"""
        coords = np.array([0.2, 0.3, 0.8, 0.9])
        result = ensure_valid_coords(coords)
        np.testing.assert_array_almost_equal(result, coords)

    def test_batch_coords(self):
        """批量处理"""
        coords = np.array([[0.8, 0.9, 0.2, 0.1], [0.1, 0.8, 0.5, 0.3]])
        result = ensure_valid_coords(coords)
        expected = np.array([[0.2, 0.1, 0.8, 0.9], [0.1, 0.3, 0.5, 0.8]])
        np.testing.assert_array_almost_equal(result, expected)
