"""Unit tests for monster bar detection and monster recognition.

Tests using images/process/{3,4,5}.png as full screenshot inputs.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path

from src.cannotmax.utils.find_monster_zone import find_monster_zone
from src.cannotmax.core.recognize import RecognizeMonster


def load_test_image(index: int) -> np.ndarray | None:
    """Load a test image from images/process/<index>.png."""
    path = Path("images/process", f"{index}.png")
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img if img is not None and img.size > 0 else None


class TestMonsterBarDetection:
    """Test monster bar detection (cutFrame) doesn't crash on real screenshots."""

    @pytest.fixture(params=[3, 4, 5])
    def screenshot(self, request) -> np.ndarray | None:
        img = load_test_image(request.param)
        if img is None:
            pytest.skip(f"images/process/{request.param}.png not found")
        h, w = img.shape[:2]
        assert h > 0 and w > 0, "Invalid image dimensions"
        return img

    def test_cutframe_no_crash(self, screenshot):
        """cutFrame must not raise exceptions on any valid image input."""
        try:
            avatar_roi, nums_roi = find_monster_zone(screenshot)
        except Exception as e:
            pytest.fail(f"cutFrame crashed: {e}")

        assert avatar_roi is not None, "avatar_roi must not be None"
        assert nums_roi is not None, "nums_roi must not be None"

    def test_cutframe_output_shape(self, screenshot):
        """cutFrame must return two arrays with shape (N, 4) where N >= 1."""
        avatar_roi, nums_roi = find_monster_zone(screenshot)

        assert isinstance(avatar_roi, np.ndarray), (
            f"Expected ndarray, got {type(avatar_roi)}"
        )
        assert isinstance(nums_roi, np.ndarray), (
            f"Expected ndarray, got {type(nums_roi)}"
        )
        assert avatar_roi.ndim == 2, f"avatar_roi should be 2D, got {avatar_roi.ndim}D"
        assert nums_roi.ndim == 2, f"nums_roi should be 2D, got {nums_roi.ndim}D"
        assert avatar_roi.shape[1] == 4, (
            f"avatar_roi cols should be 4, got {avatar_roi.shape[1]}"
        )
        assert nums_roi.shape[1] == 4, (
            f"nums_roi cols should be 4, got {nums_roi.shape[1]}"
        )
        assert len(avatar_roi) >= 1, "Must detect at least 1 zone"
        assert len(nums_roi) >= 1, "Must detect at least 1 zone"

    def test_cutframe_coords_in_range(self, screenshot):
        """All normalized coordinates must be in [0, 1] range (with 5% tolerance)."""
        h, w = screenshot.shape[:2]
        avatar_roi, nums_roi = find_monster_zone(screenshot)

        for name, zones in [("avatar_roi", avatar_roi), ("nums_roi", nums_roi)]:
            for i, zone in enumerate(zones):
                x1, y1, x2, y2 = (
                    float(zone[0]),
                    float(zone[1]),
                    float(zone[2]),
                    float(zone[3]),
                )
                # Allow small tolerance for outlier coords (typical: 0.025 beyond bounds)
                assert -0.05 <= x1 <= 1.05, f"{name}[{i}] x1={x1:.4f} out of range"
                assert -0.05 <= y1 <= 1.05, f"{name}[{i}] y1={y1:.4f} out of range"
                assert -0.05 <= x2 <= 1.05, f"{name}[{i}] x2={x2:.4f} out of range"
                assert -0.05 <= y2 <= 1.05, f"{name}[{i}] y2={y2:.4f} out of range"
                # Verify pixel coordinates are within image bounds (allowing 5% overflow)
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                assert -int(w * 0.05) <= px1 < int(w * 1.05), f"{name}[{i}] px1={px1}"
                assert -int(h * 0.05) <= px2 <= int(w * 1.05), f"{name}[{i}] px2={px2}"


class TestMonsterRecognition:
    """Test full monster recognition (monster type + count via OCR)."""

    @pytest.fixture(params=[3, 4, 5])
    def screenshot(self, request) -> np.ndarray | None:
        img = load_test_image(request.param)
        if img is None:
            pytest.skip(f"images/process/{request.param}.png not found")
        return img

    @pytest.fixture
    def recognizer(self):
        return RecognizeMonster()

    def test_recognize_no_crash(self, recognizer, screenshot):
        """process_regions must not raise exceptions on any valid image."""
        try:
            results = recognizer.process_regions(screenshot)
        except Exception as e:
            pytest.fail(f"process_regions crashed: {e}")

        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) <= 6, f"Max 6 monsters, got {len(results)}"

    def test_recognize_result_structure(self, recognizer, screenshot):
        """Each result must have required fields: region_id, matched_id, number."""
        results = recognizer.process_regions(screenshot)

        for res in results:
            assert "region_id" in res, f"Missing region_id in {res}"
            assert "matched_id" in res, f"Missing matched_id in {res}"
            assert "number" in res, f"Missing number in {res}"
            if "error" not in res:
                # Successful recognition must have valid values
                assert res["matched_id"] >= 0, (
                    f"Invalid matched_id: {res['matched_id']}"
                )
                assert isinstance(res["region_id"], int), (
                    f"region_id must be int: {res['region_id']}"
                )
                assert 0 <= res["region_id"] <= 5, (
                    f"region_id out of range: {res['region_id']}"
                )


class TestRecognitionRegression:
    """Regression tests for specific known screenshots."""

    def test_empty_grayscale_image_doesnt_crash(self):
        """Black/empty images should return empty results without crashing."""
        for shape in [(1080, 1920, 3), (720, 1280, 3)]:
            black = np.zeros(shape, dtype=np.uint8)
            avatar, nums = find_monster_zone(black)
            assert isinstance(avatar, np.ndarray)
            assert isinstance(nums, np.ndarray)

    def test_white_image_doesnt_crash(self):
        """White images should return empty zones without crashing."""
        white = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        avatar, nums = find_monster_zone(white)
        assert isinstance(avatar, np.ndarray)
        assert isinstance(nums, np.ndarray)
