"""Tests for RecognizeMonster crop_ratio and process_regions."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from cannotmax.config.settings import RECOGNITION_PARAMS
from cannotmax.core.recognize import RecognizeMonster

_ADB_CROP_RATIO = RECOGNITION_PARAMS["ADB"]["crop_regions"]
_PC_CROP_RATIO = RECOGNITION_PARAMS["PC"]["crop_regions"]


class TestRecognizeMonsterCropRatio:
    """Test crop_ratio handling."""

    def test_init_with_none_crop_ratio(self):
        recognizer = RecognizeMonster()
        assert recognizer.crop_ratio is None

    def test_init_with_custom_crop_ratio(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        assert recognizer.crop_ratio == ratio


class TestCropByRatio:
    """Test _crop_by_ratio method."""

    def test_crop_by_ratio_dimensions(self):
        recognizer = RecognizeMonster()
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        ratio = ((0.25, 0.50), (0.75, 0.80))
        cropped = recognizer._crop_by_ratio(img, ratio)
        assert cropped.shape == (300, 1000, 3)

    def test_crop_by_ratio_default(self):
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cropped = recognizer._crop_by_ratio(img, _ADB_CROP_RATIO)
        expected_h = int(0.9510 * 1080) - int(0.8410 * 1080)
        expected_w = int(0.7542 * 1920) - int(0.2464 * 1920)
        assert cropped.shape[0] == expected_h
        assert cropped.shape[1] == expected_w


class TestProcessRegions:
    """Test process_regions method with dummy image."""

    def test_process_regions_with_roi_no_crash(self):
        recognizer = RecognizeMonster(crop_ratio=((0.25, 0.80), (0.75, 0.95)))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = recognizer.process_regions(img)
        assert isinstance(results, list)
        assert 0 <= len(results) <= 6


class TestAdbRecognitionAccuracy:
    """Test ADB mode recognition against known screenshots."""

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_process_regions_returns_six_results(self, index):
        path = Path("images/tests", f"adb_original_screenshot_{index}.png")
        if not path.exists():
            pytest.skip(f"adb_original_screenshot_{index}.png not found")
        img = cv2.imread(str(path))
        if img is None:
            pytest.skip(f"Cannot read adb_original_screenshot_{index}.png")
        recognizer = RecognizeMonster(crop_ratio=_ADB_CROP_RATIO)
        results = recognizer.process_regions(img, mode="ADB")
        assert isinstance(results, list)
        assert len(results) == 6

    @pytest.mark.parametrize("index", [1, 2, 3])
    def test_correct_numbers_detected(self, index):
        path = Path("images/tests", f"adb_original_screenshot_{index}.png")
        if not path.exists():
            pytest.skip(f"adb_original_screenshot_{index}.png not found")
        img = cv2.imread(str(path))
        if img is None:
            pytest.skip(f"Cannot read adb_original_screenshot_{index}.png")
        recognizer = RecognizeMonster(crop_ratio=_ADB_CROP_RATIO)
        results = recognizer.process_regions(img, mode="ADB")

        detected = set()
        for r in results:
            if "error" not in r and r["number"] != "N/A":
                detected.add((r["matched_id"], r["number"]))

        expected = {
            1: {(49, "4"), (31, "18")},
            2: {(43, "9"), (32, "9"), (40, "3")},
            3: {(53, "5"), (24, "3"), (44, "22"), (10, "29"), (37, "2"), (30, "2")},
        }
        exp = expected[index]
        assert detected == exp, f"ADB {index}: expected {exp}, got {detected}"


class TestPcRecognitionAccuracy:
    """Test PC mode recognition against known screenshots."""

    @pytest.mark.parametrize("index", [1, 2])
    def test_pc_process_regions_returns_six_results(self, index):
        path = Path("images/tests", f"pc_original_screenshot_{index}.png")
        if not path.exists():
            pytest.skip(f"pc_original_screenshot_{index}.png not found")
        img = cv2.imread(str(path))
        if img is None:
            pytest.skip(f"Cannot read pc_original_screenshot_{index}.png")
        recognizer = RecognizeMonster(crop_ratio=_PC_CROP_RATIO)
        results = recognizer.process_regions(img, mode="PC")
        assert isinstance(results, list)
        assert len(results) == 6

    @pytest.mark.parametrize("index", [1])
    def test_pc_correct_numbers_detected(self, index):
        path = Path("images/tests", f"pc_original_screenshot_{index}.png")
        if not path.exists():
            pytest.skip(f"pc_original_screenshot_{index}.png not found")
        img = cv2.imread(str(path))
        if img is None:
            pytest.skip(f"Cannot read pc_original_screenshot_{index}.png")
        recognizer = RecognizeMonster(crop_ratio=_PC_CROP_RATIO)
        results = recognizer.process_regions(img, mode="PC")

        detected = set()
        for r in results:
            if "error" not in r and r["number"] != "N/A":
                detected.add((r["matched_id"], r["number"]))

        expected = {
            1: {(37, "1"), (18, "5")},
            2: {(12, "4"), (8, "14"), (30, "3"), (36, "5"), (56, "6"), (5, "4")},
        }
        exp = expected[index]
        assert detected == exp, f"PC {index}: expected {exp}, got {detected}"
