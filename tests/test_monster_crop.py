"""Tests for RecognizeMonster crop_ratio and process_regions with auto_fallback."""
import pytest
import cv2
import numpy as np
from src.cannotmax.core.recognize import RecognizeMonster, ROINotSelectedError
from src.cannotmax.config.constants import DEFAULT_CROP_RATIO


class TestRecognizeMonsterCropRatio:
    """Test crop_ratio handling."""

    def test_init_with_none_crop_ratio(self):
        recognizer = RecognizeMonster()
        assert recognizer.crop_ratio is None

    def test_init_with_custom_crop_ratio(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        assert recognizer.crop_ratio == ratio

    def test_resolve_fallback_returns_default(self):
        recognizer = RecognizeMonster(crop_ratio=None)
        result = recognizer._resolve_crop_ratio(auto_fallback=True)
        assert result == DEFAULT_CROP_RATIO

    def test_resolve_fallback_returns_custom(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        result = recognizer._resolve_crop_ratio(auto_fallback=True)
        assert result == ratio

    def test_resolve_no_fallback_raises_when_none(self):
        recognizer = RecognizeMonster(crop_ratio=None)
        with pytest.raises(ROINotSelectedError):
            recognizer._resolve_crop_ratio(auto_fallback=False)

    def test_resolve_no_fallback_returns_custom(self):
        ratio = ((0.1, 0.2), (0.8, 0.9))
        recognizer = RecognizeMonster(crop_ratio=ratio)
        result = recognizer._resolve_crop_ratio(auto_fallback=False)
        assert result == ratio


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
        cropped = recognizer._crop_by_ratio(img, DEFAULT_CROP_RATIO)
        expected_h = int(0.9510 * 1080) - int(0.8410 * 1080)
        expected_w = int(0.7542 * 1920) - int(0.2464 * 1920)
        assert cropped.shape[0] == expected_h
        assert cropped.shape[1] == expected_w


class TestProcessRegions:
    """Test process_regions with auto_fallback flag."""

    def test_process_regions_with_fallback_no_crash(self):
        """ADB/PC: auto_fallback=True, should not crash with black image."""
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = recognizer.process_regions(img, auto_fallback=True)
        assert isinstance(results, list)
        assert 0 <= len(results) <= 6

    def test_process_regions_no_fallback_raises_without_roi(self):
        """WIN: auto_fallback=False, crop_ratio=None should raise."""
        recognizer = RecognizeMonster()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(ROINotSelectedError):
            recognizer.process_regions(img, auto_fallback=False)

    def test_process_regions_no_fallback_with_roi_no_crash(self):
        """WIN: auto_fallback=False with crop_ratio set should not crash."""
        recognizer = RecognizeMonster(crop_ratio=((0.25, 0.80), (0.75, 0.95)))
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = recognizer.process_regions(img, auto_fallback=False)
        assert isinstance(results, list)
        assert 0 <= len(results) <= 6
