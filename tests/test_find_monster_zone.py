"""Tests for find_monster_zone on user-cropped WIN mode screenshots."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cannotmax.core.recognize import RecognizeMonster
from src.cannotmax.utils.find_monster_zone import find_monster_zone


def _load_test_image(filename: str) -> np.ndarray | None:
    path = Path("images/tests", filename)
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img if img is not None and img.size > 0 else None


class TestFindMonsterZone:
    """Test find_monster_zone on pre-cropped monster bar images."""

    @pytest.mark.parametrize("index", [1, 2, 3, 4])
    def test_returns_valid_coords(self, index):
        filename = f"win_cropped_screenshot_{index}.png"
        img = _load_test_image(filename)
        if img is None:
            pytest.skip(f"{filename} not found")
        d_avatar, d_nums = find_monster_zone(img)
        assert d_avatar is not None, f"{filename}: d_avatar is None"
        assert d_nums is not None, f"{filename}: d_nums is None"
        assert d_avatar.shape == (6, 4), f"{filename}: shape={d_avatar.shape}"
        assert d_nums.shape == (6, 4), f"{filename}: shape={d_nums.shape}"


class TestWinRecognition:
    """Test full process_regions with WIN mode on pre-cropped images."""

    @pytest.mark.parametrize("index", [1, 2, 3, 4])
    def test_process_regions_returns_six_results(self, index):
        filename = f"win_cropped_screenshot_{index}.png"
        img = _load_test_image(filename)
        if img is None:
            pytest.skip(f"{filename} not found")
        recognizer = RecognizeMonster(crop_ratio=((0.0, 0.0), (1.0, 1.0)))
        results = recognizer.process_regions(img, auto_fallback=False)
        assert isinstance(results, list)
        assert len(results) == 6, f"Expected 6 results, got {len(results)}"

    @pytest.mark.parametrize(
        "index,expected",
        [
            (1, {(46, "6"), (42, "28")}),
            (2, {(46, "6"), (42, "28")}),
            (3, {(44, "18"), (5, "4"), (46, "6"), (3, "5")}),
            (4, {(44, "18"), (5, "4"), (46, "6"), (3, "5")}),
        ],
    )
    def test_correct_numbers_detected(self, index, expected):
        filename = f"win_cropped_screenshot_{index}.png"
        img = _load_test_image(filename)
        if img is None:
            pytest.skip(f"{filename} not found")
        recognizer = RecognizeMonster(crop_ratio=((0.0, 0.0), (1.0, 1.0)))
        results = recognizer.process_regions(img, auto_fallback=False)

        detected = set()
        for r in results:
            if "error" not in r and r["number"] != "N/A":
                detected.add((r["matched_id"], r["number"]))

        assert detected == expected, f"WIN {index}: expected {expected}, got {detected}"
