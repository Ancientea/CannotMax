"""Unit tests for monster bar detection (find_monster_zone).

Tests using images/process/*.png as full screenshot inputs.
Note: These tests require actual game screenshots from the monster selection screen.
If images show "circle not detected" warnings, the images may not contain the monster bar.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path

from src.cannotmax.utils.find_monster_zone import cutFrame


class TestMonsterBarDetection:
    """Test monster bar detection with real screenshots."""
    
    @pytest.fixture
    def test_images(self):
        """Load test images from images/process/ directory."""
        base_path = Path("images/process")
        images = {}
        for i in [3, 4, 5]:
            img_path = base_path / f"{i}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    images[i] = img
        return images
    
    def test_cutframe_image_3(self, test_images):
        """Test monster bar detection on image 3.png."""
        if 3 not in test_images:
            pytest.skip("Image 3.png not found")
        
        screenshot = test_images[3]
        height, width = screenshot.shape[:2]
        
        # Run detection
        avatar_roi, nums_roi = cutFrame(screenshot)
        
        # Verify output is normalized (0-1 range)
        assert avatar_roi is not None
        assert nums_roi is not None
        assert len(avatar_roi) == 6  # 6 monster zones
        assert len(nums_roi) == 6   # 6 number zones
        
        # Check that coordinates are in valid range [0, 1]
        for zone in avatar_roi:
            x1, y1, x2, y2 = zone
            assert 0 <= x1 <= 1 and 0 <= y1 <= 1
            assert 0 <= x2 <= 1 and 0 <= y2 <= 1
            assert x1 < x2 and y1 < y2
    
    def test_cutframe_image_4(self, test_images):
        """Test monster bar detection on image 4.png."""
        if 4 not in test_images:
            pytest.skip("Image 4.png not found")
        
        screenshot = test_images[4]
        avatar_roi, nums_roi = cutFrame(screenshot)
        
        assert avatar_roi is not None
        assert nums_roi is not None
        assert len(avatar_roi) == 6
        assert len(nums_roi) == 6
    
    def test_cutframe_image_5(self, test_images):
        """Test monster bar detection on image 5.png."""
        if 5 not in test_images:
            pytest.skip("Image 5.png not found")
        
        screenshot = test_images[5]
        avatar_roi, nums_roi = cutFrame(screenshot)
        
        assert avatar_roi is not None
        assert nums_roi is not None
        assert len(avatar_roi) == 6
        assert len(nums_roi) == 6
    
    def test_cutframe_returns_normalized_coords(self, test_images):
        """Test that cutFrame returns normalized coordinates."""
        if not test_images:
            pytest.skip("No test images available")
        
        for idx, screenshot in test_images.items():
            height, width = screenshot.shape[:2]
            avatar_roi, nums_roi = cutFrame(screenshot)
            
            # Convert normalized coords to pixels and verify they fit in image
            for i, zone in enumerate(avatar_roi):
                x1, y1, x2, y2 = zone
                px1, py1 = int(x1 * width), int(y1 * height)
                px2, py2 = int(x2 * width), int(y2 * height)
                
                assert 0 <= px1 < width
                assert 0 <= py1 < height
                assert 0 <= px2 < width
                assert 0 <= py2 < height
                assert px1 < px2
                assert py1 < py2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
