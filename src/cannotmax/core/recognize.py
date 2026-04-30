"""Monster recognition via template matching and OCR.

Identifies monsters in 6 battlefield regions using:
- Template matching against reference images
- OCR for monster count extraction (RapidOCR)
- Debug mode with intermediate image saving

Regions: 3 left + 3 right, each containing monster type + count

Usage:
    recognizer = RecognizeMonster()
    results = recognizer.process_regions(screenshot)
    # results: [{region_id, matched_id, number, confidence}, ...]
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from rapidocr import RapidOCR, EngineType

from ..config import (
    MONSTER_DATA, 
    MONSTER_IMAGES, 
    MONSTER_COUNT,
    DEBUG_MODE,
)
from ..config.paths import TMP_IMAGES_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def get_rapidocr_engine(prefer_gpu=False):
    """Get RapidOCR engine with optional GPU support."""
    try:
        if prefer_gpu:
            import torch
            if torch.cuda.is_available():
                return RapidOCR(
                    params={
                        "Det.engine_type": EngineType.TORCH,
                        "Cls.engine_type": EngineType.TORCH,
                        "Rec.engine_type": EngineType.TORCH,
                        "EngineConfig.torch.use_cuda": True,
                        "EngineConfig.torch.gpu_id": 0,
                    }
                )
    except ImportError:
        logger.warning("torch 库未安装，使用 onnxruntime")
    return RapidOCR()


def load_ref_images():
    """Load reference monster images."""
    ref_images = {}
    for i in range(MONSTER_COUNT + 1):
        if i == 0:
            img = MONSTER_IMAGES.get("empty")
        else:
            img = MONSTER_IMAGES.get(MONSTER_DATA["原始名称"][i])
        
        if img is None:
            logger.error("无法加载参考图片 i=%d", i)
            continue
        
        # Crop and resize template
        img_crop = img[
            int(img.shape[0] * 0.16):int(img.shape[0] * 0.80),
            int(img.shape[1] * 0.18):int(img.shape[1] * 0.82),
        ]
        ref_resized = cv2.resize(img_crop, (74, 74))
        ref_resized = ref_resized[0:70, :]
        
        if DEBUG_MODE:
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/xref_{i}.png", ref_resized)
        
        ref_images[i] = ref_resized
    return ref_images


def find_best_match(
    target: cv2.typing.MatLike, ref_images: dict[int, cv2.typing.MatLike]
):
    """Template matching to find best match."""
    confidence = float("-inf")
    best_id = -1
    
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    
    for img_id, ref_img in ref_images.items():
        try:
            res = cv2.matchTemplate(target, ref_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > confidence:
                confidence = max_val
                best_id = img_id
        except Exception as e:
            logger.exception(f"处理参考图像 {img_id} 时出错:", e)
            continue
    
    return best_id, confidence


def preprocess(img: cv2.typing.MatLike):
    """Binarize image for OCR."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    lower_bright = np.array([180, 180, 180])
    upper_bright = np.array([255, 255, 255])
    bright_mask = cv2.inRange(img, lower_bright, upper_bright)
    
    # Remove noise
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 13:
            cv2.drawContours(bright_mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    return bright_mask


def crop_to_min_bounding_rect(image: cv2.typing.MatLike):
    """Crop to minimum bounding rectangle of contours."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return image[y:y+h, x:x+w]


def add_black_border(img: cv2.typing.MatLike, border_size=3):
    """Add black border to image."""
    return cv2.copyMakeBorder(
        img, top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


class RecognizeMonster:
    """Monster recognition via template matching and OCR."""

    def __init__(self):
        """Initialize recognizer."""
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
        self.main_roi = None  # Optional custom ROI

    def process_regions(self, screenshot: np.ndarray) -> list[dict]:
        """
        Process full-screen screenshot to identify monsters.
        
        Args:
            screenshot: Full-screen BGR image (any resolution)
        
        Returns:
            List of 6 recognition results
        """
        # 1. Detect monster bar (auto-detect via find_monster_zone)
        from ..utils import find_monster_zone
        
        # Save original screenshot for debugging before processing
        if DEBUG_MODE:
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/original_screenshot.png", screenshot)
        
        try:
            monster_roi, cropped = find_monster_zone.cutFrame(screenshot)
            # Save debug images of monster bar detection
            if DEBUG_MODE:
                cv2.imwrite(f"{TMP_IMAGES_DIR}/cropped_monster_bar.png", cropped)
        except Exception as e:
            logger.exception("Monster bar detection failed: %s", e)
            return []
        
        if monster_roi is None or cropped is None:
            logger.error("Could not detect monster bar")
            return []
        
        # 2. Crop to standard 975x119
        try:
            monster_bar = cv2.resize(cropped, (975, 119))
        except Exception as e:
            logger.error("Crop failed: %s", e)
            return []
        
        # 3. Split into 6 regions and recognize
        results = []
        region_width = 975 // 6
        
        for i in range(6):
            x1 = i * region_width
            x2 = (i + 1) * region_width if i < 5 else 975
            region_img = monster_bar[:, x1:x2]
            
            result = self._recognize_region(region_img, i)
            results.append(result)
        
        return results

    def _recognize_region(self, region_img: np.ndarray, region_id: int, matched_threshold=0.5, ocr_threshold=0.95) -> dict:
        """
        Recognize a single monster region (template matching + OCR).
        
        Expects a pre-cropped monster region image (approx 162x119),
        not a full screenshot. The region contains one monster type
        and its count.
        
        Args:
            region_img: Pre-cropped monster region (approx 162x119)
            region_id: Region index (0-5)
            matched_threshold: Template matching confidence threshold
            ocr_threshold: OCR confidence threshold
            
        Returns:
            Recognition result dict with keys: region_id, matched_id, 
            number, confidence (and optionally error)
        """
        # Template matching
        try:
            matched_id, confidence = find_best_match(region_img, self.ref_images)
            logger.info("target: %d matched_id: %d, confidence: %.4f", region_id, matched_id, confidence)
            
            if matched_id != 0 and confidence < matched_threshold:
                raise ValueError(f"模板匹配置信度过低：{confidence}")
        except Exception as e:
            logger.exception(f"区域 {region_id} 匹配失败：{str(e)}")
            return {
                "region_id": region_id, "matched_id": 0, "number": "N/A", "error": str(e)
            }
        
        # OCR on the number area (right side of region)
        try:
            # Number is typically on the right ~30% of the region
            h, w = region_img.shape[:2]
            num_x1 = int(w * 0.6)
            sub_roi_num = region_img[:, num_x1:]
            
            processed = preprocess(sub_roi_num)
            processed = crop_to_min_bounding_rect(processed)
            processed = add_black_border(processed, border_size=3)
            
            number, ocr_confidence = self.do_num_ocr(processed)
            if number != "" and ocr_confidence < ocr_threshold:
                raise ValueError(f"OCR 置信度过低：{ocr_confidence}")
            
            if DEBUG_MODE:
                TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{TMP_IMAGES_DIR}/target_{region_id}.png", region_img)
                cv2.imwrite(f"{TMP_IMAGES_DIR}/number_{region_id}.png", processed)
            
            if number == "" and matched_id != 0:
                raise ValueError("发现有怪物但无数量异常数据！")
            if matched_id == 0 and number != "":
                raise ValueError("发现无怪物但有数量异常数据！")
            
            return {
                "region_id": region_id,
                "matched_id": matched_id,
                "number": number if number else "N/A",
                "confidence": round(confidence, 2),
            }
        except Exception as e:
            logger.exception(f"区域 {region_id} OCR 识别失败：{str(e)}")
            return {
                "region_id": region_id,
                "matched_id": matched_id,
                "number": "N/A",
                "error": str(e),
            }

    def do_num_ocr(self, img: cv2.typing.MatLike):
        """Perform OCR on number region."""
        result = self.ocr(img, use_det=False, use_cls=False, use_rec=True)
        logger.info("OCR: text: '%s', score: %s", result.txts[0], result.scores[0])
        if result.txts[0] != "" and not result.txts[0].isdigit():
            raise ValueError(f"OCR 识别结果不是数字：'{result.txts[0]}'")
        return result.txts[0], result.scores[0]


if __name__ == "__main__":
    # Example usage with test image
    test_img = f"{TMP_IMAGES_DIR}/zone1.png"
    if Path(test_img).exists():
        screenshot = cv2.imread(test_img)
        recognizer = RecognizeMonster()
        results = recognizer.process_regions(screenshot)
        print("识别结果：")
        for res in results:
            if "error" in res:
                print(f"区域{res['region_id']}: 错误 - {res['error']}")
            else:
                if res["matched_id"] != 0:
                    print(f"区域{res['region_id']} => 匹配 ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")
    else:
        print(f"测试图片 {test_img} 不存在")
