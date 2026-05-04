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
from pathlib import Path

import cv2
import numpy as np
from rapidocr import EngineType, RapidOCR

from cannotmax.config import DEBUG_MODE
from cannotmax.config.paths import TMP_IMAGES_DIR
from cannotmax.utils.images import get_monster_images

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
    from cannotdl.config import MONSTER_COUNT
    from cannotmax.utils.monster_data import get_monster_avatar_path

    ref_images = {}
    monster_images = get_monster_images()
    for i in range(MONSTER_COUNT + 1):
        if i == 0:
            img = monster_images.get("empty")
        else:
            img = monster_images.get(get_monster_avatar_path(i).stem)

        if img is None:
            logger.error("无法加载参考图片 i=%d", i)
            continue

        # Crop and resize template
        img_crop = img[
            int(img.shape[0] * 0.16) : int(img.shape[0] * 0.80),
            int(img.shape[1] * 0.18) : int(img.shape[1] * 0.82),
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
            if ref_img.shape[0] > target.shape[0] or ref_img.shape[1] > target.shape[1]:
                continue
            res = cv2.matchTemplate(target, ref_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > confidence:
                confidence = max_val
                best_id = img_id
        except Exception:
            logger.exception("处理参考图像 %d 时出错", img_id)
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
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
    return image[y : y + h, x : x + w]


def add_black_border(img: cv2.typing.MatLike, border_size=3):
    """Add black border to image."""
    return cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


class ROINotSelectedError(Exception):
    """WIN 模式下用户未选择 ROI 时抛出"""

    pass


class RecognizeMonster:
    """Monster recognition via template matching and OCR."""

    def __init__(self, crop_ratio=None, avatar_regions=None, number_regions=None):
        """Initialize recognizer.

        Args:
            crop_ratio: 怪物条裁剪比例 [(x1,y1), (x2,y2)] 或 None（ADB/PC 用默认值）
            avatar_regions: 6个头像区域在 975x119 内的相对坐标，None 用默认值
            number_regions: 6个数字区域在 975x119 内的相对坐标，None 用默认值
        """
        self.ref_images = load_ref_images()
        self.ocr = get_rapidocr_engine()
        self.crop_ratio = crop_ratio
        self.avatar_regions = avatar_regions
        self.number_regions = number_regions

    def _resolve_crop_ratio(self) -> tuple:
        """解析裁剪比例。

        Returns:
            ((x1, y1), (x2, y2)) 相对坐标

        Raises:
            ROINotSelectedError: crop_ratio=None 时
        """
        if self.crop_ratio is not None:
            return self.crop_ratio
        raise ROINotSelectedError("请先选择怪物条范围")

    def _crop_by_ratio(self, screenshot: np.ndarray, ratio: tuple) -> np.ndarray:
        """按相对坐标裁切图像。

        Args:
            screenshot: BGR 图像
            ratio: ((x1, y1), (x2, y2)) 相对坐标

        Returns:
            裁切后的 BGR 图像
        """
        h, w = screenshot.shape[:2]
        (x1, y1), (x2, y2) = ratio
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        return screenshot[py1:py2, px1:px2]

    def process_regions(self, screenshot: np.ndarray, mode: str = "ADB") -> list[dict]:
        """Process full-screen screenshot to identify monsters.

        Args:
            screenshot: Full-screen BGR image (any resolution)
            mode: 捕获模式 (ADB/PC/WIN)，影响默认裁剪参数选择

        Returns:
            List of 6 recognition results
        """
        from cannotmax.utils import find_monster_zone
        from cannotmax.utils.roi_transform import transform_coords

        detected_zones = None

        # Save original screenshot for debugging
        if DEBUG_MODE:
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/original_screenshot.png", screenshot)

        # 1. Resolve crop ratio and crop, crop_ratio is set by user selection in WIN mode,
        # or use config for ADB/PC modes
        ratio = self._resolve_crop_ratio()
        cropped = self._crop_by_ratio(screenshot, ratio)

        # 2. WIN mode: find_monster_zone provides precise zone coordinates (relative to user ROI)
        if detected_zones is None:
            try:
                # 获取用户 ROI 在原始截图上的位置
                orig_h, orig_w = screenshot.shape[:2]
                roi_x1, roi_y1 = int(ratio[0][0] * orig_w), int(ratio[0][1] * orig_h)
                roi_x2, roi_y2 = int(ratio[1][0] * orig_w), int(ratio[1][1] * orig_h)
                roi_x, roi_y = roi_x1, roi_y1
                roi_w, roi_h = roi_x2 - roi_x1, roi_y2 - roi_y1

                # find_monster_zone 在 ROI 上识别（返回相对于 ROI 的归一化坐标）
                d_avatar, d_nums = find_monster_zone.find_monster_zone(cropped)
                if d_avatar is None:
                    logger.error("find_monster_zone 检测失败，请重新选择区域")
                    return []

                # 将相对于 ROI 的归一化坐标转换为相对于原始截图的像素坐标
                logger.info(
                    "ROI params: roi_x=%d, roi_y=%d, roi_w=%d, roi_h=%d",
                    roi_x,
                    roi_y,
                    roi_w,
                    roi_h,
                )
                logger.info("d_avatar before transform:\n%s", d_avatar)

                # 将相对于 ROI 的归一化坐标转换为相对于原始截图的像素坐标
                # 正向: pixel = roi_x + normalized * roi_w
                # transform: scale_x=roi_w, x_offset=-roi_x/roi_w
                avatar_px = transform_coords(
                    d_avatar,
                    x_offset=-roi_x / roi_w,
                    y_offset=-roi_y / roi_h,
                    scale_x=roi_w,
                    scale_y=roi_h,
                    clamp=False,
                )
                nums_px = transform_coords(
                    d_nums,
                    x_offset=-roi_x / roi_w,
                    y_offset=-roi_y / roi_h,
                    scale_x=roi_w,
                    scale_y=roi_h,
                    clamp=False,
                )

                logger.info("avatar_px after transform:\n%s", avatar_px)

                # 检查转换后的坐标包围盒是否合理
                bbox_x1, bbox_y1 = avatar_px[:, :2].min(axis=0)
                bbox_x2, bbox_y2 = avatar_px[:, 2:].max(axis=0)
                bbox_w, bbox_h = bbox_x2 - bbox_x1, bbox_y2 - bbox_y1
                logger.info(
                    "BBox: x1=%d, y1=%d, x2=%d, y2=%d, w=%d, h=%d, ROI_w=%d, ROI_h=%d",
                    int(bbox_x1),
                    int(bbox_y1),
                    int(bbox_x2),
                    int(bbox_y2),
                    int(bbox_w),
                    int(bbox_h),
                    roi_w,
                    roi_h,
                )

                # 从原始截图裁剪完整怪物条
                x_min = max(0, int(avatar_px[:, 0].min()))
                y_min = max(0, int(avatar_px[:, 1].min()))
                x_max = min(orig_w, int(avatar_px[:, 2].max()))
                y_max = min(orig_h, int(avatar_px[:, 3].max()))

                monster_bar_crop = screenshot[y_min:y_max, x_min:x_max]
                crop_h, crop_w = monster_bar_crop.shape[:2]
                logger.info(
                    "Cropped monster bar: x=%d, y=%d, w=%d, h=%d, crop_w=%d, crop_h=%d",
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                    crop_w,
                    crop_h,
                )
                if crop_w == 0 or crop_h == 0:
                    logger.error("怪物条裁剪失败，区域无效")
                    return []

                # 重新计算相对于怪物条区域的归一化坐标
                new_avatar = transform_coords(
                    avatar_px,
                    x_offset=x_min,
                    y_offset=y_min,
                    scale_x=1.0 / crop_w,
                    scale_y=1.0 / crop_h,
                )
                new_nums = transform_coords(
                    nums_px,
                    x_offset=x_min,
                    y_offset=y_min,
                    scale_x=1.0 / crop_w,
                    scale_y=1.0 / crop_h,
                )

                cropped = monster_bar_crop
                detected_zones = (new_avatar, new_nums)

            except Exception as e:
                logger.exception("Monster bar detection failed: %s", e)
                return []

        # 3. Resize to standard 975x119
        if cropped is None or cropped.size == 0:
            logger.error("Could not detect monster bar")
            return []

        try:
            monster_bar = cv2.resize(cropped, (975, 119))
        except Exception as e:
            logger.error("Crop failed: %s", e)
            return []

        if DEBUG_MODE:
            cv2.imwrite(f"{TMP_IMAGES_DIR}/monster_bar_975x119.png", monster_bar)

        # 可视化：画框并保存
        if detected_zones is not None and DEBUG_MODE:
            d_avatar, d_nums = detected_zones
            h, w = monster_bar.shape[:2]
            vis_img = monster_bar.copy()
            for idx in range(6):
                # 头像框（绿色）
                ax1, ay1, ax2, ay2 = d_avatar[idx]
                cv2.rectangle(
                    vis_img,
                    (int(ax1 * w), int(ay1 * h)),
                    (int(ax2 * w), int(ay2 * h)),
                    (0, 255, 0),  # 绿色
                    2,
                )
                # 数字框（红色）
                nx1, ny1, nx2, ny2 = d_nums[idx]
                cv2.rectangle(
                    vis_img,
                    (int(nx1 * w), int(ny1 * h)),
                    (int(nx2 * w), int(ny2 * h)),
                    (0, 0, 255),  # 红色
                    2,
                )
                # 标签
                cv2.putText(
                    vis_img,
                    f"{idx}",
                    (int(ax1 * w), int(ay1 * h) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{TMP_IMAGES_DIR}/debug_monster_bar_boxes.png", vis_img)

        # 4. Split into 6 regions using precise coordinates and recognize
        results = []
        if detected_zones is not None:
            d_avatar, d_nums = detected_zones
            h, w = monster_bar.shape[:2]
            for idx in range(6):
                ax1, ay1, ax2, ay2 = d_avatar[idx]
                avatar_img = monster_bar[
                    int(ay1 * h) : int(ay2 * h),
                    int(ax1 * w) : int(ax2 * w),
                ]
                nx1, ny1, nx2, ny2 = d_nums[idx]
                num_img = monster_bar[
                    int(ny1 * h) : int(ny2 * h),
                    int(nx1 * w) : int(nx2 * w),
                ]
                result = self._recognize_region(avatar_img, num_img, idx)
                results.append(result)

        if not results:
            logger.warning("No valid monster detected, recognition failed")

        return results

    def _recognize_region(
        self,
        avatar_img: np.ndarray,
        num_img: np.ndarray,
        region_id: int,
        matched_threshold=0.5,
        ocr_threshold=0.95,
    ) -> dict:
        """Recognize a single monster region (template matching + OCR).

        Args:
            avatar_img: Pre-cropped avatar sub-region from 975x119 bar
            num_img: Pre-cropped number sub-region from 975x119 bar
            region_id: Region index (0-5)
            matched_threshold: Template matching confidence threshold
            ocr_threshold: OCR confidence threshold

        Returns:
            Recognition result dict
        """
        # Template matching
        try:
            matched_id, confidence = find_best_match(avatar_img, self.ref_images)
            logger.info(
                "target: %d matched_id: %d, confidence: %.4f",
                region_id,
                matched_id,
                confidence,
            )

            if matched_id != 0 and confidence < matched_threshold:
                raise ValueError(f"模板匹配置信度过低：{confidence}")
        except Exception as e:
            logger.exception(f"区域 {region_id} 匹配失败：{str(e)}")
            return {
                "region_id": region_id,
                "matched_id": 0,
                "number": "N/A",
                "error": str(e),
            }

        # OCR on the number area
        try:
            processed = preprocess(num_img)
            processed = crop_to_min_bounding_rect(processed)
            processed = add_black_border(processed, border_size=3)

            number, ocr_confidence = self.do_num_ocr(processed)
            if number != "" and ocr_confidence < ocr_threshold:
                raise ValueError(f"OCR 置信度过低：{ocr_confidence}")

            if DEBUG_MODE:
                TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{TMP_IMAGES_DIR}/target_{region_id}.png", avatar_img)
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
        recognizer = RecognizeMonster(crop_ratio=((0.0, 0.0), (1.0, 1.0)))
        results = recognizer.process_regions(screenshot, mode="WIN")
        print("识别结果：")
        for res in results:
            if "error" in res:
                print(f"区域{res['region_id']}: 错误 - {res['error']}")
            else:
                if res["matched_id"] != 0:
                    print(
                        f"区域{res['region_id']} => 匹配 ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}"
                    )
    else:
        print(f"测试图片 {test_img} 不存在")
