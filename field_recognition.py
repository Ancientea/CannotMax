import json
import re
import os
import logging
from pathlib import Path
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# 场地识别ROI坐标
ROI_COORDINATES = {
    "middle_row_blocks": [{"x": 674, "y": 411, "width": 574, "height": 135}],
    "side_fire_cannon": [{"x": 139, "y": 343, "width": 138, "height": 97},
                         {"x": 1691, "y": 530, "width": 156, "height": 103}],
    "top_crossbow": [{"x": 718, "y": 18, "width": 109, "height": 93}, {"x": 913, "y": 27, "width": 109, "height": 92},
                     {"x": 1096, "y": 19, "width": 113, "height": 99}],
    "top_fire_cannon": [{"x": 533, "y": 25, "width": 95, "height": 97},
                        {"x": 1317, "y": 23, "width": 71, "height": 102}],
    "two_row_blocks": [{"x": 696, "y": 239, "width": 528, "height": 125},
                       {"x": 652, "y": 611, "width": 619, "height": 144}]
}

class FieldRecognizer:
    def __init__(self):
        self.field_model = None
        self.field_transform = None
        self.field_device = None
        self.idx_to_class = {}
        self.grouped_elements = {}
        self.image_feature_columns = []
        self.is_initialized = False
        
        # 初始化场地识别模型
        self._init_field_recognition()

    def _init_field_recognition(self):
        """初始化场地识别模型和相关组件"""
        try:
            # 设置设备
            self.field_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"场地识别将使用设备: {self.field_device}")
            
            # 加载类别映射
            model_dir = Path("tools/battlefield_recognize")
            class_map_path = model_dir / "class_to_idx.json"
            pth_model_path = model_dir / "field_recognize.pth"
            
            if not class_map_path.exists():
                logger.warning("找不到场地识别类别映射文件，跳过场地识别初始化")
                return

            with open(class_map_path, 'r', encoding='utf-8') as f:
                class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            num_classes = len(class_to_idx)

            # 准备特征列
            self.grouped_elements = defaultdict(list)
            for class_name in class_to_idx.keys():
                if class_name.endswith('_none'):
                    continue
                condensed_name = re.sub(r'_position_\d+', '', class_name)
                self.grouped_elements[condensed_name].append(class_name)
            self.image_feature_columns = sorted(self.grouped_elements.keys())
         
            if not pth_model_path.exists():
                logger.warning("找不到场地识别模型文件，跳过场地识别初始化")
                return

            # 加载模型
            self.field_model = self._load_pytorch_model(str(pth_model_path), num_classes, self.field_device)
            
            # 设置图像变换
            self.field_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.is_initialized = True
            logger.info(f"场地识别初始化成功，将生成 {len(self.image_feature_columns)} 个特征列")
            
        except Exception as e:
            logger.error(f"场地识别初始化失败: {e}")
            self.is_initialized = False

    def _load_pytorch_model(self, model_path: str, num_classes: int, device: torch.device):
        """加载 PyTorch 模型并设置为评估模式"""
        logger.info(f"正在加载 PyTorch 模型: {model_path}")
        model = models.mobilenet_v3_small(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("模型加载成功并已切换到评估模式。")
        return model

    def _predict_scene_pytorch(self, image_path: str, threshold: float = 0.5) -> list[str]:
        """使用 PyTorch 模型对给定图片的所有 ROI 进行分类预测"""
        try:
            full_image = Image.open(image_path).convert('RGB')
        except Exception:
            return []

        if full_image.size != (1920, 1080):
            return []

        detected_classes = []
        with torch.no_grad():
            for location, boxes in ROI_COORDINATES.items():
                for i, box in enumerate(boxes):
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    roi_pil = full_image.crop((x, y, x + w, y + h))
                    input_tensor = self.field_transform(roi_pil).unsqueeze(0)
                    input_tensor = input_tensor.to(self.field_device)
                    outputs = self.field_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    max_prob, predicted_index_tensor = torch.max(probabilities, 0)
                    predicted_index = predicted_index_tensor.item()

                    if max_prob.item() >= threshold:
                        predicted_class = self.idx_to_class[predicted_index]
                        if not predicted_class.endswith('_none'):
                            detected_classes.append(predicted_class)
        return detected_classes

    def recognize_field_elements(self, screenshot):
        """识别场地元素"""
        if not self.is_initialized or self.field_model is None:
            logger.debug("场地识别模型未初始化，跳过场地识别")
            return {}
        
        try:
            # 将OpenCV图像转换为PIL图像
            screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(screenshot_rgb)
            
            # 保存临时文件用于识别
            temp_image_path = "temp_screenshot.png"
            pil_image.save(temp_image_path)
            
            # 进行场地识别
            detected_full_names = set(self._predict_scene_pytorch(temp_image_path, threshold=0.5))
            
            # 删除临时文件
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            # 处理识别结果
            field_data = {}
            for condensed_name, full_names in self.grouped_elements.items():
                num_positions = len(full_names)
                if num_positions == 1:
                    field_data[condensed_name] = 1 if full_names[0] in detected_full_names else 0
                else:
                    detections_in_group = [fn in detected_full_names for fn in full_names]
                    num_detected = sum(detections_in_group)
                    if num_detected == num_positions:
                        field_data[condensed_name] = 1
                    elif num_detected == 0:
                        field_data[condensed_name] = 0
                    else:
                        field_data[condensed_name] = -1
            
            logger.debug(f"场地识别完成，检测到元素: {list(detected_full_names)}")
            return field_data
            
        except Exception as e:
            logger.error(f"场地识别失败: {e}")
            return {}

    def get_feature_columns(self):
        """获取场地特征列名列表"""
        return self.image_feature_columns.copy()

    def is_ready(self):
        """检查场地识别器是否已准备就绪"""
        return self.is_initialized
