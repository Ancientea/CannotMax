"""
基于位置顺序的预测模块
"""
import re
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import torch
import logging

from constants import POSITIONS_PER_SIDE, FEATURES_PER_POSITION
from config import FIELD_FEATURE_COUNT
from recognize import MONSTER_COUNT

logger = logging.getLogger(__name__)


def get_device(prefer_gpu=True):
    if prefer_gpu:
        if torch.cuda.is_available():
            logger.info("Use torch with cuda")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Use torch with mps")
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            logger.info("Use torch with xpu")
            return torch.device("xpu")
    logger.info("Use torch with cpu")
    return torch.device("cpu")


class CannotModelPosition:
    """位置感知的模型加载和预测"""
    
    def __init__(self, model_path="models"):
        self.device = get_device()
        self.is_model_loaded = False
        self.model_path = self._resolve_model_path(model_path)
        try:
            self.load_model()
            self.is_model_loaded = True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
    
    def _resolve_model_path(self, path):
        """解析模型路径，优先查找position模型"""
        if Path(path).is_dir():
            logger.info(f"Searching for the latest position model in directory: {path}")
            model_dir = Path(path)
            models = [f for f in model_dir.iterdir() if f.suffix == ".pth" and f.is_file() and "position" in f.name]
            
            if not models:
                logger.error(f"No position model files (.pth) found in {path}")
                return ""
            
            priority = {"loss": 0, "acc": 1, "full": 2}
            valid_models = []
            
            pattern = re.compile(
                r"best_model_position_(acc|loss|full)_.*_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.pth$"
            )
            
            for model_file_path in models:
                match = pattern.match(model_file_path.name)
                if match:
                    model_type = match.group(1)
                    timestamp_str = match.group(2)
                    try:
                        model_time = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
                        valid_models.append((model_time, priority.get(model_type, 3), model_file_path))
                    except ValueError:
                        continue
            
            if valid_models:
                valid_models.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                latest_model_path = valid_models[0][2]
                logger.info(f"Found latest position model: {latest_model_path}")
                return str(latest_model_path)
            else:
                logger.error(f"No valid position models found in {path}")
                return ""
        
        elif Path(path).is_file():
            logger.info(f"Using specified model file: {path}")
            return path
        else:
            logger.error(f"Provided model path is invalid: {path}")
            return ""
    
    def load_model(self):
        """加载模型"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"未找到训练好的模型文件 {self.model_path}，请先训练模型"
                )
            
            try:
                model = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                model = torch.load(self.model_path, map_location=self.device)
            
            model.eval()
            self.model = model.to(self.device)
        
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            raise e
    
    def get_prediction_with_position(self, position_data):
        """
        使用位置顺序数据进行预测
        
        Args:
            position_data: dict with keys:
                - 'left_positions': list of (monster_id, count) tuples, length 3
                - 'right_positions': list of (monster_id, count) tuples, length 3
                - 'left_field': array of field features
                - 'right_field': array of field features
        
        Returns:
            float: 预测的右方胜率 (0-1)
        """
        if self.model is None:
            raise RuntimeError("模型未正确初始化")
        
        # 提取数据
        left_positions = position_data['left_positions']
        right_positions = position_data['right_positions']
        left_field = position_data.get('left_field', np.zeros(FIELD_FEATURE_COUNT))
        right_field = position_data.get('right_field', np.zeros(FIELD_FEATURE_COUNT))
        
        # 转换为张量
        left_ids = torch.tensor([pos[0] for pos in left_positions], dtype=torch.long).unsqueeze(0).to(self.device)
        left_counts = torch.tensor([pos[1] for pos in left_positions], dtype=torch.float).unsqueeze(0).to(self.device)
        right_ids = torch.tensor([pos[0] for pos in right_positions], dtype=torch.long).unsqueeze(0).to(self.device)
        right_counts = torch.tensor([pos[1] for pos in right_positions], dtype=torch.float).unsqueeze(0).to(self.device)
        left_field_tensor = torch.tensor(left_field, dtype=torch.float).unsqueeze(0).to(self.device)
        right_field_tensor = torch.tensor(right_field, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            prediction = self.model(
                left_ids, left_counts, right_ids, right_counts,
                left_field_tensor, right_field_tensor
            ).item()
            
            # 确保预测值在有效范围内
            if np.isnan(prediction) or np.isinf(prediction):
                logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5
            
            prediction = max(0, min(1, prediction))
        
        return prediction
    
    def export_onnx(self, outputpath):
        """导出为ONNX格式"""
        self.model = self.model.cpu()
        self.model.eval()
        
        device = next(self.model.parameters()).device
        
        # 生成虚拟输入
        dummy_left_ids = torch.randint(0, MONSTER_COUNT, (1, POSITIONS_PER_SIDE), dtype=torch.long, device=device)
        dummy_left_counts = torch.rand(1, POSITIONS_PER_SIDE, device=device)
        dummy_right_ids = torch.randint(0, MONSTER_COUNT, (1, POSITIONS_PER_SIDE), dtype=torch.long, device=device)
        dummy_right_counts = torch.rand(1, POSITIONS_PER_SIDE, device=device)
        dummy_left_field = torch.rand(1, FIELD_FEATURE_COUNT, device=device)
        dummy_right_field = torch.rand(1, FIELD_FEATURE_COUNT, device=device)
        
        input_names = ["left_ids", "left_counts", "right_ids", "right_counts", "left_field", "right_field"]
        dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
        dynamic_axes["output"] = {0: 'batch_size'}
        
        torch.onnx.export(
            self.model,
            (dummy_left_ids, dummy_left_counts, dummy_right_ids, dummy_right_counts, dummy_left_field, dummy_right_field),
            outputpath,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=20,
            verbose=True
        )
        
        logger.info(f"模型已导出到: {outputpath}")
