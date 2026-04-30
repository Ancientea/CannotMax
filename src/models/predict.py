import re
import sys
import importlib
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import torch
import logging

from src.core.config import MONSTER_COUNT
from src.core.config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)


def _register_legacy_model_aliases():
    """
    为旧版 pickle 模型提供模块路径兼容：
    例如历史权重里记录的是 `models.model`，而当前代码已迁移到 `src.models.model`。
    """
    legacy_to_new = {
        "models": "src.models",
        "models.model": "src.models.model",
        "models.muon": "src.models.muon",
    }

    for legacy_name, new_name in legacy_to_new.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = importlib.import_module(new_name)
        except Exception as e:
            logger.debug(
                "兼容模块别名注册失败: %s -> %s, %s",
                legacy_name,
                new_name,
                e,
            )


def get_device(prefer_gpu=True):
    """
    prefer_gpu (bool): 是否优先尝试使用GPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            logger.info("Use torch with cuda")
            return torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            logger.info("Use torch with mps")
            return torch.device("mps")  # Apple Silicon GPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
            logger.info("Use torch with xpu")
            return torch.device("xpu")
    logger.info("Use torch with cpu")
    return torch.device("cpu")


class CannotModel:
    def __init__(self, model_path="models"):
        self.device = get_device()
        self.is_model_loaded = False
        self.model_path = self._resolve_model_path(model_path)
        try:
            self.load_model()  # 初始化时加载模型
            self.is_model_loaded = True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None

    def _resolve_model_path(self, path):
        """
        Resolves the model path. If a directory is given, finds the latest model file.
        If a file is given, returns it directly.
        """
        if Path(path).is_dir():
            logger.info(f"Searching for the latest model in directory: {path}")
            model_dir = Path(path)
            models = [
                f
                for f in model_dir.iterdir()
                if f.suffix == ".pth" and f.is_file()
            ]
            if not models:
                logger.error(f"No model files (.pth) found in {path}")

            priority = {"loss": 0, "acc": 1, "full": 2}
            valid_models = []

            pattern = re.compile(
                r"best_model_(acc|loss|full)_data\d+_acc\d+\.\d+_loss\d+\.\d+_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.pth$"
            )

            for model_file_path in models:
                match = pattern.match(model_file_path.name)
                if match:
                    model_type = match.group(1)
                    timestamp_str = match.group(
                        2
                    )  # Group 2 captures the timestamp
                    try:
                        model_time = datetime.strptime(
                            timestamp_str, "%Y_%m_%d_%H_%M_%S"
                        )
                        valid_models.append(
                            (
                                model_time,
                                priority.get(model_type, 3),
                                model_file_path,
                            )
                        )
                    except ValueError:
                        continue  # Ignore files with malformed timestamps

            if valid_models:
                # Sort by time DESC, then priority ASC (loss=0, acc=1, full=2)
                valid_models.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                latest_model_path = valid_models[0][2]
                logger.info(f"Found latest model: {latest_model_path}")
                return str(latest_model_path)
            else:
                logger.error(
                    f"No models with the expected name format found in {path}"
                )

        elif Path(path).is_file():
            logger.info(f"Using specified model file: {path}")
            return path
        else:
            logger.error(f"Provided model path is invalid: {path}")
            return ""

    def load_model(self):
        """初始化时加载模型"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    rf"未找到训练好的模型文件 {self.model_path}，请先训练模型"
                )

            # 兼容项目重构前保存的整模型 pickle 路径
            _register_legacy_model_aliases()

            try:
                model = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            self.model = model.to(self.device)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            raise e  # 无法继续运行，退出程序

    def export_onnx(self, outputpath, monster_count=MONSTER_COUNT):
        # 确保模型在 CPU 上（避免设备不一致）
        self.model = self.model.cpu()
        self.model.eval()

        # 生成虚拟输入（与模型同设备）
        device = next(self.model.parameters()).device
        dummy_left_counts = torch.randint(
            0, 10, (1, monster_count), dtype=torch.int16, device=device
        )
        dummy_right_counts = torch.randint(
            0, 10, (1, monster_count), dtype=torch.int16, device=device
        )

        # 获取符号和绝对值张量（确保在相同设备）
        left_signs = torch.sign(dummy_left_counts.to(torch.int64)).to(device)
        left_counts = torch.abs(dummy_left_counts.to(torch.int64)).to(device)
        right_signs = torch.sign(dummy_right_counts.to(torch.int64)).to(device)
        right_counts = torch.abs(dummy_right_counts.to(torch.int64)).to(device)

        # 导出参数
        input_names = [
            "left_signs",
            "left_counts",
            "right_signs",
            "right_counts",
        ]
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["output"] = {0: "batch_size"}

        # 导出 ONNX
        torch.onnx.export(
            self.model,
            (left_signs, left_counts, right_signs, right_counts),
            outputpath,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=20,
            verbose=True,  # 开启详细输出便于调试
        )

    def get_prediction(
        self,
        left_counts: np.typing.ArrayLike,
        right_counts: np.typing.ArrayLike,
    ):
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        # 转换为张量并处理符号和绝对值
        left_signs = (
            torch.sign(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        left_counts = (
            torch.abs(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_signs = (
            torch.sign(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_counts = (
            torch.abs(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )

        # 预测流程
        with torch.no_grad():
            # 使用修改后的模型前向传播流程
            prediction = self.model(
                left_signs, left_counts, right_signs, right_counts
            ).item()

            # 确保预测值在有效范围内
            if np.isnan(prediction) or np.isinf(prediction):
                logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5

            # 检查预测结果是否在[0,1]范围内
            if prediction < 0 or prediction > 1:
                prediction = max(0, min(1, prediction))

        return prediction

    def get_prediction_with_terrain(self, full_features: np.typing.ArrayLike):
        """使用包含地形特征的完整特征向量进行预测"""
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        # 检查特征向量长度
        expected_length = (
            MONSTER_COUNT * 2 + FIELD_FEATURE_COUNT * 2
        )  # 77L + 6L + 77R + 6R = 166
        if len(full_features) != expected_length:
            logger.warning(
                f"特征向量长度不匹配: 期望{expected_length}, 实际{len(full_features)}"
            )
            # 如果长度不匹配，回退到原始方法
            left_counts = full_features[:MONSTER_COUNT]
            right_counts = full_features[MONSTER_COUNT : MONSTER_COUNT * 2]
            return self.get_prediction(left_counts, right_counts)

        # 提取各个部分
        left_monsters = full_features[:MONSTER_COUNT]  # 1L-77L
        left_terrain = full_features[
            MONSTER_COUNT : MONSTER_COUNT + FIELD_FEATURE_COUNT
        ]  # 78L-83L
        right_monsters = full_features[
            MONSTER_COUNT
            + FIELD_FEATURE_COUNT : MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT
        ]  # 1R-77R
        right_terrain = full_features[
            MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT : MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT * 2
        ]  # 78R-83R

        # 合并怪物特征和地形特征（按照训练时的格式）
        left_counts = np.concatenate([left_monsters, left_terrain])
        right_counts = np.concatenate([right_monsters, right_terrain])

        # 转换为张量并处理符号和绝对值
        # 对于怪物特征，使用符号和绝对值
        # 对于地形特征，不需要符号处理（地形特征本身就是0/1值）
        left_monster_signs = torch.sign(
            torch.tensor(left_monsters, dtype=torch.int16)
        )
        left_terrain_signs = torch.ones_like(
            torch.tensor(left_terrain, dtype=torch.int16)
        )  # 地形特征符号为1
        left_signs = (
            torch.cat([left_monster_signs, left_terrain_signs])
            .unsqueeze(0)
            .to(self.device)
        )

        left_monster_counts = torch.abs(
            torch.tensor(left_monsters, dtype=torch.int16)
        )
        left_terrain_counts = torch.tensor(
            left_terrain, dtype=torch.int16
        )  # 地形特征直接使用原值
        left_counts_tensor = (
            torch.cat([left_monster_counts, left_terrain_counts])
            .unsqueeze(0)
            .to(self.device)
        )

        right_monster_signs = torch.sign(
            torch.tensor(right_monsters, dtype=torch.int16)
        )
        right_terrain_signs = torch.ones_like(
            torch.tensor(right_terrain, dtype=torch.int16)
        )  # 地形特征符号为1
        right_signs = (
            torch.cat([right_monster_signs, right_terrain_signs])
            .unsqueeze(0)
            .to(self.device)
        )

        right_monster_counts = torch.abs(
            torch.tensor(right_monsters, dtype=torch.int16)
        )
        right_terrain_counts = torch.tensor(
            right_terrain, dtype=torch.int16
        )  # 地形特征直接使用原值
        right_counts_tensor = (
            torch.cat([right_monster_counts, right_terrain_counts])
            .unsqueeze(0)
            .to(self.device)
        )

        # 预测流程
        with torch.no_grad():
            # 使用修改后的模型前向传播流程，现在包含地形特征
            prediction = self.model(
                left_signs,
                left_counts_tensor,
                right_signs,
                right_counts_tensor,
            ).item()

            # 确保预测值在有效范围内
            if np.isnan(prediction) or np.isinf(prediction):
                logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5

            # 检查预测结果是否在[0,1]范围内
            if prediction < 0 or prediction > 1:
                prediction = max(0, min(1, prediction))

        return prediction
