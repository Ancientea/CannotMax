from pathlib import Path
import sys, importlib
import numpy as np
import torch
import json
import logging
from train import UnitAwareTransformer
from config import MONSTER_COUNT, FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)

def get_device(prefer_gpu=True):
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    return torch.device("cpu")


class CannotModel:
    def __init__(self, model_path="models", load_all=False):
        self.device = get_device()
        self.models = []
        self.model_names = []
        
        base_dir = Path(__file__).parent
        model_dir = base_dir / model_path
        if not model_dir.is_dir():
            model_dir = base_dir / "models"
        self.model_path = str(model_dir.resolve())
        
        # 兼容旧模型
        import train
        if not hasattr(train, "EnsembleModel"):
            train.EnsembleModel = UnitAwareTransformer
        if not hasattr(train, "UnitAwareTransformer"):
            train.UnitAwareTransformer = UnitAwareTransformer
        sys.modules["models.model"] = train
        sys.modules.setdefault("ensemble_model", train)
        try:
            importlib.import_module("models")
        except ImportError:
            pass

        # 读取权重文件，确定需要加载的模型
        weights_path = Path(self.model_path) / "model_weights.json"
        target_names = None
        if weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    target_names = set(json.load(f).keys())
            except Exception:
                pass

        all_pth = sorted(model_dir.glob("best_model_*.pth"))
        if not load_all and target_names:
            load_targets = [f for f in all_pth if f.stem in target_names or f.name in target_names]
            logger.info(f"精简加载: {len(load_targets)}/{len(all_pth)} 个模型")
        else:
            load_targets = all_pth
            if load_all:
                logger.info(f"全加载模式: {len(all_pth)} 个模型")

        def _load_one(f):
            try:
                m = torch.load(str(f), map_location=self.device, weights_only=False)
                if not hasattr(m, 'unit_embed'):
                    return False
                m.eval()
                self.models.append(m)
                self.model_names.append(f.name)
                return True
            except Exception:
                return False

        for f in load_targets:
            if _load_one(f):
                logger.info(f"加载: {f.name}")

        # 精简模式一个都没加载到 → 回退全加载
        if not self.models and not load_all:
            logger.info("精简加载失败，回退全加载...")
            for f in all_pth:
                if _load_one(f):
                    logger.info(f"加载: {f.name}")

        # 终极兜底：不限命名格式
        if not self.models:
            for f in sorted(model_dir.glob("*.pth")):
                if _load_one(f):
                    logger.info(f"加载: {f.name}")
        
        self.is_model_loaded = len(self.models) > 0
        if self.is_model_loaded:
            self.model_weights = {name: 3.0 for name in self.model_names}
            self._last_predictions = {}
            self._load_weights()
            logger.info(f"加载完成: {len(self.models)} 个模型")
        else:
            logger.error(f"在 {model_dir} 中未找到 .pth 模型")

    def _weights_path(self):
        return Path(self.model_path) / "model_weights.json"

    def _load_weights(self):
        p = self._weights_path()
        if p.exists():
            try:
                saved = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    loaded = 0
                    for name in self.model_names:
                        # 兼容 model_selection.py 输出（不带 .pth）和本模块格式（带 .pth）
                        stem = Path(name).stem
                        if name in saved:
                            self.model_weights[name] = saved[name]
                            loaded += 1
                        elif stem in saved:
                            self.model_weights[name] = saved[stem]
                            loaded += 1
                    logger.info(f"加载权重: {loaded}/{len(self.model_names)} 个")
                    if loaded == 0:
                        logger.warning("权重文件中未匹配到任何模型（key 格式不一致）")
            except Exception as e:
                logger.warning(f"加载权重文件失败: {e}")

    def _save_weights(self):
        # 统一用 stem（不带 .pth），与 模型权重计算.py 输出一致
        clean = {Path(k).stem: v for k, v in self.model_weights.items()}
        with open(self._weights_path(), "w") as f:
            json.dump(clean, f)

    def adjust_weights(self, actual: float):
        """actual=0(左胜) 或 1(右胜)。LR 骨架 + 动态微调：步长=权重1%，最小 0.002。"""
        if not hasattr(self, '_last_predictions') or not self._last_predictions:
            return
        lo, hi = 1/3, 2/3
        skipped = []
        for name, p in self._last_predictions.items():
            if name not in self.model_weights:
                continue
            w = self.model_weights[name]
            if w == 0.0:
                skipped.append(name[:40])
                continue
            if p <= lo:       zone = "L"
            elif p >= hi:     zone = "R"
            else:             zone = "C"
            correct = (actual < 0.5 and zone == "L") or (actual > 0.5 and zone == "R")
            wrong   = (actual < 0.5 and zone == "R") or (actual > 0.5 and zone == "L")
            step = max(0.002, abs(w) * 0.01)  # 步长 = 当前权重的 1%
            if correct:
                self.model_weights[name] = min(8.0, w + step)
            elif wrong:
                self.model_weights[name] = max(-1.0, w - step)
        if skipped:
            logger.info(f"LR 排除模型（权重锁定0）: {skipped}")
        self._save_weights()
        logger.info(f"权重更新完成")

    def _ensemble_predict(self, left_signs, left_counts, right_signs, right_counts):
        preds = []
        names = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                p = model(left_signs, left_counts, right_signs, right_counts).item()
                if not (np.isnan(p) or np.isinf(p)):
                    preds.append(p)
                    names.append(self.model_names[i] if i < len(self.model_names) else f"model_{i}")
        if not preds:
            return 0.5
        # 加权平均
        total_w = sum(self.model_weights.get(n, 3.0) for n in names)
        if total_w == 0:
            return float(np.clip(np.mean(preds), 0.0, 1.0))
        weighted = sum(p * self.model_weights.get(n, 3.0) for p, n in zip(preds, names))
        return float(np.clip(weighted / total_w, 0.0, 1.0))

    def get_prediction(self, left_counts: np.ndarray, right_counts: np.ndarray):
        if not self.models:
            raise RuntimeError("模型未正确初始化")
        ls = torch.sign(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        lc = torch.abs(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        rs = torch.sign(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        rc = torch.abs(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        return self._ensemble_predict(ls, lc, rs, rc)

    def get_prediction_with_terrain(self, full_features: np.ndarray):
        """带地形特征的预测 (FIELD_FEATURE_COUNT>0 时触发)"""
        result = self.get_individual_predictions_with_terrain(full_features)
        return result.get("ensemble", 0.5)

    def get_individual_predictions_with_terrain(self, full_features: np.ndarray):
        """返回每个模型的独立预测 + 集成平均值，用于 GUI 互相参照"""
        if not self.models:
            return {"ensemble": 0.5}
        N, F = MONSTER_COUNT, FIELD_FEATURE_COUNT
        lm = full_features[:N]; lt = full_features[N:N+F]
        rm = full_features[N+F:N*2+F]; rt = full_features[N*2+F:]
        ls = torch.cat([torch.sign(torch.tensor(lm, dtype=torch.int16)), torch.ones(F, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        lc = torch.cat([torch.abs(torch.tensor(lm, dtype=torch.int16)), torch.tensor(lt, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        rs = torch.cat([torch.sign(torch.tensor(rm, dtype=torch.int16)), torch.ones(F, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        rc = torch.cat([torch.abs(torch.tensor(rm, dtype=torch.int16)), torch.tensor(rt, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        results = {}
        preds = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                p = model(ls, lc, rs, rc).item()
                p = float(np.clip(p, 0.0, 1.0))
                name = self.model_names[i] if i < len(self.model_names) else f"model_{i+1}"
                results[name] = p
                preds.append(p)
        # 加权集成
        total_w = sum(self.model_weights.get(n, 3.0) for n in results)
        results["ensemble"] = float(
            sum(p * self.model_weights.get(n, 3.0) for n, p in results.items()) / total_w
        ) if total_w > 0 and results else 0.5
        self._last_predictions = {k: v for k, v in results.items() if k != "ensemble"}
        return results

    def get_prediction_with_terrain(self, full_features: np.ndarray):
        if not self.models:
            raise RuntimeError("模型未正确初始化")
        N = MONSTER_COUNT
        F = FIELD_FEATURE_COUNT
        lm = full_features[:N]; lt = full_features[N:N+F]
        rm = full_features[N+F:N*2+F]; rt = full_features[N*2+F:]
        ls = torch.cat([torch.sign(torch.tensor(lm, dtype=torch.int16)),
                        torch.ones(F, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        lc = torch.cat([torch.abs(torch.tensor(lm, dtype=torch.int16)),
                        torch.tensor(lt, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        rs = torch.cat([torch.sign(torch.tensor(rm, dtype=torch.int16)),
                        torch.ones(F, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        rc = torch.cat([torch.abs(torch.tensor(rm, dtype=torch.int16)),
                        torch.tensor(rt, dtype=torch.int16)]).unsqueeze(0).to(self.device)
        return self._ensemble_predict(ls, lc, rs, rc)
