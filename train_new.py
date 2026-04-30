from datetime import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
import importlib

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import lightning as L

from callbacks import CompatibilityCheckpoint, TrainingLogCallback
from config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from data_module import ArknightsDataModule
from logger_utils import create_composite_logger
from training_module import ArknightsLightningModule


def get_accelerator_and_device():
    if torch.cuda.is_available():
        return "cuda", 1
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", 1
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", 1
    return "cpu", 1


def get_precision(accelerator: str) -> Literal["16-mixed", "32-true"]:
    if accelerator == "cuda":
        return "16-mixed"
    return "32-true"


def plot_learning_curve(save_dir: str, output_name: str | None = None, logger_name: str = "lightning_logs"):
    save_path = Path(save_dir)

    csv_dir = save_path / logger_name
    csv_files = list(csv_dir.rglob("metrics.csv")) if csv_dir.exists() else []

    if not csv_files:
        latest_csv = sorted(csv_files)[-1] if csv_files else None
    else:
        latest_csv = sorted(csv_files)[-1]

    if latest_csv is None:
        tb_dir = save_path / "lightning_logs"
        tb_version_dirs = sorted(tb_dir.glob("version_*")) if tb_dir.exists() else []
        if tb_version_dirs:
            try:
                EventAccumulator = importlib.import_module(
                    "tensorboard.backend.event_processing.event_accumulator"
                ).EventAccumulator

                ea = EventAccumulator(str(tb_version_dirs[-1]))
                ea.Reload()

                tags = ea.Tags().get("scalars", [])
                metrics = {}
                for tag in ["train_loss", "val_loss", "train_acc", "val_acc"]:
                    if tag in tags:
                        events = ea.Scalars(tag)
                        metrics[tag] = [e.value for e in events]

                if metrics:
                    _plot_from_dict(metrics, save_path, output_name)
                    return
            except (ImportError, ModuleNotFoundError):
                pass

        print("警告: 未找到可用的训练日志数据，跳过学习曲线绘制")
        return

    pd = importlib.import_module("pandas")

    df = pd.read_csv(latest_csv)
    metrics = {}
    for col in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        if col in df.columns:
            metrics[col] = df[col].dropna().values.tolist()

    if not metrics:
        print("警告: CSV 中未找到训练指标，跳过学习曲线绘制")
        return

    _plot_from_dict(metrics, save_path, output_name)


def _plot_from_dict(metrics: dict, save_path: Path, output_name: str | None = None):
    if output_name is None:
        current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_name = f"learning_curve_{current_time_str}.png"

    output_path = save_path / output_name

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train_loss" in metrics and "val_loss" in metrics:
        ax = axes[0]
        epochs = range(1, len(metrics["train_loss"]) + 1)
        ax.plot(epochs, metrics["train_loss"], label="Train Loss", color="blue")
        val_epochs = range(1, len(metrics["val_loss"]) + 1)
        ax.plot(val_epochs, metrics["val_loss"], label="Val Loss", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Curve - Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if "train_acc" in metrics and "val_acc" in metrics:
        ax = axes[1]
        epochs = range(1, len(metrics["train_acc"]) + 1)
        ax.plot(epochs, metrics["train_acc"], label="Train Acc", color="blue")
        val_epochs = range(1, len(metrics["val_acc"]) + 1)
        ax.plot(val_epochs, metrics["val_acc"], label="Val Acc", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Learning Curve - Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"学习曲线已保存: {output_path}")


def _flatten_params(data, prefix: str = "", sep: str = ".") -> dict[str, object]:
    flattened: dict[str, object] = {}

    if isinstance(data, Mapping):
        for key, value in data.items():
            full_key = f"{prefix}{sep}{key}" if prefix else str(key)
            flattened.update(_flatten_params(value, full_key, sep=sep))
    else:
        flattened[prefix] = data

    return flattened


def log_config_to_loggers(logger_instance, cfg: DictConfig):
    """将训练配置写入所有 logger，方便在面板中查看超参数。"""
    if logger_instance is None:
        return

    config_payload: dict[str, object] = {}
    section_display_names = {
        "model": "MODEL_YAML",
        "data": "DATA_YAML",
        "runtime": "RUNTIME_YAML",
        "trainer": "TRAINER_YAML",
    }

    for section_name in ("model", "data", "runtime", "trainer"):
        section_cfg = cfg.get(section_name)
        if section_cfg is None:
            continue

        group_name = section_display_names.get(section_name, section_name.upper())
        section_container = OmegaConf.to_container(section_cfg, resolve=True)
        if isinstance(section_container, Mapping):
            section_payload = _flatten_params(section_container, group_name, sep="/")
            config_payload.update(section_payload)
        else:
            config_payload[group_name] = section_container

        config_payload[f"{group_name}/__source__"] = f"conf/{section_name}/{section_name}.yaml"

    if not config_payload:
        return

    loggers = logger_instance if isinstance(logger_instance, list) else [logger_instance]
    for single_logger in loggers:
        if single_logger is None or not hasattr(single_logger, "log_hyperparams"):
            continue
        try:
            single_logger.log_hyperparams(config_payload)
        except Exception as e:
            print(f"警告: 写入 logger 超参数失败: {e}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig | None = None):
    if cfg is None:
        return

    print(f"配置:\n{OmegaConf.to_yaml(cfg)}")

    L.seed_everything(cfg.runtime.seed, workers=True)

    accelerator, devices = get_accelerator_and_device()
    precision = get_precision(accelerator)

    print(f"使用设备: {accelerator}")
    print(f"场地特征数量: {FIELD_FEATURE_COUNT}")

    if accelerator == "cuda":
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif accelerator == "cpu":
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")

    save_dir = cfg.runtime.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data_module = ArknightsDataModule(cfg.data, seed=cfg.runtime.seed)

    lightning_module = ArknightsLightningModule(cfg.model, cfg.trainer)

    total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
    print(f"模型使用特征数: 怪物({MONSTER_COUNT}) + 场地({FIELD_FEATURE_COUNT}) = {total_units}")
    if cfg.get("model") is not None and cfg.model.get("dropout") is not None:
        print(f"模型Dropout: {cfg.model.dropout}")
    print(
        f"模型参数数量: {sum(p.numel() for p in lightning_module.model.parameters() if p.requires_grad)}"
    )

    # 使用 Hydra instantiate 创建 logger
    logger_cfg = cfg.get("logger", None)
    if logger_cfg is not None:
        if logger_cfg.get("_target_") == "logger_utils.create_composite_logger":
            logger_instance = create_composite_logger(logger_cfg, save_dir)
        else:
            logger_cfg_copy = dict(logger_cfg)
            logger_cfg_copy["save_dir"] = save_dir
            logger_instance = instantiate(logger_cfg_copy)
    else:
        logger_instance = None

    log_config_to_loggers(logger_instance, cfg)

    compat_checkpoint = CompatibilityCheckpoint(save_dir=save_dir)
    training_log_callback = TrainingLogCallback(total_epochs=cfg.trainer.epochs)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=0.0,
        callbacks=[compat_checkpoint, training_log_callback],
        default_root_dir=save_dir,
        logger=logger_instance,
    )

    trainer.fit(lightning_module, datamodule=data_module)

    val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0))
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf")))
    print(f"训练完成! 最佳验证准确率: {val_acc:.2f}%, 最佳验证损失: {val_loss:.4f}")

    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_filename = f"acc{val_acc:.4f}_loss{val_loss:.4f}_{current_time_str}.pth"

    plot_learning_curve(save_dir, output_name=f"learning_curve_{base_filename}.png")

    save_dir_path = Path(save_dir)
    for prefix in ["best_model_acc", "best_model_loss", "best_model_full"]:
        old_path = save_dir_path / f"{prefix}.pth"
        if old_path.exists():
            new_path = save_dir_path / f"{prefix}_{base_filename}"
            old_path.rename(new_path)
            print(f"模型文件已重命名: {old_path} -> {new_path}")


if __name__ == "__main__":
    main()
