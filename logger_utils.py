from collections.abc import Mapping

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def create_composite_logger(cfg, save_dir):
    """
    创建复合 logger，支持同时使用多个 logger
    
    配置示例:
    _target_: logger_utils.create_composite_logger
    loggers:
      - _target_: lightning.pytorch.loggers.TensorBoardLogger
        name: lightning_logs
      - _target_: lightning.pytorch.loggers.CSVLogger
        name: csv_logs
    """
    loggers = []
    for logger_cfg in cfg.get("loggers", []):
        if isinstance(logger_cfg, DictConfig):
            logger_cfg_copy = OmegaConf.to_container(logger_cfg, resolve=True)
        elif isinstance(logger_cfg, Mapping):
            logger_cfg_copy = dict(logger_cfg)
        else:
            print(f"警告: logger 配置不完整，跳过: {logger_cfg}")
            continue

        if not isinstance(logger_cfg_copy, dict) or "_target_" not in logger_cfg_copy:
            print(f"警告: logger 配置不完整，跳过: {logger_cfg}")
            continue

        try:
            # 确保 save_dir 被传入
            logger_cfg_copy["save_dir"] = logger_cfg_copy.get("save_dir", save_dir)
            logger_instance = instantiate(logger_cfg_copy)
            if logger_instance is not None:
                loggers.append(logger_instance)
        except Exception as e:
            print(f"警告: 创建 logger 失败: {logger_cfg.get('_target_', 'unknown')} - {e}")
    
    return loggers if loggers else None
