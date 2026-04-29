from hydra.utils import instantiate


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
        if not isinstance(logger_cfg, dict) or "_target_" not in logger_cfg:
            print(f"警告: logger 配置不完整，跳过: {logger_cfg}")
            continue
        try:
            logger_cfg_copy = dict(logger_cfg)
            # 确保 save_dir 被传入
            logger_cfg_copy["save_dir"] = logger_cfg_copy.get("save_dir", save_dir)
            logger_instance = instantiate(logger_cfg_copy)
            if logger_instance is not None:
                loggers.append(logger_instance)
        except Exception as e:
            print(f"警告: 创建 logger 失败: {logger_cfg.get('_target_', 'unknown')} - {e}")
    
    return loggers if loggers else None
