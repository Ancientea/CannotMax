def _check_gradio_deps() -> None:
    missing: list[str] = []
    try:
        import gradio  # noqa: F401
    except ImportError:
        missing.append("gradio")
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    if missing:
        raise ImportError(
            f"Gradio可视化依赖未安装: {', '.join(missing)}。"
            f"请执行: uv sync --extra gradio"
        )


def main():
    _check_gradio_deps()

    from gradio_ml.infrastructure.config_manager import ConfigManager
    from gradio_ml.infrastructure.log_buffer import LogBuffer
    from gradio_ml.infrastructure.metric_store import MetricStore
    from gradio_ml.adapter.model_adapter import ModelAdapter
    from gradio_ml.service.log_service import LogService
    from gradio_ml.service.metric_service import MetricService
    from gradio_ml.service.predict_service import PredictService
    from gradio_ml.service.train_control_service import TrainControlService
    from gradio_ml.presentation.app import GradioApp
    from gradio_ml.presentation.predict_tab import PredictTab
    from gradio_ml.presentation.train_monitor_tab import TrainMonitorTab

    config = ConfigManager("config/gradio_config.yaml")

    model_adapter = ModelAdapter()
    predict_service = PredictService(model_adapter, inference_timeout=config.predict.inference_timeout)

    metric_store = MetricStore(downsampling_threshold=config.metrics.downsampling_threshold)
    metric_service = MetricService(metric_store)

    log_buffer = LogBuffer(
        capacity=config.log.buffer_size,
        rate_threshold=config.log.rate_threshold,
        sampling_ratio=config.log.sampling_ratio,
    )
    log_service = LogService(log_buffer, sanitize_patterns=config.log.sanitize_patterns)

    train_control_service = TrainControlService()

    predict_tab = PredictTab(predict_service, config)
    train_monitor_tab = TrainMonitorTab(train_control_service, metric_service, log_service, config)

    app = GradioApp(config)
    app.build(predict_tab=predict_tab, train_monitor_tab=train_monitor_tab)

    print(f"启动 Gradio 可视化服务: http://{config.server.host}:{config.server.port}")
    app.launch()


if __name__ == "__main__":
    main()
