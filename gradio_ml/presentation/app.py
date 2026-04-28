from __future__ import annotations

import gradio as gr

from gradio_ml.infrastructure.config_manager import ConfigManager
from gradio_ml.presentation.predict_tab import PredictTab
from gradio_ml.presentation.train_monitor_tab import TrainMonitorTab


class GradioApp:
    def __init__(self, config: ConfigManager | None = None):
        self._config = config or ConfigManager()
        self._predict_tab: PredictTab | None = None
        self._train_monitor_tab: TrainMonitorTab | None = None

    @property
    def predict_tab(self) -> PredictTab | None:
        return self._predict_tab

    @property
    def train_monitor_tab(self) -> TrainMonitorTab | None:
        return self._train_monitor_tab

    def build(
        self,
        predict_tab: PredictTab | None = None,
        train_monitor_tab: TrainMonitorTab | None = None,
    ) -> gr.Blocks:
        self._predict_tab = predict_tab
        self._train_monitor_tab = train_monitor_tab

        with gr.Blocks(
            title="CannotMax ML 可视化",
            theme=gr.themes.Soft(),
        ) as app:
            with gr.Tabs():
                if self._config.tabs.predict_enabled and predict_tab is not None:
                    with gr.Tab("推理预测"):
                        predict_tab.render()

                if self._config.tabs.train_monitor_enabled and train_monitor_tab is not None:
                    with gr.Tab("训练监控"):
                        train_monitor_tab.render()

            self._app = app
        return self._app

    def launch(self, **kwargs) -> None:
        if not hasattr(self, "_app"):
            raise RuntimeError("请先调用 build() 构建界面")

        server_config = self._config.server
        launch_kwargs = {
            "server_name": server_config.host,
            "server_port": server_config.port,
            "share": server_config.share,
        }
        launch_kwargs.update(kwargs)
        self._app.launch(**launch_kwargs)
