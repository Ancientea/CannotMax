from __future__ import annotations

import time
from datetime import datetime

import gradio as gr

from gradio_ml.infrastructure.config_manager import ConfigManager
from gradio_ml.infrastructure.models import (
    LogFilter,
    LogLevel,
    MetricQuery,
    OptimizerType,
    TrainingConfig,
    TrainingSessionStatus,
)
from gradio_ml.infrastructure.queues import LOG_QUEUE, METRIC_QUEUE, PROGRESS_QUEUE
from gradio_ml.service.log_service import LogService
from gradio_ml.service.metric_service import MetricService
from gradio_ml.service.train_control_service import TrainControlService


class TrainMonitorTab:
    def __init__(
        self,
        train_service: TrainControlService,
        metric_service: MetricService,
        log_service: LogService,
        config: ConfigManager,
    ):
        self._train_svc = train_service
        self._metric_svc = metric_service
        self._log_svc = log_service
        self._config = config
        self._training_fn = None

    def set_training_fn(self, fn) -> None:
        self._training_fn = fn

    def render(self) -> None:
        constraints = self._config.training.constraints
        defaults = self._config.training

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 训练参数")
                self.lr_input = gr.Number(
                    label="学习率",
                    value=defaults.default_learning_rate,
                    precision=6,
                )
                self.epochs_input = gr.Number(
                    label="训练轮数",
                    value=defaults.default_epochs,
                    precision=0,
                )
                self.batch_size_input = gr.Dropdown(
                    choices=[str(x) for x in constraints.batch_size_options],
                    label="批次大小",
                    value=str(defaults.default_batch_size),
                )
                self.optimizer_input = gr.Dropdown(
                    choices=constraints.optimizer_options,
                    label="优化器",
                    value=defaults.default_optimizer,
                )

                gr.Markdown("### 控制按钮")
                self.start_btn = gr.Button("启动训练", variant="primary")
                self.pause_btn = gr.Button("暂停", interactive=False)
                self.stop_btn = gr.Button("终止", variant="stop", interactive=False)
                self.save_model_check = gr.Checkbox(label="终止时保存模型", value=False)
                self.train_status = gr.Textbox(label="训练状态", value="Idle", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 训练进度")
                self.progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="进度", interactive=False,
                )
                self.progress_text = gr.Textbox(
                    label="进度信息", value="", interactive=False,
                )
                self.eta_text = gr.Textbox(
                    label="预估剩余时间", value="", interactive=False,
                )

                gr.Markdown("### 指标监控")
                self.metric_checkboxes = gr.CheckboxGroup(
                    choices=self._config.metrics.default_monitored,
                    value=self._config.metrics.default_monitored,
                    label="监控指标",
                )
                self.smoothing_check = gr.Checkbox(
                    label="启用平滑",
                    value=self._config.metrics.smoothing_enabled,
                )
                self.smoothing_window = gr.Slider(
                    minimum=2,
                    maximum=self._config.metrics.smoothing_max_window,
                    value=self._config.metrics.smoothing_default_window,
                    step=1,
                    label="平滑窗口",
                )
                self.metric_plot = gr.LinePlot(
                    x_title="Epoch",
                    y_title="Value",
                    title="训练指标",
                )

                gr.Markdown("### 训练日志")
                self.log_level_filter = gr.Dropdown(
                    choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
                    value="INFO",
                    label="日志级别",
                )
                self.log_search = gr.Textbox(label="关键词搜索", placeholder="输入搜索关键词...")
                self.log_output = gr.Textbox(
                    label="日志",
                    lines=15,
                    interactive=False,
                    max_lines=50,
                )

        self._bind_events()

    def _bind_events(self) -> None:
        self.start_btn.click(
            fn=self._on_start,
            inputs=[self.lr_input, self.epochs_input, self.batch_size_input, self.optimizer_input],
            outputs=[self.train_status, self.start_btn, self.pause_btn, self.stop_btn],
        ).then(
            fn=self._monitor_generator,
            inputs=None,
            outputs=[
                self.progress_bar,
                self.progress_text,
                self.eta_text,
                self.metric_plot,
                self.log_output,
                self.train_status,
                self.start_btn,
                self.pause_btn,
                self.stop_btn,
            ],
        )
        self.pause_btn.click(
            fn=self._on_pause,
            inputs=None,
            outputs=[self.train_status, self.pause_btn],
        )
        self.stop_btn.click(
            fn=self._on_stop,
            inputs=self.save_model_check,
            outputs=[self.train_status, self.start_btn, self.pause_btn, self.stop_btn],
        )
        self.log_level_filter.change(
            fn=self._on_log_filter_change,
            inputs=self.log_level_filter,
            outputs=self.log_output,
        )

    def _on_start(self, lr, epochs, batch_size_str, optimizer_str):
        try:
            optimizer_map = {
                "SGD": OptimizerType.Sgd,
                "Adam": OptimizerType.Adam,
                "AdamW": OptimizerType.AdamW,
                "RMSprop": OptimizerType.RMSprop,
            }
            config = TrainingConfig(
                learning_rate=float(lr),
                epochs=int(epochs),
                batch_size=int(batch_size_str),
                optimizer=optimizer_map.get(optimizer_str, OptimizerType.Adam),
            )
            session = self._train_svc.start_training(config, training_fn=self._training_fn)
            self._metric_svc.register_metrics(list(config.validate()) or self._config.metrics.default_monitored)
            self._metric_svc.set_training_active(True)
            return (
                TrainingSessionStatus.Running.value,
                gr.update(interactive=False),
                gr.update(interactive=True, value="暂停"),
                gr.update(interactive=True),
            )
        except Exception as e:
            return (
                f"启动失败: {e}",
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    def _on_pause(self):
        active = self._train_svc.get_active_session()
        if active is None:
            return "Idle", gr.update(value="暂停")
        if active.status == TrainingSessionStatus.Running:
            self._train_svc.pause_training(active.session_id)
            return TrainingSessionStatus.Paused.value, gr.update(value="继续")
        elif active.status == TrainingSessionStatus.Paused:
            self._train_svc.resume_training(active.session_id)
            return TrainingSessionStatus.Running.value, gr.update(value="暂停")
        return active.status.value, gr.update(value="暂停")

    def _on_stop(self, save_model: bool):
        active = self._train_svc.get_active_session()
        if active is not None:
            self._train_svc.stop_training(active.session_id, save_model=save_model)
            self._metric_svc.set_training_active(False)
        return (
            TrainingSessionStatus.Terminated.value,
            gr.update(interactive=True),
            gr.update(interactive=False, value="暂停"),
            gr.update(interactive=False),
        )

    def _on_log_filter_change(self, level_str: str):
        if level_str == "ALL":
            log_filter = None
        else:
            level_map = {
                "DEBUG": LogLevel.Debug,
                "INFO": LogLevel.Info,
                "WARNING": LogLevel.Warning,
                "ERROR": LogLevel.Error,
            }
            log_filter = LogFilter(min_level=level_map.get(level_str, LogLevel.Info))
        return self._log_svc.get_logs(log_filter)

    def _monitor_generator(self):
        refresh_interval = self._config.metrics.refresh_interval
        stagnation_threshold = self._config.progress.stagnation_threshold

        while True:
            active = self._train_svc.get_active_session()
            status = active.status if active else TrainingSessionStatus.Idle

            while not METRIC_QUEUE.empty():
                try:
                    raw = METRIC_QUEUE.get_nowait()
                    self._metric_svc.receive(raw)
                    if active:
                        self._train_svc.session_manager.update_progress(
                            active.session_id, raw.epoch, raw.step
                        )
                except Exception:
                    break

            while not LOG_QUEUE.empty():
                try:
                    LOG_QUEUE.get_nowait()
                except Exception:
                    break

            progress_pct = 0.0
            progress_txt = ""
            eta_txt = ""
            if active and active.status == TrainingSessionStatus.Running:
                total = active.config.epochs
                current = active.current_epoch
                progress_pct = (current / total * 100) if total > 0 else 0
                progress_txt = f"Epoch {current}/{total} - 训练中"

                if active.last_epoch_callback_time:
                    elapsed = (datetime.now() - active.last_epoch_callback_time).total_seconds()
                    if elapsed > stagnation_threshold:
                        progress_txt += " ⚠️ 进度停滞"

            plot_data = None
            monitored = self._config.metrics.default_monitored
            if monitored:
                query = MetricQuery(metric_names=monitored)
                if self.smoothing_check.value:
                    data = self._metric_svc.get_smoothed_data(query, int(self.smoothing_window.value))
                else:
                    data = self._metric_svc.get_metric_data(query)
                all_points = []
                for name, points in data.items():
                    for p in points:
                        all_points.append({"epoch": p.epoch, "value": p.value, "metric": name})
                if all_points:
                    import pandas as pd
                    plot_data = pd.DataFrame(all_points)

            log_text = self._log_svc.get_logs()

            is_done = status in (TrainingSessionStatus.Completed, TrainingSessionStatus.Terminated, TrainingSessionStatus.Failed, TrainingSessionStatus.Idle)

            start_interactive = is_done
            pause_interactive = status == TrainingSessionStatus.Running or status == TrainingSessionStatus.Paused
            stop_interactive = status == TrainingSessionStatus.Running or status == TrainingSessionStatus.Paused
            pause_label = "继续" if status == TrainingSessionStatus.Paused else "暂停"

            yield (
                progress_pct,
                progress_txt,
                eta_txt,
                plot_data,
                log_text,
                status.value,
                gr.update(interactive=start_interactive),
                gr.update(interactive=pause_interactive, value=pause_label),
                gr.update(interactive=stop_interactive),
            )

            if is_done:
                break

            time.sleep(refresh_interval)
