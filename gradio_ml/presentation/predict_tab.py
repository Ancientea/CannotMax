from __future__ import annotations

import gradio as gr
import numpy as np

from gradio_ml.infrastructure.config_manager import ConfigManager
from gradio_ml.infrastructure.models import PredictInput, PredictResult
from gradio_ml.service.predict_service import PredictService


class PredictTab:
    def __init__(self, predict_service: PredictService, config: ConfigManager):
        self._service = predict_service
        self._config = config

    def render(self) -> None:
        model_choices = self._service.list_available_models(self._config.predict.default_model_path)

        with gr.Row():
            with gr.Column(scale=1):
                self.model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="选择模型",
                    value=model_choices[0] if model_choices else None,
                )
                self.load_btn = gr.Button("加载模型", variant="secondary")
                self.model_status = gr.Textbox(label="模型状态", value="未加载", interactive=False)

                gr.Markdown("### 输入数据")
                self.input_mode = gr.Radio(
                    choices=["左右数据", "完整特征"],
                    value="左右数据",
                    label="输入模式",
                )
                self.left_data = gr.Textbox(
                    label="左侧数据（逗号分隔）",
                    placeholder="1,2,3,...",
                )
                self.right_data = gr.Textbox(
                    label="右侧数据（逗号分隔）",
                    placeholder="1,2,3,...",
                )
                self.full_features = gr.Textbox(
                    label="完整特征向量（逗号分隔）",
                    placeholder="1,2,3,...",
                    visible=False,
                )
                self.predict_btn = gr.Button("预测", variant="primary")
                self.clear_btn = gr.Button("清空")

            with gr.Column(scale=1):
                gr.Markdown("### 预测结果")
                self.result_prediction = gr.Number(label="预测值", interactive=False)
                self.result_label = gr.Textbox(label="预测标签", interactive=False)
                self.result_confidence = gr.Number(label="置信度", interactive=False)
                self.result_time = gr.Number(label="推理耗时(ms)", interactive=False)
                self.result_model = gr.Textbox(label="使用模型", interactive=False)
                self.error_msg = gr.Textbox(label="错误信息", visible=False)

        self._bind_events()

    def _bind_events(self) -> None:
        self.input_mode.change(
            fn=self._toggle_input_mode,
            inputs=self.input_mode,
            outputs=[self.left_data, self.right_data, self.full_features],
        )
        self.load_btn.click(
            fn=self._on_load_model,
            inputs=self.model_dropdown,
            outputs=self.model_status,
        )
        self.predict_btn.click(
            fn=self._on_predict,
            inputs=[self.input_mode, self.left_data, self.right_data, self.full_features],
            outputs=[
                self.result_prediction,
                self.result_label,
                self.result_confidence,
                self.result_time,
                self.result_model,
                self.error_msg,
            ],
        )
        self.clear_btn.click(
            fn=self._on_clear,
            inputs=None,
            outputs=[
                self.left_data,
                self.right_data,
                self.full_features,
                self.result_prediction,
                self.result_label,
                self.result_confidence,
                self.result_time,
                self.result_model,
                self.error_msg,
            ],
        )

    def _toggle_input_mode(self, mode: str):
        if mode == "左右数据":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def _on_load_model(self, model_path: str) -> str:
        if not model_path:
            return "请选择模型"
        try:
            info = self._service.load_model(model_path)
            return f"已加载: {info.name} ({info.status.value})"
        except Exception as e:
            return f"加载失败: {e}"

    def _on_predict(self, mode: str, left_str: str, right_str: str, features_str: str):
        try:
            if mode == "左右数据":
                left = [float(x.strip()) for x in left_str.split(",") if x.strip()]
                right = [float(x.strip()) for x in right_str.split(",") if x.strip()]
                input_data = PredictInput(left_counts=left, right_counts=right)
            else:
                features = [float(x.strip()) for x in features_str.split(",") if x.strip()]
                input_data = PredictInput(full_features=features)

            result: PredictResult = self._service.predict(input_data)
            return (
                result.prediction,
                result.label,
                result.confidence,
                round(result.inference_time_ms, 2),
                result.model_name,
                gr.update(visible=False, value=""),
            )
        except Exception as e:
            return 0.0, "", 0.0, 0.0, "", gr.update(visible=True, value=str(e))

    def _on_clear(self):
        return "", "", "", 0.0, "", 0.0, 0.0, "", gr.update(visible=False, value="")
