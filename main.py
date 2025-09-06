import json
import logging

import subprocess
import sys
import time
import toml
import numpy as np
from pathlib import Path
import onnxruntime  # workaround: Pre-import to avoid ImportError: DLL load failed while importing onnxruntime_pybind11_state: 动态链接库(DLL)初始化例程失败。
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox
from PyQt6.QtWidgets import QGroupBox, QMessageBox, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPainter, QColor
import PyQt6.QtCore as QtCore

import loadData
import auto_fetch
import similar_history_match
import recognize
from recognize import MONSTER_COUNT
from specialmonster import SpecialMonsterHandler
import data_package
import winrt_capture
from config import FIELD_FEATURE_COUNT, MONSTER_DATA
from simular_history_match_ui import HistoryMatchUI
from input_panel_ui import InputPanelUI

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from predict import CannotModel
    from train import UnitAwareTransformer

    logger.info("Using PyTorch model for predictions.")
except:
    from predict_onnx import CannotModel

    logger.info("Using ONNX model for predictions.")


class ADBConnectorThread(QThread):
    """
    Worker thread to run loadData.AdbConnector.connect() without blocking the UI.
    """

    connect_finished = pyqtSignal()

    def __init__(self, app: "ArknightsApp"):
        super().__init__()
        self.app = app

    def run(self):
        self.app.adb_connector.connect()
        self.connect_finished.emit()

class ArknightsApp(QMainWindow):
    # 添加自定义信号
    update_button_signal = pyqtSignal(str)  # 用于更新按钮文本
    update_monster_signal = pyqtSignal(list)
    update_prediction_signal = pyqtSignal(float)
    update_statistics_signal = pyqtSignal()  # 用于更新统计信息
    qt_button_style = """
        QPushButton {
            background-color: #313131;
            color: #F3F31F;
            border-radius: 16px;
            padding: 8px;
            font-weight: bold;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #414141;
        }
        QPushButton:pressed {
            background-color: #212121;
        }
    """

    def __init__(self):
        super().__init__()
        # 尝试连接模拟器
        self.adb_connector = loadData.AdbConnector()
        self.adb_connector_thread = ADBConnectorThread(self)
        self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
        self.adb_connector_thread.start()

        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = False
        self.game_mode = "单人"

        # 模型
        self.cannot_model = CannotModel()

        # 怪物识别模块
        self.recognizer = recognize.RecognizeMonster()

        # 初始化UI后加载历史数据
        logger.info("尝试获取错题本")
        self.history_match = None
        self.history_match = similar_history_match.HistoryMatch()
        # Ensure feat_past and N_history are initialized
        try:
            self.history_match.feat_past = np.hstack([self.history_match.past_left, self.history_match.past_right])
        except Exception:
            self.history_match.feat_past = None
        self.history_match.N_history = 0 if self.history_match.labels is None else len(self.history_match.labels)
        logger.info("错题本加载成功")

        # 初始化特殊怪物语言触发处理程序
        self.special_monster_handler = SpecialMonsterHandler()

        self.init_ui()

    def init_ui(self):
        try:
            with open("pyproject.toml", "r", encoding="utf-8") as f:
                pyproject_data = toml.load(f)
                version = pyproject_data["project"]["version"]
        except (FileNotFoundError, KeyError):
            version = "unknown"
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.setWindowTitle(f"铁鲨鱼_Arknights Neural Network - v{version} - model: {model_name}")
        self.setWindowIcon(QIcon("ico/icon.ico"))
        self.setGeometry(100, 100, 500, 580)
        self.setMinimumWidth(580)
        self.setMaximumWidth(580)
        self.background = QPixmap("ico/background.png")

        # 初始化动画对象
        self.size_animation = QPropertyAnimation(self, b"size")
        self.size_animation.setDuration(300)
        self.size_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 左侧面板
        self.input_panel = InputPanelUI()
        self.input_panel.setFixedWidth(528)
        self.input_panel.predict_requested.connect(self.predict)
        self.input_panel.reset_requested.connect(self.reset_entries)
        self.input_panel.input_changed.connect(self.update_input_display)

        # 右侧面板 - 结果和控制区
        right_panel = QWidget()
        right_panel.setFixedWidth(550)  # 固定右侧面板宽度
        right_layout = QVBoxLayout(right_panel)

        # 顶部区域 - 输入显示
        input_display = QGroupBox()
        input_display.setStyleSheet(
            """
                QGroupBox {
                    background-color: rgba(0, 0, 0, 120);
                    border-radius: 15px;
                    border: 5px solid #F5EA2D;
                    margin-top: 10px;
                    padding: 10px 0;
                }
                QGroupBox::title {
                    color: white;
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 0 5px;
                }
            """
        )
        input_layout = QHBoxLayout(input_display)

        # 左侧人物显示
        left_input_group = QWidget()
        left_input_layout = QHBoxLayout(left_input_group)
        self.left_input_content = QWidget()
        self.left_input_layout = QHBoxLayout(self.left_input_content)
        self.left_input_layout.setSpacing(5)
        left_input_layout.addWidget(self.left_input_content)

        # 右侧人物显示
        right_input_group = QWidget()
        right_input_layout = QHBoxLayout(right_input_group)
        self.right_input_content = QWidget()
        self.right_input_layout = QHBoxLayout(self.right_input_content)
        self.right_input_layout.setSpacing(5)
        right_input_layout.addWidget(self.right_input_content)

        # 将左右两部分添加到主输入布局
        input_layout.addWidget(left_input_group)
        input_layout.addWidget(right_input_group)

        right_layout.addWidget(input_display)

        # 中部区域 - 预测结果
        result_group = QGroupBox()
        result_group.setStyleSheet(
            """
            QGroupBox {
                background-color: rgba(120, 120, 120, 10);
                border-radius: 15px;
                border: 1px solid #747474;
            }
            """
        )
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setFont(QFont("Microsoft YaHei", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_label)

        # 添加模型名称显示
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.model_name_label = QLabel(f"model: {model_name}")
        self.model_name_label.setFont(QFont("Microsoft YaHei", 8))
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.model_name_label.setStyleSheet("color: #888888;")  # 小字灰色
        result_layout.addWidget(self.model_name_label)

        # 第二行按钮result_identify_group
        result_identify_group = QWidget()
        result_identify_layout = QHBoxLayout(result_identify_group)

        self.recognize_button = QPushButton("识别并预测")
        self.recognize_button.clicked.connect(self.recognize_and_predict)
        self.recognize_button.setStyleSheet(self.qt_button_style)
        result_identify_layout.addWidget(self.recognize_button)
        result_layout.addWidget(result_identify_group)

        right_layout.addWidget(result_group)

        # 底部区域 - 控制面板
        self.bottom_group = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_group)

        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        # 第一行按钮
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.duration_label = QLabel("训练时长(小时):")
        self.duration_entry = QLineEdit("-1")
        self.duration_entry.setFixedWidth(50)

        self.auto_fetch_button = QPushButton("自动获取数据")
        self.auto_fetch_button.clicked.connect(self.toggle_auto_fetch)

        self.mode_menu = QComboBox()
        self.mode_menu.addItems(["单人", "30人"])
        self.mode_menu.currentTextChanged.connect(self.update_game_mode)

        self.invest_checkbox = QCheckBox("投资")
        self.invest_checkbox.stateChanged.connect(self.update_invest_status)

        row1_layout.addWidget(self.duration_label)
        row1_layout.addWidget(self.duration_entry)
        row1_layout.addWidget(self.auto_fetch_button)
        row1_layout.addWidget(self.mode_menu)
        row1_layout.addWidget(self.invest_checkbox)

        # 第三行按钮
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)

        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QLineEdit()
        self.serial_entry.setFixedWidth(100)
        self.serial_entry.setPlaceholderText("127.0.0.1:5555")  # 设置默认灰色文本

        self.serial_button = QPushButton("更新")
        self.serial_button.clicked.connect(self.update_device_serial)

        self.package_data_button = QPushButton("数据打包")
        self.package_data_button.clicked.connect(self.package_data_and_show)
        row2_layout.addWidget(self.package_data_button)
        row2_layout.addWidget(self.serial_label)
        row2_layout.addWidget(self.serial_entry)
        row2_layout.addWidget(self.serial_button)

        # 第三行按钮
        row3 = QWidget()
        row3_layout = QHBoxLayout(row3)
        row3_layout.setContentsMargins(0, 0, 0, 0)

        self.reselect_button = QPushButton("选择范围")
        self.reselect_button.clicked.connect(self.reselect_roi)
        self.choose_window_button = QPushButton("选择截屏窗口")
        self.choose_window_button.clicked.connect(self.choose_capture_window)

        row3_layout.addWidget(self.choose_window_button)
        row3_layout.addWidget(self.reselect_button)

        # 第四行 - 地形选择
        row4 = QWidget()
        row4_layout = QHBoxLayout(row4)

        # 地形选择标签
        terrain_label = QLabel("地形选择:")
        terrain_label.setStyleSheet("color: #414141; font-weight: bold;")
        row3_layout.addWidget(terrain_label)

        # 创建地形选择按钮组
        self.terrain_group = QWidget()
        terrain_group_layout = QHBoxLayout(self.terrain_group)
        terrain_group_layout.setSpacing(5)

        # 地形选项：无地形 + 6种地形
        self.terrain_buttons = {}
        terrain_options = [
            ("常规地形", "none"),
            ("中路阻挡", "middle_row_blocks_blocks"),
            ("侧边弩箭", "side_fire_cannon_crossbow"),
            ("侧边火炮", "side_fire_cannon_fire"),
            ("顶部弩箭", "top_crossbow_crossbow"),
            ("顶部火炮", "top_fire_cannon_fire"),
            ("双行阻挡", "two_row_blocks_blocks"),
        ]

        for text, key in terrain_options:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #7B7B7B;
                    color: #FAFAFA;
                    border-radius: 8px;
                    padding: 4px 8px;
                    font-size: 10px;
                    min-height: 20px;
                }
                QPushButton:checked {
                    background-color: #F3F31F;
                    color: #313131;
                    border: 2px solid #7B7B7B;
                    padding: 1px
                }
                QPushButton:hover {
                    border: 2px solid #616161;
                    background-color: #c0c00c;
                    color: #313131;
                    padding: 1px
                }
                """
            )
            btn.clicked.connect(lambda checked, b=btn, k=key: self.on_terrain_selected(b, k))
            self.terrain_buttons[key] = btn
            terrain_group_layout.addWidget(btn)

        # 默认选中"常规地形"
        self.terrain_buttons["none"].setChecked(True)

        row4_layout.addWidget(self.terrain_group)

        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))

        # 添加所有行到控制布局
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(row3)
        control_layout.addWidget(row4)

        # GitHub链接
        github_label = QLabel(
            '<a href="https://github.com/Ancientea/CannotMax" style="color: #2196F3; text-decoration: none;">https://github.com/Ancientea/CannotMax</a>'
        )
        github_label.setMargin(0)
        github_label.setAlignment(Qt.AlignmentFlag.AlignLeft)  # 改为左对齐
        github_label.setOpenExternalLinks(True)
        github_label.setFont(QFont("Microsoft YaHei", 9))
        github_label.setContentsMargins(0, 0, 0, 0)  # 减少内边距和外边距

        control_layout.addWidget(github_label)

        control_layout.addWidget(self.stats_label)

        # 第五行按钮 (其实是纵向的，懒得改名了)
        row5 = QWidget()
        row5_layout = QVBoxLayout(row5)

        self.simulate_button = QPushButton("显示沙盒模拟")
        self.simulate_button.clicked.connect(self.run_simulation)
        self.simulate_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.simulate_button)

        # 在右侧面板添加显示输入面板按钮
        self.toggle_input_button = QPushButton("显示输入面板")
        self.toggle_input_button.clicked.connect(self.toggle_input_panel)
        self.toggle_input_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.toggle_input_button)

        # 在右侧面板添加历史对局按钮
        self.history_button = QPushButton("显示历史对局")
        self.history_button.clicked.connect(self.toggle_history_panel)
        self.history_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.history_button)

        # 排布按钮
        self.bottom_layout.addWidget(control_group)
        self.bottom_layout.addWidget(row5)

        right_layout.addWidget(self.bottom_group)

        # 创建并添加HistoryMatchUI实例
        self.history_match_ui = HistoryMatchUI(self.history_match)
        self.history_match_ui.setVisible(False)  # 初始隐藏

        main_layout.addWidget(right_panel, 1)
        main_layout.addWidget(self.input_panel)
        main_layout.addWidget(self.history_match_ui)  # 添加到主布局

        self.setCentralWidget(main_widget)
        # 初始化输入面板状态
        self.input_panel_visible = False
        self.input_panel.setVisible(False)  # 默认折叠左侧输入面板

        # 连接AutoFetch信号到槽
        self.update_button_signal.connect(self.auto_fetch_button.setText)
        self.update_monster_signal.connect(self.update_monster)
        self.update_prediction_signal.connect(self.update_prediction)
        self.update_statistics_signal.connect(self.update_statistics)

    def toggle_input_panel(self):
        """切换输入面板的显示"""
        target_width = self.width()
        is_visible = self.input_panel.isVisible()
        self.input_panel.setVisible(not is_visible)
        if not is_visible:
            self.toggle_input_button.setText("隐藏输入面板")
            target_width += self.input_panel.width()
        else:
            self.toggle_input_button.setText("显示输入面板")
            target_width -= self.input_panel.width()
        self.animate_size_change(target_width)

    def animate_size_change(self, target_width, target_height=None):
        """通用的尺寸动画方法"""
        if target_height is None:
            target_height = self.height()
        if self.size_animation.state() == QPropertyAnimation.State.Running:
            self.size_animation.stop()

        self.setMinimumWidth(min(self.width(), target_width))
        self.setMaximumWidth(max(self.width(), target_width))

        self.size_animation.setStartValue(self.size())
        self.size_animation.setEndValue(QtCore.QSize(target_width, target_height))
        self.size_animation.start()

        def set_fixed_after_animation():
            self.setFixedWidth(self.width())

        self.size_animation.finished.connect(set_fixed_after_animation)

    def on_adb_connected(self):
        logger.info("模拟器初始化完成")

    def choose_capture_window(self):
        """弹出窗口选择器，切换 WinRT 截屏源（窗口标题或整屏）。"""
        import traceback, cv2

        if getattr(self, "_switching_source", False):
            return
        self._switching_source = True
        self.choose_window_button.setEnabled(False)
        try:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            dlg = winrt_capture.WindowPickerDialog(self)
            if dlg.exec():
                sel = dlg.get_selection()
                logger.info(f"选择了截屏源: {sel}")
                if not sel:
                    QMessageBox.information(self, "提示", "未选择任何项")
                    return
                hint = ""
                if "window_name" in sel:
                    self.recognizer = recognize.RecognizeMonster(window_name=sel["window_name"], monitor_index=None)
                    hint = f"已切换至窗口：{sel['window_name']}"
                else:
                    idx = max(1, sel["monitor_index"])
                    self.recognizer = recognize.RecognizeMonster(window_name=None, monitor_index=idx)
                    hint = f"已切换至整屏：显示器 {sel['monitor_index']}"

                self.no_region = True
                QMessageBox.information(self, "成功", hint + "\n建议重新选择范围。")
        except Exception as e:
            QMessageBox.critical(self, "异常", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._switching_source = False
            self.choose_window_button.setEnabled(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 缩放图片以适应窗口（保持宽高比）
        scaled_pixmap = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # 居中绘制
        painter.drawPixmap(
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap,
        )

    def update_input_display(self):
        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()

        def update_input_display_half(input_layout, monsters_dict):
            # 清除现有显示
            for i in reversed(range(input_layout.count())):
                widget = input_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            has_input = False
            for i in range(1, MONSTER_COUNT + 1):
                value = monsters_dict[str(i)].text()
                if value.isdigit() and int(value) > 0:
                    has_input = True
                    monster_widget = self.create_monster_display_widget(i, value)
                    input_layout.addWidget(monster_widget)
            # 如果没有输入，显示提示
            if not has_input:
                input_layout.addWidget(QLabel("无"))

        update_input_display_half(self.left_input_layout, left_monsters_dict)
        update_input_display_half(self.right_input_layout, right_monsters_dict)

    def create_monster_display_widget(self, monster_id, count):
        """创建人物显示组件"""
        widget = QWidget()
        widget.setFixedWidth(67)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(0)  # 模糊半径（控制发光范围）
        shadow.setColor(QColor("#313131"))  # 发光颜色
        shadow.setOffset(2)  # 偏移量（0表示均匀四周发光）
        widget.setGraphicsEffect(shadow)

        widget.setStyleSheet(
            """
                QWidget {
                    border-radius: 0px;
                }
            """
        )

        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 人物图片
        img_label = QLabel()
        img_label.setFixedSize(70, 70)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        try:
            pixmap = QPixmap(f"images/{MONSTER_DATA['原始名称'][monster_id]}.png")
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                img_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"加载人物{monster_id}图片错误: {str(e)}")
            pass

        # 添加鼠标悬浮提示
        if monster_id in MONSTER_DATA.index:
            data = MONSTER_DATA.loc[monster_id].to_dict()
            tooltip_text = ""
            for key, value in data.items():
                tooltip_text += f"{key}: {value}\n"
            img_label.setToolTip(tooltip_text.strip())

        # 数量标签
        count_label = QLabel(count)
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_label.setStyleSheet(
            """
            color: #EDEDED;
            font: bold 20px SimHei;
            border-radius: 5px;
            padding: 2px 5px;
            min-width: 20px;
        """
        )

        layout.addWidget(img_label)
        layout.addWidget(count_label)

        return widget

    def reset_entries(self):
        self.result_label.setText("预测结果将显示在这里")
        self.result_label.setStyleSheet("color: black;")
        self.update_input_display()

    def get_prediction(self):
        try:
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            for name, entry in left_monsters_dict.items():
                value = entry.text()
                left_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            for name, entry in right_monsters_dict.items():
                value = entry.text()
                right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            # 获取当前选择的地形
            current_terrain = self.get_current_terrain()

            # 构建包含地形的完整特征向量
            full_features = self.build_terrain_features(left_counts, right_counts, current_terrain)

            # 添加调试日志
            logger.info(f"当前地形: {current_terrain}")
            logger.info(f"完整特征向量长度: {len(full_features)}")
            logger.info(f"地形特征部分: {full_features[MONSTER_COUNT:MONSTER_COUNT+FIELD_FEATURE_COUNT]}")

            prediction = self.cannot_model.get_prediction_with_terrain(full_features)
            return prediction
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到模型文件，请先训练")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                QMessageBox.critical(self, "错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数字（0或正整数）")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时发生错误: {str(e)}")

        return 0.5

    def update_prediction(self, prediction):
        """更新预测结果显示"""
        # 模型结果处理
        right_win_prob = prediction
        left_win_prob = 1 - right_win_prob

        # 判断胜负方向
        winner = "左方" if left_win_prob > 0.5 else "右方"
        if 0.6 > left_win_prob > 0.4:
            winner = "难说"

        # 设置结果标签样式
        if winner == "左方":
            self.result_label.setStyleSheet("color: #E23F25; font: bold,14px;")
        else:
            self.result_label.setStyleSheet("color: #25ace2; font: bold,14px;")

        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
        # 生成结果文本
        if winner != "难说":
            result_text = f"预测胜方: {winner}\n" f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n"
        else:
            result_text = (
                f"这一把{winner}\n" f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n" f"难道说？难道说？难道说？\n"
            )
            self.result_label.setStyleSheet("color: black; font: bold,24px;")

        # 添加特殊干员提示
        special_messages = self.special_monster_handler.check_special_monsters(
            left_monsters_dict, right_monsters_dict, winner
        )
        if special_messages:
            result_text += "\n" + special_messages

        self.result_label.setText(result_text)

    def predict(self):
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        self.update_input_display()

        if self.history_match_ui.isVisible():
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)

    def recognize(self):
        if self.auto_fetch_running:
            screenshot = self.adb_connector.capture_screenshot()
        else:
            screenshot = None

        if self.no_region:  # TODO: 判断需要移至recognize
            if self.first_recognize:
                self.adb_connector.connect()
                self.first_recognize = False
            screenshot = self.adb_connector.capture_screenshot()  # TODO: 如果 self.no_region 为 True，则会被调用两次。

        results = self.recognizer.process_regions(screenshot)
        return results, screenshot

    def update_monster(self, results):
        """
        根据识别结果更新怪物面板
        """
        left_counts = {}
        right_counts = {}
        for res in results:
            if "error" not in res:
                region_id = res["region_id"]
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:
                        left_counts[str(matched_id)] = int(number)
                    else:
                        right_counts[str(matched_id)] = int(number)
        self.input_panel.set_monster_counts(left_counts, right_counts)

    def recognize_and_predict(self):
        results, screenshot = self.recognize()
        self.update_monster(results)
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        # 历史对局
        if self.history_match_ui.isVisible():
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)
        return prediction, results, screenshot

    def toggle_history_panel(self):
        """切换历史对局面板的显示"""
        target_width = self.width()
        if self.history_match is None:
            QMessageBox.warning(self, "警告", "历史数据加载失败，无法显示历史对局")
            return

        is_visible = self.history_match_ui.isVisible()
        self.history_match_ui.setVisible(not is_visible)
        if not is_visible:
            self.history_button.setText("隐藏历史对局")
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)
            target_width += 540
        else:
            self.history_button.setText("显示历史对局")
            target_width -= 540
        self.animate_size_change(target_width)

    def reselect_roi(self):
        self.recognizer.select_roi()
        self.no_region = False

    def toggle_auto_fetch(self):
        if not (hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running):
            self.auto_fetch = auto_fetch.AutoFetch(
                self.adb_connector,
                self.game_mode,
                self.is_invest,
                update_prediction_callback=self.update_prediction_callback,
                update_monster_callback=self.update_monster_callback,
                updater=self.update_statistics_callback,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
                training_duration=float(self.duration_entry.text()) * 3600,  # 获取训练时长
            )
            self.auto_fetch.start_auto_fetch()
        else:
            self.auto_fetch.stop_auto_fetch()

    def update_statistics(self):
        elapsed_time = time.time() - self.auto_fetch.start_time if self.auto_fetch.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.auto_fetch.total_fill_count},    "
            f"填写×次数: {self.auto_fetch.incorrect_fill_count},    "
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟"
        )
        self.stats_label.setText(stats_text)

    def update_device_serial(self):
        new_serial = self.serial_entry.text()
        self.adb_connector.update_device_serial(new_serial)
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {new_serial}")

    def start_callback(self):
        self.update_button_signal.emit("停止自动获取数据")

    def stop_callback(self):
        self.update_button_signal.emit("自动获取数据")

    def update_monster_callback(self, results: list):
        self.update_monster_signal.emit(results)

    def update_prediction_callback(self, prediction: float):
        self.update_prediction_signal.emit(prediction)

    def update_statistics_callback(self):
        self.update_statistics_signal.emit()

    def run_simulation(self):
        """
        获取左右怪物信息，转换为JSON格式，并通过stdin传递给main_sim.py子进程。
        """
        left_monsters_data = {}
        right_monsters_data = {}

        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()

        # 获取左侧怪物信息
        for monster_id, entry in left_monsters_dict.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                # Need to map monster_id (string) to monster name
                # Assuming MONSTER_MAPPING is accessible or can be imported
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        left_monsters_data[monster_name] = int(count)
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        # 获取右侧怪物信息
        for monster_id, entry in right_monsters_dict.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        right_monsters_data[monster_name] = int(count)
                    else:
                        logger.error(f"Monster name not found for ID {monster_id}")
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        simulation_data = {"left": left_monsters_data, "right": right_monsters_data}

        json_data = json.dumps(simulation_data, ensure_ascii=False)
        logger.info(f"Simulation data JSON: {json_data}")

        try:
            # 启动main_sim.py子进程 (非阻塞)
            # Use sys.executable to ensure the same Python interpreter is used
            process = subprocess.Popen(
                [sys.executable, "main_sim.py"], stdin=subprocess.PIPE, text=True, encoding="utf-8"
            )
            # 通过stdin传递JSON数据并关闭stdin
            process.stdin.write(json_data)
            process.stdin.close()
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到 main_sim.py 文件，请检查路径。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动模拟器时发生错误: {str(e)}")

    def get_monster_name_by_id(self, monster_id: int):
        """根据怪物ID获取怪物名称"""
        # Need to import MONSTER_MAPPING from simulator.utils
        try:
            from simulator.utils import MONSTER_MAPPING

            # Adjust for 1-based UI IDs vs 0-based mapping keys
            monster_name = MONSTER_MAPPING.get(monster_id - 1)
            if not monster_name:
                logger.error(f"Monster ID {monster_id} not found in MONSTER_MAPPING.")
            return monster_name
        except ImportError:
            logger.error("Error importing MONSTER_MAPPING from simulator.utils")
            return None

    def update_game_mode(self, mode):
        self.game_mode = mode

    def update_invest_status(self, state):
        self.is_invest = state == Qt.CheckState.Checked.value

    def on_terrain_selected(self, clicked_button, terrain_key):
        """处理地形选择事件"""
        # 取消其他按钮的选中状态
        for key, btn in self.terrain_buttons.items():
            if btn != clicked_button:
                btn.setChecked(False)

        # 确保当前按钮被选中
        clicked_button.setChecked(True)
        logger.info(f"选择地形: {terrain_key}")
        self.predict()

    def get_current_terrain(self):
        """获取当前选择的地形"""
        for key, btn in self.terrain_buttons.items():
            if btn.isChecked():
                return key
        return "none"  # 默认无地形

    def build_terrain_features(self, left_counts, right_counts, terrain):
        """构建包含地形的完整特征向量"""
        # 获取场地特征列数（从FieldRecognizer获取）
        try:
            from field_recognition import FieldRecognizer

            field_recognizer = FieldRecognizer()
            field_feature_columns = field_recognizer.get_feature_columns()
            num_field_features = len(field_feature_columns)

            # 构建地形特征向量
            terrain_features = np.zeros(num_field_features)

            if terrain != "none":
                # 直接使用特征列名称
                if terrain in field_feature_columns:
                    terrain_idx = field_feature_columns.index(terrain)
                    terrain_features[terrain_idx] = 1
                else:
                    logger.warning(f"地形 {terrain} 不在特征列中: {field_feature_columns}")
        except Exception as e:
            logger.warning(f"无法获取场地识别特征列，使用默认值: {e}")
            # 如果无法获取，使用全局默认值
            num_field_features = FIELD_FEATURE_COUNT
            terrain_features = np.zeros(num_field_features)

        # 按照data_cleaning_with_field_recognize_gpu.py的格式组织数据
        # 1L-77L (左侧怪物特征)
        # 78L-83L (场地特征L)
        # 1R-77R (右侧怪物特征)
        # 78R-83R (场地特征R，复制)

        full_features = np.concatenate(
            [left_counts, terrain_features, right_counts, terrain_features]  # 1L-77L  # 78L-83L  # 1R-77R  # 78R-83R
        )

        return full_features

    def update_result(self, text):
        self.result_label.setText(text)

    def update_stats(self, total, incorrect, duration):
        stats_text = f"总共: {total}, 错误: {incorrect}, 时长: {duration}"
        self.stats_label.setText(stats_text)

    def update_image_display(self, qimage):
        self.image_display.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def package_data_and_show(self):
        try:
            zip_filename = data_package.package_data()
            if zip_filename:
                # 在文件浏览器中高亮显示文件
                subprocess.run(f'explorer /select,"{zip_filename}"')
                QMessageBox.information(self, "成功", f"数据已打包到 {zip_filename}")
            else:
                QMessageBox.warning(self, "警告", "没有找到可以打包的数据目录。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打包数据时发生错误: {str(e)}")


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec()
