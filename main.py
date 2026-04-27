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
from PyQt6.QtWidgets import QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QButtonGroup
from PyQt6.QtWidgets import QGroupBox, QMessageBox, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPainter, QColor
import PyQt6.QtCore as QtCore

import loadData
import auto_fetch
import maa_adb_connector
from maa_adb_connector import AdbConnectorAdapter, ConnectionTypeRegistry, InputMethodRegistry, MaaFrameworkDetector
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
        # 捕获模式：ADB, PC, WIN
        self.current_capture_mode = "ADB"

        # 尝试连接模拟器
        self.adb_connector = AdbConnectorAdapter()
        self.pc_connector = loadData.PcConnector()
        self.adb_connector_thread = ADBConnectorThread(self)
        self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
        self.adb_connector_thread.start()

        self.auto_fetch_running = False
        self.is_invest = False
        self.game_mode = "单人"

        # 模型
        self.cannot_model = CannotModel()

        # 怪物识别模块
        self.recognizer = recognize.RecognizeMonster(method="ADB")

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

        # 如果模型未加载，显示提示并禁用预测相关按钮
        if not self.cannot_model.is_model_loaded:
            self.recognize_button.setEnabled(False)
            self.recognize_button.setToolTip("模型未加载，无法使用此功能")
            self.input_panel.predict_button.setEnabled(False)
            self.input_panel.predict_button.setToolTip("模型未加载，无法使用此功能")

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
        self.input_panel.terrain_changed.connect(self.predict)

        # 中央面板 - 结果和控制区
        center_panel = QWidget()
        center_panel.setFixedWidth(550)  # 固定右侧面板宽度
        center_layout = QVBoxLayout(center_panel)

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

        center_layout.addWidget(input_display)

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

        self.recognize_only_button = QPushButton("仅识别")
        self.recognize_only_button.clicked.connect(self.recognize_only)
        self.recognize_only_button.setStyleSheet(self.qt_button_style)
        self.recognize_only_button.setFixedWidth(80)  # 小按钮
        result_identify_layout.addWidget(self.recognize_only_button)

        result_layout.addWidget(result_identify_group)

        center_layout.addWidget(result_group)

        # 底部区域 - 控制面板和连接设置
        self.bottom_group = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_group)

        # 左侧垂直布局：控制面板 + 连接设置
        left_column = QVBoxLayout()

        # --- 控制面板 ---
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        # 第一行按钮
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.duration_label = QLabel("训练时长(小时):")
        self.duration_entry = QLineEdit("325")
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

        # 第二行按钮 - 数据操作和统计
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)

        self.package_data_button = QPushButton("数据打包")
        self.package_data_button.clicked.connect(self.package_data_and_show)
        row2_layout.addWidget(self.package_data_button)

        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))
        row2_layout.addWidget(self.stats_label)

        # GitHub链接
        github_label = QLabel(
            '<a href="https://github.com/Ancientea/CannotMax" style="color: #2196F3; text-decoration: none;">https://github.com/Ancientea/CannotMax</a>'
        )
        github_label.setMargin(0)
        github_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        github_label.setOpenExternalLinks(True)
        github_label.setFont(QFont("Microsoft YaHei", 9))
        github_label.setContentsMargins(0, 0, 0, 0)

        # 添加到控制布局
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(github_label)

        # --- 连接设置 ---
        connection_group = QGroupBox("连接设置")
        connection_layout = QVBoxLayout(connection_group)

        # 模式选择行
        mode_row = QWidget()
        mode_row_layout = QHBoxLayout(mode_row)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.adb_mode_btn = QPushButton("安卓端-ADB")
        self.pc_mode_btn = QPushButton("PC端(UI比例100%)")
        self.win_mode_btn = QPushButton("窗口截取")

        mode_btns = [self.adb_mode_btn, self.pc_mode_btn, self.win_mode_btn]
        mode_style = """
            QPushButton {
                background-color: #313131;
                color: #F3F31F;
                border-radius: 10px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #414141;
            }
            QPushButton:checked {
                background-color: #F5EA2D;
                color: #313131;
            }
        """
        for btn in mode_btns:
            btn.setCheckable(True)
            btn.setStyleSheet(mode_style)
            self.mode_group.addButton(btn)
            mode_row_layout.addWidget(btn)

        self.adb_mode_btn.setChecked(True)
        self.adb_mode_btn.clicked.connect(lambda: self.on_mode_changed("ADB"))
        self.pc_mode_btn.clicked.connect(lambda: self.on_mode_changed("PC"))
        self.win_mode_btn.clicked.connect(lambda: self.on_mode_changed("WIN"))
        connection_layout.addWidget(mode_row)

        # MAA连接方式行
        maa_row = QWidget()
        maa_row_layout = QHBoxLayout(maa_row)
        maa_row_layout.setContentsMargins(0, 0, 0, 0)

        self.connection_type_label = QLabel("连接方式:")
        self.connection_type_combo = QComboBox()
        for ct in ConnectionTypeRegistry.get_all_types():
            self.connection_type_combo.addItem(ct.display_name, ct.type_id)
        self.connection_type_combo.currentIndexChanged.connect(self.on_connection_type_changed)

        self.input_method_label = QLabel("输入方式:")
        self.input_method_combo = QComboBox()
        default_method = InputMethodRegistry.get_default_method()
        for m in InputMethodRegistry.get_all_methods():
            self.input_method_combo.addItem(m.display_name, m.method_id)
        idx = self.input_method_combo.findData(default_method.method_id)
        if idx >= 0:
            self.input_method_combo.setCurrentIndex(idx)
        self.input_method_combo.currentIndexChanged.connect(self.on_input_method_changed)

        maa_row_layout.addWidget(self.connection_type_label)
        maa_row_layout.addWidget(self.connection_type_combo)
        maa_row_layout.addWidget(self.input_method_label)
        maa_row_layout.addWidget(self.input_method_combo)
        connection_layout.addWidget(maa_row)

        # 序列号行
        conn_row1 = QWidget()
        conn_row1_layout = QHBoxLayout(conn_row1)
        conn_row1_layout.setContentsMargins(0, 0, 0, 0)

        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QComboBox()
        self.serial_entry.setEditable(True)
        self.serial_entry.setFixedWidth(150)
        self.serial_entry.lineEdit().setPlaceholderText("127.0.0.1:5555")

        self.serial_button = QPushButton("更新")
        self.serial_button.clicked.connect(self.update_device_serial)

        self.maa_status_label = QLabel("")
        self.maa_status_label.setStyleSheet("color: #888; font-size: 10px;")

        conn_row1_layout.addWidget(self.serial_label)
        conn_row1_layout.addWidget(self.serial_entry)
        conn_row1_layout.addWidget(self.serial_button)
        conn_row1_layout.addWidget(self.maa_status_label)

        # 捕获设置行
        conn_row2 = QWidget()
        conn_row2_layout = QHBoxLayout(conn_row2)
        conn_row2_layout.setContentsMargins(0, 0, 0, 0)

        self.choose_window_button = QPushButton("选择截屏窗口")
        self.choose_window_button.clicked.connect(self.choose_capture_window)
        self.reselect_button = QPushButton("选择范围")
        self.reselect_button.clicked.connect(self.reselect_roi)

        # 初始为 ADB 模式，禁用窗口捕获相关按钮
        self.choose_window_button.setEnabled(False)
        self.reselect_button.setEnabled(False)

        conn_row2_layout.addWidget(self.choose_window_button)
        conn_row2_layout.addWidget(self.reselect_button)

        connection_layout.addWidget(conn_row1)
        connection_layout.addWidget(conn_row2)

        # 将两个组框添加到左侧列
        left_column.addWidget(control_group)
        left_column.addWidget(connection_group)

        # 第五行按钮 (纵向排列的功能按钮)
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

        # 窗口置顶按钮
        self.always_on_top_button = QPushButton("窗口置顶")
        self.always_on_top_button.clicked.connect(self.toggle_always_on_top)
        self.always_on_top_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.always_on_top_button)

        # 排布底部布局
        self.bottom_layout.addLayout(left_column)
        self.bottom_layout.addWidget(row5)

        center_layout.addWidget(self.bottom_group)

        # 创建并添加HistoryMatchUI实例
        self.history_match_ui = HistoryMatchUI(self.history_match)
        self.history_match_ui.setVisible(False)  # 初始隐藏

        main_layout.addWidget(center_panel, 1)
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
        self.refresh_device_list()

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

    @property
    def active_connector(self):
        if self.current_capture_mode == "PC":
            return self.pc_connector
        return self.adb_connector

    def on_mode_changed(self, mode):
        """切换捕获模式"""
        self.current_capture_mode = mode
        logger.info(f"切换捕获模式为: {mode}")

        is_win_mode = (mode == "WIN")
        is_adb_mode = (mode == "ADB")
        is_pc_mode = (mode == "PC")

        # 切换窗口捕获相关控件
        self.choose_window_button.setEnabled(is_win_mode)
        self.reselect_button.setEnabled(is_win_mode)
        
        # 切换 ADB 相关控件
        self.serial_label.setEnabled(is_adb_mode)
        self.serial_entry.setEnabled(is_adb_mode)
        self.serial_button.setEnabled(is_adb_mode)
        self.connection_type_label.setEnabled(is_adb_mode)
        self.connection_type_combo.setEnabled(is_adb_mode)
        self.input_method_label.setEnabled(is_adb_mode)
        self.input_method_combo.setEnabled(is_adb_mode)

        if mode == "ADB":
            self.refresh_device_list()
            self.recognizer = recognize.RecognizeMonster(method="ADB")
            if not self.adb_connector.device_serial:
                self.adb_connector_thread.start()
        elif mode == "WIN":
            if self.recognizer.method != "WIN":
                self.recognizer = recognize.RecognizeMonster(method="WIN")
            if self.recognizer._winrt is None:
                self.choose_capture_window()
        elif mode == "PC":
            self.recognizer = recognize.RecognizeMonster(method="ADB") # reuse ADB reading methodology but on PC Connector
            if not self.pc_connector.is_connected:
                self.pc_connector.connect()
                if not self.pc_connector.is_connected:
                    QMessageBox.warning(self, "警告", "未能连接到PC端窗口(明日方舟)。")

    def on_adb_connected(self):
        logger.info("模拟器初始化完成")
        if self.adb_connector.is_maa_available:
            self.maa_status_label.setText("MAA Framework已连接")
            self.maa_status_label.setStyleSheet("color: #00aa00; font-size: 10px;")
        else:
            self.maa_status_label.setText("使用自有ADB实现（MAA Framework不可用）")
            self.maa_status_label.setStyleSheet("color: #aa8800; font-size: 10px;")

    def on_connection_type_changed(self, index):
        type_id = self.connection_type_combo.currentData()
        if not type_id:
            return
        default_address = ConnectionTypeRegistry.get_default_address(type_id)
        if default_address:
            self.serial_entry.setCurrentText(default_address)
            self.adb_connector.set_connection_type(type_id)
            self.adb_connector.set_device_serial(default_address)
        if self.adb_connector.is_connected:
            self.adb_connector.disconnect()
            self.maa_status_label.setText("已断开，请重新连接")
            self.maa_status_label.setStyleSheet("color: #aa0000; font-size: 10px;")

    def on_input_method_changed(self, index):
        method_id = self.input_method_combo.currentData()
        if not method_id:
            return
        self.adb_connector.set_input_method(method_id)
        if self.adb_connector.is_connected:
            QMessageBox.information(self, "提示", "输入方式已更改，请重新连接以生效")

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
                    self.recognizer = recognize.RecognizeMonster(method="WIN", window_name=sel["window_name"], monitor_index=None)
                    hint = f"已切换至窗口：{sel['window_name']}"
                else:
                    idx = max(1, sel["monitor_index"])
                    self.recognizer = recognize.RecognizeMonster(method="WIN", window_name=None, monitor_index=idx)
                    hint = f"已切换至整屏：显示器 {sel['monitor_index']}"

                self.no_region = True
                QMessageBox.information(self, "成功", hint + "\n建议重新选择范围。")
        except Exception as e:
            QMessageBox.critical(self, "异常", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._switching_source = False
            self.choose_window_button.setEnabled(self.current_capture_mode == "WIN")

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

            # 构建包含地形的完整特征向量
            full_features = self.input_panel.build_terrain_features(left_counts, right_counts)

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

    def get_recognize(self):
        """
        根据当前模式获取截图并识别
        """
        screenshot = None
        if self.current_capture_mode in ["ADB", "PC"]:
            screenshot = self.active_connector.capture_screenshot()
            if screenshot is None:
                # 尝试重新连接一次
                self.active_connector.connect()
                screenshot = self.active_connector.capture_screenshot()
            if screenshot is None:
                logger.error(f"{self.current_capture_mode} 截图失败")
            
            results = self.recognizer.process_regions(screenshot)
        else:
            # WIN 模式，recognizer 内部处理 WinRT 或 PIL
            results = self.recognizer.process_regions(None)

        return results

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

    def recognize_only(self):
        recognize_results = self.get_recognize()
        self.update_monster(recognize_results)

    def recognize_and_predict(self):
        recognize_results = self.get_recognize()
        self.update_monster(recognize_results)
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        # 历史对局
        if self.history_match_ui.isVisible():
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)

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

    def toggle_auto_fetch(self):
        if not (hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running):
            self.auto_fetch = auto_fetch.AutoFetch(
                self.active_connector,
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

    def refresh_device_list(self):
        """刷新并更新模拟器序列号下拉列表"""
        current_text = self.serial_entry.currentText()
        devices = self.adb_connector.get_device_list()
        self.serial_entry.clear()
        if devices:
            self.serial_entry.addItems(devices)
            if current_text in devices:
                self.serial_entry.setCurrentText(current_text)
            else:
                self.serial_entry.setCurrentIndex(0)
        else:
            self.serial_entry.addItem("127.0.0.1:5555")
            self.serial_entry.setCurrentText(current_text if current_text else "127.0.0.1:5555")

    def update_device_serial(self):
        new_serial = self.serial_entry.currentText()
        device_serial = self.adb_connector.update_device_serial(new_serial)
        self.adb_connector.connect()  # 尝试连接新设备
        self.serial_entry.setCurrentText(device_serial)
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {device_serial}")

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


    def toggle_always_on_top(self):
        if self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
            self.always_on_top_button.setText("窗口置顶")
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            self.always_on_top_button.setText("取消置顶")
        self.show() # Reapply window flags

    def closeEvent(self, event):
        """窗口关闭时的处理"""
        if hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running:
            self.auto_fetch.stop_auto_fetch()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec()
