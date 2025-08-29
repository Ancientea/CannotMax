# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from typing import List

import numpy as np
import recognize
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QScrollArea,
    QMessageBox,
    QGridLayout,
    QSizePolicy,
    QGraphicsDropShadowEffect,
    QFrame,
    QDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QColor
from sklearn.metrics.pairwise import cosine_similarity
import PyQt6.QtCore as QtCore

import loadData
import auto_fetch
import similar_history_match
from recognize import MONSTER_COUNT, intelligent_workers_debug
from specialmonster import SpecialMonsterHandler

import ctypes
from ctypes import wintypes


def list_visible_window_titles() -> list[str]:
    """列出所有**可见**窗口的标题（去重并按字典序排序）。"""
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW

    titles: List[str] = []

    def foreach(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buff, length + 1)
                t = buff.value.strip()
                if t:
                    titles.append(t)
        return True

    EnumWindows(EnumWindowsProc(foreach), 0)
    return sorted(set(titles))


# 动态加载推理模型（Torch/ONNX 二选一）
try:
    from predict import CannotModel
    from train import UnitAwareTransformer
except Exception:
    from predict_onnx import CannotModel

# ------------------------------- 日志配置 -------------------------------
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ADBConnectorThread(QThread):
    """在后台线程内调用 `loadData.AdbConnector.connect()`，避免阻塞 UI。"""

    connect_finished = pyqtSignal()

    def __init__(self, app: "ArknightsApp"):
        super().__init__()
        self.app = app

    def run(self):
        self.app.adb_connector.connect()
        self.connect_finished.emit()


class HistoryLoader(QThread):
    """后台加载 `HistoryMatch`，并在完成后通过信号返回实例。"""

    history_loaded = pyqtSignal(object)

    def run(self):
        history_match = similar_history_match.HistoryMatch()
        # 兜底：确保 feat_past 与 N_history 初始化
        try:
            history_match.feat_past = np.hstack([history_match.past_left, history_match.past_right])
        except Exception:
            history_match.feat_past = None
        history_match.N_history = 0 if history_match.labels is None else len(history_match.labels)
        self.history_loaded.emit(history_match)


class AsyncHistoryMatch(QObject):
    """异步包装器：在加载完成前拦截访问，加载完成后代理到真实对象。"""

    history_loaded = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._match = None
        self._loader = HistoryLoader()
        self._loader.history_loaded.connect(self._on_loaded)
        self._loader.start()

    def _on_loaded(self, history_match):
        self._match = history_match
        self.history_loaded.emit(history_match)

    def __getattr__(self, name):
        if self._match is None:
            raise AttributeError(f"HistoryMatch not loaded yet: '{name}'")
        return getattr(self._match, name)


class ArknightsApp(QMainWindow):
    """应用主窗口：负责 UI 布局、数据流串联、识别与预测的一站式入口。"""

    # 自定义信号（用于跨线程/模块更新 UI）
    update_button_signal = pyqtSignal(str)  # 更新按钮文本
    update_monster_signal = pyqtSignal(list)
    update_prediction_signal = pyqtSignal(float)
    update_statistics_signal = pyqtSignal()  # 更新统计

    def __init__(self):
        super().__init__()

        # 连接模拟器（ADB）
        self.adb_connector = loadData.AdbConnector()
        self.adb_connector_thread = ADBConnectorThread(self)
        self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
        self.adb_connector_thread.start()

        # 运行时状态
        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = False
        self.game_mode = "单人"
        self.device_serial = self.adb_connector.manual_serial

        # UI 输入缓存
        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}

        # 预测模型与识别器
        self.cannot_model = CannotModel()
        self.recognizer = recognize.RecognizeMonster()

        # 历史对局与特殊怪物提示
        self.history_visible = False
        self.history_data_loaded = False
        self.history_widget = None
        self.history_scroll_area = None
        self.history_match = AsyncHistoryMatch()
        self.history_match.history_loaded.connect(self.on_history_loaded)
        self.special_monster_handler = SpecialMonsterHandler()

        # 初始化 UI 与资源
        self.init_ui()
        self.load_images()

    # ------------------------------ UI 构建 ------------------------------
    def init_ui(self):
        """搭建主界面布局与控件、样式初始化与信号绑定。"""
        self.setWindowTitle("铁鳍鱼_Arknights Neural Network")
        self.setWindowIcon(QIcon("ico/icon.ico"))
        self.setGeometry(100, 100, 1700, 800)
        self.background = QPixmap("ico/background.png")

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧面板：怪物网格输入
        left_panel = QWidget()
        left_panel.setObjectName("left_panel_id")
        left_panel.setStyleSheet(
            """
            QWidget#left_panel_id {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            """
        )
        left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        monster_group = QWidget()
        monster_layout = QVBoxLayout(monster_group)

        # 滚动区
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0);
                width: 12px; margin: 0px;
            }
            QScrollBar::handle:horizontal { background: rgba(100,100,100,150); min-height: 20px; border-radius: 8px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: none; }

            QScrollBar:vertical { background: rgba(0, 0, 0, 0); width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background: rgba(100,100,100,150); min-height: 20px; border-radius: 8px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }

            QScrollArea { background-color: rgba(0, 0, 0, 0); border:0px }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: rgba(50, 50, 50, 100); width: 12px; margin: 15px 0 15px 0; }
            QScrollBar::handle:vertical { background: rgba(100,100,100,150); min-height: 20px; border-radius: 6px; }
            """
        )

        scroll_content = QWidget()
        self.scroll_grid = QGridLayout(scroll_content)
        self.scroll_grid.setSpacing(5)
        self.scroll_grid.setContentsMargins(5, 5, 5, 5)

        # 7 列网格，每单元 120px 高
        self.COLUMNS = 7
        self.ROW_HEIGHT = 120

        scroll.setWidget(scroll_content)
        monster_layout.addWidget(scroll)
        left_layout.addWidget(monster_group)

        # 右侧面板：结果与控制
        right_panel = QWidget()
        right_panel.setFixedWidth(550)
        right_layout = QVBoxLayout(right_panel)

        # 顶部：当前输入展示
        input_display = QGroupBox()
        input_display.setStyleSheet(
            """
            QGroupBox {
                background-color: rgba(0, 0, 0, 120);
                border-radius: 15px; border: 5px solid #F5EA2D;
                margin-top: 10px; padding: 10px 0;
            }
            QGroupBox::title { color: white; left: 15px; padding: 0 5px; }
            """
        )
        input_layout = QHBoxLayout(input_display)

        # 左右输入展示容器
        left_input_group = QWidget(); left_input_layout = QHBoxLayout(left_input_group)
        self.left_input_content = QWidget(); self.left_input_layout = QHBoxLayout(self.left_input_content)
        self.left_input_layout.setSpacing(5)
        left_input_layout.addWidget(self.left_input_content)

        right_input_group = QWidget(); right_input_layout = QHBoxLayout(right_input_group)
        self.right_input_content = QWidget(); self.right_input_layout = QHBoxLayout(self.right_input_content)
        self.right_input_layout.setSpacing(5)
        right_input_layout.addWidget(self.right_input_content)

        input_layout.addWidget(left_input_group)
        input_layout.addWidget(right_input_group)
        right_layout.addWidget(input_display)

        # 中部：预测结果
        result_group = QGroupBox()
        result_group.setStyleSheet(
            """
            QGroupBox { background-color: rgba(120,120,120,10); border-radius: 15px; border: 1px solid #747474; }
            """
        )
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setFont(QFont("Microsoft YaHei", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_label)

        result_button = QWidget(); result_button_layout = QHBoxLayout(result_button)
        self.predict_button = QPushButton("开始预测"); self.predict_button.clicked.connect(self.predict)
        self.predict_button.setStyleSheet(
            """
            QPushButton { background-color: #313131; color: #F3F31F; border-radius: 16px; padding: 8px; font-weight: bold; min-height: 30px; }
            QPushButton:hover { background-color: #414141; }
            QPushButton:pressed { background-color: #212121; }
            """
        )
        self.predict_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.reset_button = QPushButton("重置"); self.reset_button.clicked.connect(self.reset_entries)
        self.reset_button.setStyleSheet(
            """
            QPushButton { background-color: #313131; color: #F3F31F; border-radius: 16px; padding: 8px; font-weight: bold; min-height: 30px; }
            QPushButton:hover { background-color: #414141; }
            QPushButton:pressed { background-color: #212121; }
            """
        )
        result_button_layout.addWidget(self.predict_button)
        result_button_layout.addWidget(self.reset_button)
        result_layout.addWidget(result_button)
        right_layout.addWidget(result_group)

        # 底部：控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        # 第一行：时长 + 自动抓取 + 模式 + 投资
        row1 = QWidget(); row1_layout = QHBoxLayout(row1)
        self.duration_label = QLabel("训练时长(小时):")
        self.duration_entry = QLineEdit("-1"); self.duration_entry.setFixedWidth(50)
        self.auto_fetch_button = QPushButton("自动获取数据"); self.auto_fetch_button.clicked.connect(self.toggle_auto_fetch)
        self.mode_menu = QComboBox(); self.mode_menu.addItems(["单人", "30人"]) 
        self.invest_checkbox = QCheckBox("投资")
        row1_layout.addWidget(self.duration_label)
        row1_layout.addWidget(self.duration_entry)
        row1_layout.addWidget(self.auto_fetch_button)
        row1_layout.addWidget(self.mode_menu)
        row1_layout.addWidget(self.invest_checkbox)

        # 第二行：识别并预测
        row2 = QWidget(); row2_layout = QHBoxLayout(row2)
        self.recognize_button = QPushButton("识别并预测"); self.recognize_button.clicked.connect(self.recognize_and_predict)
        self.recognize_button.setStyleSheet(
            """
            QPushButton { background-color: #313131; color: #F3F31F; border-radius: 16px; padding: 8px; font-weight: bold; min-height: 30px; }
            QPushButton:hover { background-color: #414141; }
            QPushButton:pressed { background-color: #212121; }
            """
        )
        row2_layout.addWidget(self.recognize_button)

        # 第三行：区域与设备
        row3 = QWidget(); row3_layout = QHBoxLayout(row3)
        self.reselect_button = QPushButton("选择范围"); self.reselect_button.clicked.connect(self.reselect_roi)
        self.choose_window_button = QPushButton("选择截屏窗口"); self.choose_window_button.clicked.connect(self.choose_capture_window)
        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QLineEdit(self.device_serial); self.serial_entry.setFixedWidth(100)
        self.serial_button = QPushButton("更新"); self.serial_button.clicked.connect(self.update_device_serial)
        row3_layout.addWidget(self.choose_window_button)
        row3_layout.addWidget(self.reselect_button)
        row3_layout.addWidget(self.serial_label)
        row3_layout.addWidget(self.serial_entry)
        row3_layout.addWidget(self.serial_button)

        # 第四行：沙盒模拟
        row4 = QWidget(); row4_layout = QHBoxLayout(row4)
        self.simulate_button = QPushButton("沙盒模拟"); self.simulate_button.clicked.connect(self.run_simulation)
        self.simulate_button.setStyleSheet(
            """
            QPushButton { background-color: #313131; color: #F3F31F; border-radius: 16px; padding: 8px; font-weight: bold; min-height: 30px; }
            QPushButton:hover { background-color: #414141; }
            QPushButton:pressed { background-color: #212121; }
            """
        )
        row4_layout.addWidget(self.simulate_button)

        # 统计信息
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))

        # 汇总到控制布局
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(row3)
        control_layout.addWidget(row4)
        control_layout.addWidget(self.stats_label)

        right_layout.addWidget(control_group)

        # 历史对局按钮
        self.history_button = QPushButton("显示历史对局"); self.history_button.clicked.connect(self.toggle_history_panel)
        self.history_button.setStyleSheet(
            """
            QPushButton { background-color: #313131; color: #F3F31F; border-radius: 16px; padding: 8px; font-weight: bold; min-height: 30px; }
            QPushButton:hover { background-color: #414141; }
            QPushButton:pressed { background-color: #212121; }
            """
        )
        right_layout.addWidget(self.history_button)

        # 信号连接
        self.mode_menu.currentTextChanged.connect(self.update_game_mode)
        self.invest_checkbox.stateChanged.connect(self.update_invest_status)

        # 自动抓取信号桥
        self.update_button_signal.connect(self.auto_fetch_button.setText)
        self.update_monster_signal.connect(self.update_monster)
        self.update_prediction_signal.connect(self.update_prediction)
        self.update_statistics_signal.connect(self.update_statistics)

        # 主布局拼装
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)
        self.setCentralWidget(main_widget)

    # --------------------------------- 回调 ---------------------------------
    def on_adb_connected(self):
        print("模拟器初始化完成")

    def on_history_loaded(self, history_match):
        print("尝试获取错题本")
        # 同步属性
        self.past_left = history_match.past_left
        self.past_right = history_match.past_right
        self.labels = history_match.labels
        self.feat_past = history_match.feat_past
        self.N_history = history_match.N_history
        self.history_data_loaded = True
        print("错题本加载成功")

    # ------------------------- 截屏源选择对话框 -------------------------
    class WindowPickerDialog(QDialog):
        """列出可见窗口标题，并内置“整屏(1/2/3)”选项。双击确定。"""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("选择截屏窗口")
            self.resize(520, 480)
            self.selected_title = None

            self.search = QLineEdit(self)
            self.search.setPlaceholderText("输入关键字过滤（支持大小写不敏感）")

            self.listw = QListWidget(self)
            # 预置“整屏（主屏0/副屏1/2）”选项在最上面
            self.listw.addItem("【整屏】主屏(1)")
            self.listw.addItem("【整屏】副屏(2)")
            self.listw.addItem("【整屏】副屏(3)")
            self.listw.addItem("—————— 窗口列表 ——————")

            self._all_titles = list_visible_window_titles()
            for t in self._all_titles:
                self.listw.addItem(t)

            self.search.textChanged.connect(self._filter)
            self.listw.itemDoubleClicked.connect(self._accept)

            layout = QVBoxLayout(self)
            layout.addWidget(self.search)
            layout.addWidget(self.listw)

        def _filter(self, text: str):
            text_low = (text or "").lower()
            self.listw.clear()
            self.listw.addItem("【整屏】主屏(1)")
            self.listw.addItem("【整屏】副屏(2)")
            self.listw.addItem("【整屏】副屏(3)")
            self.listw.addItem("—————— 窗口列表 ——————")
            for t in self._all_titles:
                if text_low in t.lower():
                    self.listw.addItem(t)

        def _accept(self):
            self.accept()

        def get_selection(self):
            item = self.listw.currentItem()
            if not item:
                return None
            text = item.text()
            if text.startswith("【整屏】"):
                # 返回 monitor_index
                if "(1)" in text:
                    return {"monitor_index": 1}
                if "(2)" in text:
                    return {"monitor_index": 2}
                if "(3)" in text:
                    return {"monitor_index": 3}
            elif text.startswith("——————"):
                return None
            else:
                # 返回 window_name
                return {"window_name": text}

    # ------------------------------ 功能操作 ------------------------------
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
            dlg = self.WindowPickerDialog(self)
            if dlg.exec():
                sel = dlg.get_selection()
                if not sel:
                    QMessageBox.information(self, "提示", "未选择任何项")
                    return
                ok = False
                if "window_name" in sel:
                    ok = self.recognizer.update_capture_target(window_name=sel["window_name"], monitor_index=None)
                    hint = f"已切换至窗口：{sel['window_name']}"
                else:
                    idx = max(1, sel["monitor_index"])
                    ok = self.recognizer.update_capture_target(window_name=None, monitor_index=idx)
                    hint = f"已切换至整屏：显示器 {sel['monitor_index']}"
                if ok:
                    self.no_region = True
                    QMessageBox.information(self, "成功", hint + "\n建议重新选择范围。")
                else:
                    QMessageBox.critical(self, "失败", "切换截屏目标失败，请重试。")
        except Exception as e:
            QMessageBox.critical(self, "异常", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._switching_source = False
            self.choose_window_button.setEnabled(True)

    def paintEvent(self, event):
        """绘制背景图（按窗口等比放大/居中）。"""
        painter = QPainter(self)
        scaled_pixmap = self.background.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap((self.width() - scaled_pixmap.width()) // 2, (self.height() - scaled_pixmap.height()) // 2, scaled_pixmap)

    def load_images(self):
        """加载素材并构建左侧 7×N 网格输入区。"""
        for i in reversed(range(self.scroll_grid.count())):
            self.scroll_grid.itemAt(i).widget().setParent(None)

        row = 0
        col = 0
        for i in range(1, MONSTER_COUNT + 1):
            # 容器
            monster_container = QWidget()
            monster_container.setFixedHeight(self.ROW_HEIGHT)
            shadow01 = QGraphicsDropShadowEffect(); shadow01.setBlurRadius(5); shadow01.setColor(QColor(0, 0, 0, 120)); shadow01.setOffset(3)
            monster_container.setGraphicsEffect(shadow01)
            monster_container.setStyleSheet("QWidget {border-radius: 0px;}")
            container_layout = QVBoxLayout(monster_container); container_layout.setSpacing(2); container_layout.setContentsMargins(2, 2, 2, 2)

            # 图片
            img_label = QLabel(); img_label.setFixedSize(60, 60); img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            try:
                pixmap = QPixmap(f"images/{i}.png")
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(pixmap)
            except Exception as e:
                print(f"加载人物{i}图片错误: {str(e)}")

            # 左右输入
            left_entry = QLineEdit(); left_entry.setFixedWidth(60); left_entry.setPlaceholderText("左"); left_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.left_monsters[str(i)] = left_entry

            right_entry = QLineEdit(); right_entry.setFixedWidth(60); right_entry.setPlaceholderText("右"); right_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.right_monsters[str(i)] = right_entry

            container_layout.addWidget(img_label, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(left_entry, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(right_entry, 0, Qt.AlignmentFlag.AlignCenter)

            self.scroll_grid.addWidget(monster_container, row, col, Qt.AlignmentFlag.AlignCenter)

            col += 1
            if col >= self.COLUMNS:
                col = 0
                row += 1

    def update_input_display(self):
        """根据左/右输入框的数值，实时在右侧上方展示当前阵容。"""
        # 清空现有
        for i in reversed(range(self.left_input_layout.count())):
            widget = self.left_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        for i in reversed(range(self.right_input_layout.count())):
            widget = self.right_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        left_has_input, right_has_input = False, False
        for i in range(1, MONSTER_COUNT + 1):
            left_value = self.left_monsters[str(i)].text()
            right_value = self.right_monsters[str(i)].text()

            if left_value.isdigit() and int(left_value) > 0:
                left_has_input = True
                monster_widget = self.create_monster_display_widget(i, left_value)
                self.left_input_layout.addWidget(monster_widget)
            if right_value.isdigit() and int(right_value) > 0:
                right_has_input = True
                monster_widget = self.create_monster_display_widget(i, right_value)
                self.right_input_layout.addWidget(monster_widget)

        if not left_has_input:
            self.left_input_layout.addWidget(QLabel("无"))
        if not right_has_input:
            self.right_input_layout.addWidget(QLabel("无"))

    def create_monster_display_widget(self, monster_id, count):
        """创建右侧“当前输入展示”的单项组件（头像 + 数量）。"""
        widget = QWidget(); widget.setFixedWidth(67)
        shadow = QGraphicsDropShadowEffect(); shadow.setBlurRadius(0); shadow.setColor(QColor("#313131")); shadow.setOffset(2)
        widget.setGraphicsEffect(shadow)
        widget.setStyleSheet("""
            QWidget { border-radius: 0px; }
        """)
        layout = QVBoxLayout(widget); layout.setSpacing(2); layout.setContentsMargins(2, 2, 2, 2); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_label = QLabel(); img_label.setFixedSize(70, 70); img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pixmap = QPixmap(f"images/{monster_id}.png")
            if not pixmap.isNull():
                pixmap = pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                img_label.setPixmap(pixmap)
        except Exception:
            pass

        count_label = QLabel(count); count_label.setAlignment(Qt.AlignmentFlag.AlignCenter); count_label.setStyleSheet(
            """
            color: #EDEDED; font: bold 20px SimHei; border-radius: 5px; padding: 2px 5px; min-width: 20px;
            """
        )

        layout.addWidget(img_label)
        layout.addWidget(count_label)
        return widget

    def reset_entries(self):
        """清空左/右输入并重置结果展示。"""
        for entry in self.left_monsters.values():
            entry.clear(); entry.setStyleSheet("")
        for entry in self.right_monsters.values():
            entry.clear(); entry.setStyleSheet("")
        self.result_label.setText("预测结果将显示在这里")
        self.result_label.setStyleSheet("color: black;")
        self.update_input_display()

    # ------------------------------- 预测相关 -------------------------------
    def get_prediction(self):
        """读取 UI 输入并喂给模型，返回“右侧胜率”。"""
        try:
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            for name, entry in self.left_monsters.items():
                value = entry.text()
                left_counts[int(name) - 1] = int(value) if value.isdigit() else 0
            for name, entry in self.right_monsters.items():
                value = entry.text()
                right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            prediction = self.cannot_model.get_prediction(left_counts, right_counts)
            return prediction
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到模型文件，请先训练")
            return 0.5
        except RuntimeError as e:
            if "size mismatch" in str(e):
                QMessageBox.critical(self, "错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            return 0.5
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数字（0或正整数）")
            return 0.5
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时发生错误: {str(e)}")
            return 0.5

    def update_prediction(self, prediction):
        """将模型计算得到的概率映射到 UI 文案。"""
        right_win_prob = prediction
        left_win_prob = 1 - right_win_prob

        winner = "左方" if left_win_prob > 0.5 else "右方"
        if 0.6 > left_win_prob > 0.4:
            winner = "难说"

        if winner == "左方":
            self.result_label.setStyleSheet("color: #E23F25; font: bold,14px;")
        else:
            self.result_label.setStyleSheet("color: #25ace2; font: bold,14px;")

        if winner != "难说":
            result_text = f"预测胜方: {winner}\n左 {left_win_prob:.2%}  右 {right_win_prob:.2%}\n"
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages
        else:
            result_text = f"这一把{winner}\n左 {left_win_prob:.2%}  右 {right_win_prob:.2%}\n难道说？难道说？难道说？\n"
            self.result_label.setStyleSheet("color: black; font: bold,24px;")
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages

        self.result_label.setText(result_text)

    def predict(self):
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        self.update_input_display()
        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()

    # ------------------------------- 识别相关 -------------------------------
    def recognize(self):
        if self.auto_fetch_running:
            screenshot = self.adb_connector.capture_screenshot()
        else:
            screenshot = None
        if self.no_region:  # TODO: 判断需要移至recognize
            if self.first_recognize:
                self.adb_connector.connect()
                self.first_recognize = False
            screenshot = self.adb_connector.capture_screenshot()
        results = self.recognizer.process_regions(screenshot)
        return results, screenshot

    def update_monster(self, results):
        """根据识别结果回填左/右面板。"""
        self.reset_entries()
        for res in results:
            if "error" not in res:
                region_id = res["region_id"]
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.setText(str(number))
                    if entry.text():
                        entry.setStyleSheet("background-color: yellow;")
        self.update_input_display()

    def recognize_and_predict(self):
        results, screenshot = self.recognize()
        self.update_monster(results)
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()
        return prediction, results, screenshot

    # ------------------------------ 历史对局 ------------------------------
    def toggle_history_panel(self):
        """切换历史对局面板显示。"""
        if not self.history_visible:
            self.show_history_panel()
            self.history_button.setText("隐藏历史对局")
        else:
            self.hide_history_panel()
            self.history_button.setText("显示历史对局")
        self.history_visible = not self.history_visible

    def show_history_panel(self):
        """显示历史对局面板。"""
        if not self.history_data_loaded:
            QMessageBox.warning(self, "警告", "历史数据加载失败，无法显示历史对局")
            return
        self.history_scroll_area = QScrollArea(); self.history_scroll_area.setFixedWidth(540); self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setStyleSheet(
            """
            QScrollBar:horizontal { background: rgba(0,0,0,0); width: 12px; margin: 0px; }
            QScrollBar::handle:horizontal { background: rgba(100,100,100,150); min-height: 20px; border-radius: 8px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: none; }
            QScrollBar:vertical { background: rgba(0,0,0,0); width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background: rgba(100,100,100,150); min-height: 20px; border-radius: 8px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }
            QScrollArea { background-color: rgba(0, 0, 0, 40); border-radius: 15px; border: 5px solid #F5EA2D; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: rgba(50, 50, 50, 100); width: 12px; margin: 15px 0 15px 0; }
            QScrollBar::handle:vertical { background: rgba(100, 100, 100, 150); min-height: 20px; border-radius: 6px; }
            """
        )
        self.history_widget = QWidget(); self.history_layout = QVBoxLayout(self.history_widget); self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.render_similar_matches()
        self.history_scroll_area.setWidget(self.history_widget)
        self.centralWidget().layout().addWidget(self.history_scroll_area)

    def hide_history_panel(self):
        """隐藏历史对局面板。"""
        if self.history_scroll_area:
            self.history_scroll_area.setParent(None)
            self.history_scroll_area = None
            self.history_widget = None

    def render_similar_matches(self):
        """根据当前输入，从历史数据中渲染 Top-N 相似对局。"""
        try:
            cur_left = np.zeros(56, dtype=float)
            cur_right = np.zeros(56, dtype=float)
            for name, entry in self.left_monsters.items():
                v = entry.text();
                if v.isdigit():
                    cur_left[int(name) - 1] = float(v)
            for name, entry in self.right_monsters.items():
                v = entry.text();
                if v.isdigit():
                    cur_right[int(name) - 1] = float(v)

            self.history_match.render_similar_matches(cur_left, cur_right)
            sims = self.history_match.sims
            top_indices = self.history_match.top20_idx

            # 清空现有
            for i in reversed(range(self.history_layout.count())):
                self.history_layout.itemAt(i).widget().setParent(None)

            title_label = QLabel(f"错题本")
            shadow = QGraphicsDropShadowEffect(); shadow.setBlurRadius(0); shadow.setColor(QColor("#313131")); shadow.setOffset(2)
            title_label.setGraphicsEffect(shadow)
            title_label.setStyleSheet(
                """
                QWidget { border-radius: 0px; font-size: 24px; font-weight: bold; color: white; }
                """
            )
            self.history_layout.addWidget(title_label)

            for idx in top_indices:
                self.add_history_match(idx, sims[idx])
        except Exception as e:
            print(f"渲染历史对局失败: {str(e)}")

    def add_history_match(self, idx, similarity):
        """在面板中添加单个历史对局卡片。"""
        left = self.past_left[idx]
        right = self.past_right[idx]
        result = self.labels[idx]

        cur_left = np.zeros(56, dtype=float)
        cur_right = np.zeros(56, dtype=float)
        for name, entry in self.left_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_left[int(name) - 1] = float(v)
        for name, entry in self.right_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_right[int(name) - 1] = float(v)

        setL_cur = set(np.where(cur_left > 0)[0])
        setR_cur = set(np.where(cur_right > 0)[0])
        setL_past = set(np.where(left > 0)[0])
        setR_past = set(np.where(right > 0)[0])

        should_swap = len(setL_cur ^ setR_past) + len(setR_cur ^ setL_past) < len(setL_cur ^ setL_past) + len(setR_cur ^ setR_past)

        match_widget = QWidget()
        match_widget.setStyleSheet(
            """
            QWidget { background-color: rgba(50, 50, 50, 150); border-radius: 10px; padding: 0px; margin: 5px; }
            """
        )
        match_widget.setFixedSize(500, 150)
        match_layout = QVBoxLayout(match_widget)

        teams_widget = QWidget(); teams_layout = QHBoxLayout(teams_widget)
        if should_swap:
            left_team = self.create_team_widget("右方", right, result == "R")
            right_team = self.create_team_widget("左方", left, result == "L")
        else:
            left_team = self.create_team_widget("左方", left, result == "L")
            right_team = self.create_team_widget("右方", right, result == "R")
        teams_layout.addWidget(left_team)
        teams_layout.addWidget(right_team)
        match_layout.addWidget(teams_widget)
        self.history_layout.addWidget(match_widget)

    def create_team_widget(self, side, counts, is_winner):
        """创建历史对局中的单侧队伍展示。"""
        team_widget = QWidget()
        team_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {'rgba(250, 250, 50, 150)' if is_winner else 'rgba(50, 50, 50, 100)'};
                border-radius: 8px; padding: 0px; margin: 0px;
            }}
            """
        )
        layout = QVBoxLayout(team_widget)

        ops_widget = QWidget()
        shadow01 = QGraphicsDropShadowEffect(); shadow01.setBlurRadius(5); shadow01.setColor(QColor(0, 0, 0, 120)); shadow01.setOffset(3)
        ops_widget.setGraphicsEffect(shadow01)
        ops_widget.setStyleSheet("""
            QWidget { background-color: rgba(0, 0, 0, 0); border-radius: 0px; padding: 0px; margin: 0px; }
        """)
        ops_layout = QHBoxLayout(ops_widget); ops_layout.setSpacing(5); ops_layout.setContentsMargins(0, 0, 0, 0)

        for i, count in enumerate(counts):
            if count > 0:
                op_widget = QWidget(); op_widget.setStyleSheet("background-color: rgba(0,0,0,0); padding: 0px 0; margin: 0px;")
                op_layout = QVBoxLayout(op_widget); op_layout.setContentsMargins(0,0,0,0); op_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

                img_label = QLabel(); img_label.setFixedSize(60, 60); img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                try:
                    pixmap = QPixmap(f"images/{i + 1}.png")
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        img_label.setPixmap(pixmap)
                except Exception:
                    pass

                count_label = QLabel(str(int(count)))
                count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                count_label.setStyleSheet("""
                    color: #EDEDED; font: bold 20px SimHei; min-width: 20px;
                """)

                op_layout.addWidget(img_label, stretch=3)
                op_layout.addWidget(count_label, stretch=1)
                ops_layout.addWidget(op_widget)

        layout.addWidget(ops_widget)
        return team_widget

    # ------------------------------ 其它操作 ------------------------------
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
                training_duration=float(self.duration_entry.text()) * 3600,  # 小时→秒
            )
            self.auto_fetch.start_auto_fetch()
        else:
            self.auto_fetch.stop_auto_fetch()

    def update_statistics(self):
        elapsed_time = time.time() - self.auto_fetch.start_time if self.auto_fetch.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.auto_fetch.total_fill_count}, "
            f"填写×次数: {self.auto_fetch.incorrect_fill_count}, "
            f"本次运行时长: {int(hours)}小时{int(minutes)}分钟"
        )
        self.stats_label.setText(stats_text)

    def update_device_serial(self):
        new_serial = self.serial_entry.text()
        self.adb_connector.set_device_serial(new_serial)
        self.adb_connector.device_serial = None
        self.adb_connector.get_device_serial()
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {new_serial}")

    # ---- AutoFetch 回调桥 ----
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

    # ------------------------------- 沙盒模拟 -------------------------------
    def run_simulation(self):
        """汇总左右怪物→JSON，经 stdin 传递给 `main_sim.py` 子进程。"""
        left_monsters_data = {}
        right_monsters_data = {}

        # 左侧
        for monster_id, entry in self.left_monsters.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                try:
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        left_monsters_data[monster_name] = int(count)
                except ValueError:
                    print(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    print(f"Error getting monster name for ID {monster_id}: {e}")

        # 右侧
        for monster_id, entry in self.right_monsters.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                try:
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        right_monsters_data[monster_name] = int(count)
                except ValueError:
                    print(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    print(f"Error getting monster name for ID {monster_id}: {e}")

        simulation_data = {"left": left_monsters_data, "right": right_monsters_data}
        json_data = json.dumps(simulation_data, ensure_ascii=False)
        print(f"Simulation data JSON: {json_data}")
        try:
            process = subprocess.Popen([sys.executable, "main_sim.py"], stdin=subprocess.PIPE, text=True, encoding="utf-8")
            process.stdin.write(json_data)
            process.stdin.close()
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到 main_sim.py 文件，请检查路径。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动模拟器时发生错误: {str(e)}")

    def get_monster_name_by_id(self, monster_id: int):
        """根据 UI 中的 1-based ID 获取怪物名（依赖 `simulator.utils.MONSTER_MAPPING`）。"""
        try:
            from simulator.utils import MONSTER_MAPPING
            return MONSTER_MAPPING.get(monster_id - 1)
        except ImportError:
            print("Error importing MONSTER_MAPPING from simulator.utils")
            return None

    # ------------------------------ 其它小方法 ------------------------------
    def update_game_mode(self, mode):
        self.game_mode = mode

    def update_invest_status(self, state):
        self.is_invest = state == Qt.CheckState.Checked

    def update_result(self, text):
        self.result_label.setText(text)

    def update_stats(self, total, incorrect, duration):
        stats_text = f"总共: {total}, 错误: {incorrect}, 时长: {duration}"
        self.stats_label.setText(stats_text)

    def update_image_display(self, qimage):
        self.image_display.setPixmap(
            QPixmap.fromImage(qimage).scaled(self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio)
        )


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec()
