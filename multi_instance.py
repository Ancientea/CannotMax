import sys
import threading
import time
import logging
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QPlainTextEdit, QSpinBox, QComboBox, QCheckBox,
    QMessageBox, QSplitter, QScrollArea, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtGui import QFont

import loadData
import auto_fetch
import data_package
from recognize import MONSTER_COUNT
from login import LoginManager
from config import FIELD_FEATURE_COUNT


class LogDisplay(QPlainTextEdit):
    log_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._auto_scroll = True
        self.log_signal.connect(self._on_log)
    
    def is_at_bottom(self):
        scrollbar = self.verticalScrollBar()
        return scrollbar.value() >= scrollbar.maximum() - 10
    
    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._auto_scroll = self.is_at_bottom()
    
    def _on_log(self, text):
        was_at_bottom = self._auto_scroll
        self.appendPlainText(text)
        if was_at_bottom:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def append_log(self, text):
        self.log_signal.emit(text)

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


class SmartPortsLineEdit(QLineEdit):
    """智能端口输入框：支持延迟格式化、失去焦点格式化和粘贴立即格式化"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.format_timer = QTimer(self)
        self.format_timer.setSingleShot(True)
        self.format_timer.timeout.connect(self._format_text)
        self._formatting = False

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if not self._formatting:
            self.format_timer.start(800)  # 800ms 延迟

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._format_text()

    def paste(self):
        super().paste()
        self.format_timer.start(100)  # 粘贴后快速格式化

    def _format_text(self):
        if self._formatting:
            return
        self._formatting = True
        
        text = self.text()
        parts = text.replace('\n', ',').replace('，', ',').replace(';', ',').replace(' ', ',').split(',')
        ports = [p.strip() for p in parts if p.strip().isdigit()]
        formatted = ", ".join(ports)
        
        if text != formatted and ports:
            cursor_pos = self.cursorPosition()
            self.setText(formatted)
            self.setCursorPosition(min(cursor_pos, len(formatted)))
        
        self._formatting = False

# 用于共享资源密集型对象，减少多开时的内存占用
_cannot_model = None
_recognizer = None
_field_recognizer = None


def get_cannot_model():
    """获取共享的 CannotModel 实例"""
    global _cannot_model
    if _cannot_model is None:
        logger.info("首次初始化 CannotModel...")
        try:
            from predict import CannotModel
            logger.info("Using PyTorch model for predictions.")
        except Exception:
            from predict_onnx import CannotModel
            logger.info("Using ONNX model for predictions.")
        
        _cannot_model = CannotModel()
        logger.info("CannotModel 初始化完成")
    return _cannot_model


def get_recognizer():
    """获取共享的 RecognizeMonster 实例"""
    global _recognizer
    if _recognizer is None:
        logger.info("首次初始化 RecognizeMonster...")
        from recognize import RecognizeMonster
        _recognizer = RecognizeMonster(method="ADB")
        logger.info("RecognizeMonster 初始化完成")
    return _recognizer


def get_field_recognizer():
    """获取共享的 FieldRecognizer 实例"""
    global _field_recognizer
    if _field_recognizer is None:
        logger.info("首次初始化 FieldRecognizer...")
        from field_recognition import FieldRecognizer
        _field_recognizer = FieldRecognizer()
        logger.info("FieldRecognizer 初始化完成")
    return _field_recognizer


def clear_all_singleton_resources():
    """清空所有单例缓存（用于测试或重新加载）"""
    global _cannot_model, _recognizer, _field_recognizer
    _cannot_model = None
    _recognizer = None
    _field_recognizer = None
    logger.info("所有单例资源已清空")


class DeviceInstance:
    def __init__(self, port):
        self.port = port
        self.serial = f"127.0.0.1:{port}"
        self.connector = loadData.AdbConnector(self.serial)
        self.auto_fetch = None
        self.login_manager = None
        self.status = "已停止"
        self.auto_fetch_thread = None  # 保存线程引用
        self.stop_event = threading.Event()  # 使用Event对象进行线程间通信，更可靠
        self.start_time = None  # 实例启动时间戳
        self.last_activity_time = time.time()  # 最后活动时间戳（用于检测崩溃）
        self.thread_running = False  # 线程是否正在运行的标志
        self.game_mode = None  # 保存游戏模式
        self.is_invest = None  # 保存投资设置

    def start(self, game_mode, is_invest):
        try:
            # 保存设置
            self.game_mode = game_mode
            self.is_invest = is_invest
            
            # 重置停止事件和标志
            self.stop_event.clear()
            self.thread_running = True
            self.last_activity_time = time.time()
            self.status = "连接中"
            
            logger.info(f"[{self.serial}] 开始启动实例，游戏模式: {game_mode}, 自动投资: {is_invest}")
            
            # 记录实例启动时间戳
            self.start_time = time.time()
            
            self.connector.connect()
            if not self.connector.is_connected:
                self.status = "连接失败"
                logger.error(f"[{self.serial}] 连接失败")
                self.thread_running = False
                return False
            
            self.login_manager = LoginManager(self.connector)
            logger.info(f"[{self.serial}] 尝试首次启动自动登录")
            self.status = "登录中"
            login_success = self.login_manager.auto_login_with_restart(first_start=True, stop_callback=lambda: not self.stop_event.is_set())
            
            # 检查是否在登录过程中用户点击了停止
            if self.stop_event.is_set():
                logger.info(f"[{self.serial}] 登录过程被用户停止")
                self.status = "已停止"
                self.thread_running = False
                return False
            
            if login_success:
                logger.info(f"[{self.serial}] 首次启动自动登录成功")
            else:
                logger.warning(f"[{self.serial}] 首次启动自动登录失败，继续启动")
            
            # 再次检查停止标志（防止在登录成功后、创建auto_fetch前用户点击停止）
            if self.stop_event.is_set():
                logger.info(f"[{self.serial}] 用户在登录成功后点击了停止")
                self.status = "已停止"
                self.thread_running = False
                return False
            
            self.auto_fetch = auto_fetch.AutoFetch(
                self.connector,
                game_mode,
                is_invest,
                update_prediction_callback=self._update_activity_time,
                update_monster_callback=self._update_activity_time,
                updater=self._update_activity_time,
                start_callback=lambda: None,
                stop_callback=self._on_stop_callback,
                training_duration=-1,
                recognizer=get_recognizer(),
                cannot_model=get_cannot_model(),
                field_recognizer=get_field_recognizer() if FIELD_FEATURE_COUNT > 0 else None,
                start_timestamp=self.start_time,  # 传递实例启动时间戳
            )
            logger.info(f"[{self.serial}] 初始化 AutoFetch 成功")
            self.auto_fetch.start_auto_fetch()
            self.status = "正在运行"
            logger.info(f"[{self.serial}] 启动成功，状态: {self.status}")
            return True
        except Exception as e:
            self.status = f"错误: {str(e)}"
            logger.error(f"[{self.serial}] 启动失败: {str(e)}")
            self.thread_running = False
            return False
    
    def _update_activity_time(self, *args, **kwargs):
        """更新活动时间"""
        self.last_activity_time = time.time()
        
    def _on_stop_callback(self):
        """当 auto_fetch 停止时的回调"""
        self.thread_running = False
        logger.info(f"[{self.serial}] auto_fetch 已停止")

    def stop(self):
        logger.info(f"[{self.serial}] 强制停止实例")
        # 设置停止事件，用于中断登录流程
        self.stop_event.set()
        self.thread_running = False
        
        if self.auto_fetch:
            # 强制设置停止标志，不等待线程退出
            self.auto_fetch.auto_fetch_running = False
            logger.info(f"[{self.serial}] 强制停止 AutoFetch")
        else:
            # 如果 auto_fetch 还未创建，说明可能正在登录过程中
            logger.info(f"[{self.serial}] auto_fetch 尚未创建，停止登录过程")
        
        # 状态改为已停止（stop_event 会在下次启动时自动清除）
        self.status = "已停止"
        logger.info(f"[{self.serial}] 强制停止成功，状态: {self.status}")

    def get_status_line(self):
        if not self.auto_fetch or not self.auto_fetch.auto_fetch_running:
            return f"[{self.serial:<15}] 状态: {self.status}"
        
        af = self.auto_fetch
        elapsed = time.time() - af.start_time if af.start_time else 0
        hours, remainder = divmod(elapsed, 3600)
        minutes, _ = divmod(remainder, 60)
        
        state_name = "过场动画"
        if hasattr(af, 'last_state') and af.last_state:
            state_name = af.last_state.name if hasattr(af.last_state, 'name') else str(af.last_state)
        
        return (f"[{self.serial:<15}] "
                f"状态: {state_name:<8} | "
                f"填写: {af.total_fill_count:<3} | "
                f"错误: {af.incorrect_fill_count:<3} | "
                f"预测: {af.current_prediction:.2f} | "
                f"时长: {int(hours)}h {int(minutes)}m")

class MultiInstanceManager(QMainWindow):
    instance = None  # 类变量，用于在回调函数中引用当前实例
    
    def __init__(self):
        super().__init__()
        MultiInstanceManager.instance = self  # 保存当前实例的引用
        self.setWindowTitle("铁鲨鱼多开自动化工具")
        self.setGeometry(100, 100, 530, 720)
        
        self.instances = {}
        self.starting_ports = set()  # 正在启动中的端口集合，防止重复启动
        self.init_ui()
        self.setup_logger()
        
        # 定时更新界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)
        
        # 崩溃检测定时器（每 5 秒检查一次）
        self.crash_detection_timer = QTimer()
        self.crash_detection_timer.timeout.connect(self.check_instances_crash)
        self.crash_detection_timer.start(5000)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 配置区域
        settings_layout = QHBoxLayout()
        
        self.game_mode_combo = QComboBox()
        self.game_mode_combo.addItems(["单人", "30人"])
        settings_layout.addWidget(QLabel("模式:"))
        settings_layout.addWidget(self.game_mode_combo)
        
        self.invest_check = QCheckBox("自动投资")
        self.invest_check.setChecked(False)
        settings_layout.addWidget(self.invest_check)
        settings_layout.addStretch()
        
        layout.addLayout(settings_layout)
        
        # 端口输入（横向，逗号分隔）
        ports_layout = QHBoxLayout()
        ports_layout.addWidget(QLabel("端口:"))
        self.ports_input = SmartPortsLineEdit()
        self.ports_input.setPlaceholderText("16416, 16448, 16480, 16512")
        try:
            if Path("multi_ports.txt").exists():
                raw = Path("multi_ports.txt").read_text().strip()
                ports = self._parse_ports(raw)
                self.ports_input.setText(", ".join(ports))
        except:
            pass
        ports_layout.addWidget(self.ports_input)
        layout.addLayout(ports_layout)
        
        # 按钮和端口选择器
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("全部启动")
        self.start_btn.clicked.connect(self.start_all)
        self.stop_btn = QPushButton("全部停止")
        self.stop_btn.clicked.connect(self.stop_all)
        self.package_btn = QPushButton("打包数据")
        self.package_btn.clicked.connect(self.package_data)
        
        # 端口选择器，用于选择要查看哪个端口的日志
        self.port_combo = QComboBox()
        self.port_combo.addItem("全部端口")
        self.port_combo.currentTextChanged.connect(self.update_log_filter)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.package_btn)
        btn_layout.addWidget(QLabel("日志过滤:"))
        btn_layout.addWidget(self.port_combo)
        layout.addLayout(btn_layout)
        
        font = QFont("Courier New", 10)
        if sys.platform == "win32":
            font = QFont("Consolas", 10)
        
        # 使用 QSplitter 让状态和日志区域可以自由拖动调整大小
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 状态显示区域（带单独控制按钮的滚动区域）
        self.status_scroll = QScrollArea()
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.status_container = QWidget()
        self.status_layout = QVBoxLayout(self.status_container)
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_layout.setSpacing(2)
        self.status_layout.addStretch()
        self.status_scroll.setWidget(self.status_container)
        self.status_scroll.setMinimumHeight(80)
        splitter.addWidget(self.status_scroll)
        
        # 日志显示
        self.log_display = LogDisplay()
        self.log_display.setFont(font)
        self.log_display.setMaximumBlockCount(2000)
        splitter.addWidget(self.log_display)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([440, 330])
        layout.addWidget(splitter)
        
        self.port_widgets = {}

    @staticmethod
    def _parse_ports(text):
        parts = text.replace('\n', ',').replace('，', ',').replace(';', ',').replace(' ', ',').split(',')
        return [p.strip() for p in parts if p.strip().isdigit()]

    def start_all(self):
        try:
            Path("multi_ports.txt").write_text(self.ports_input.text())
            logger.info("保存端口配置到 multi_ports.txt")
        except Exception as e:
            logger.error(f"保存端口配置失败: {str(e)}")
            
        ports = self._parse_ports(self.ports_input.text())
        game_mode = self.game_mode_combo.currentText()
        is_invest = self.invest_check.isChecked()
        
        logger.info(f"开始启动多开实例，端口列表: {ports}, 游戏模式: {game_mode}, 自动投资: {is_invest}")
        
        def start_single_instance(port):
            # 检查是否已经在运行或正在启动中
            is_running = False
            if port in self.instances:
                inst = self.instances[port]
                if inst.auto_fetch and inst.auto_fetch.auto_fetch_running:
                    is_running = True
            
            # 使用启动锁防止重复启动
            if port in self.starting_ports:
                logger.info(f"端口 {port} 的实例正在启动中，跳过重复启动")
                return
            
            if not is_running:
                # 添加到启动中集合
                self.starting_ports.add(port)
                logger.info(f"启动端口 {port} 的实例")
                instance = DeviceInstance(port)
                # 立即添加到实例字典，以便 stop_all 可以找到它
                self.instances[port] = instance
                try:
                    if instance.start(game_mode, is_invest):
                        logger.info(f"端口 {port} 的实例启动成功")
                    else:
                        logger.error(f"端口 {port} 的实例启动失败")
                finally:
                    # 从启动中集合移除
                    self.starting_ports.discard(port)
            else:
                logger.info(f"端口 {port} 的实例已经在运行，跳过启动")
        
        def start_all_instances():
            threads = []
            for port in ports:
                t = threading.Thread(target=start_single_instance, args=(port,), daemon=True)
                t.start()
                threads.append(t)
                time.sleep(3)
            for t in threads:
                t.join()
        
        threading.Thread(target=start_all_instances, daemon=True).start()
    
    def stop_all(self):
        logger.info("开始停止所有实例")
        # 清空启动中集合，防止正在启动的实例继续运行
        self.starting_ports.clear()
        for instance in self.instances.values():
            instance.stop()
        self.instances.clear()
        logger.info("所有实例已停止并清除")
        self.update_display()

    def package_data(self):
        try:
            zip_filename = data_package.package_data()
            if zip_filename and Path(zip_filename).exists():
                # 在文件浏览器中高亮显示文件
                subprocess.run(f'explorer /select,"{Path(zip_filename).absolute()}"')
                QMessageBox.information(self, "成功", f"数据已打包到 {zip_filename}")
            else:
                QMessageBox.warning(self, "警告", "没有找到可以打包的数据目录或打包失败。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打包数据时发生错误: {str(e)}")

    def setup_logger(self):
        class QTextEditLogger(logging.Handler):
            def __init__(self, text_edit):
                super().__init__()
                self.text_edit = text_edit
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.target_port = None
                self.log_history = []
            
            def set_target_port(self, port):
                self.target_port = port
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                except Exception:
                    return
                
                self.log_history.append(msg)
                if len(self.log_history) > 2000:
                    self.log_history.pop(0)
                
                if self.target_port:
                    port_str = str(self.target_port)
                    if not (f"[127.0.0.1:{port_str}]" in msg or f"[{port_str}]" in msg or f"端口 {port_str}" in msg):
                        return
                
                if self.text_edit is not None:
                    self.text_edit.append_log(msg)
        
        root_logger = logging.getLogger()
        self.text_edit_logger = QTextEditLogger(self.log_display)
        self.text_edit_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.text_edit_logger)
        root_logger.setLevel(logging.INFO)
        
        # 确保子模块日志传播到 root logger
        for name in ['login', 'auto_fetch', 'recognize', 'multi_instance', '__main__']:
            child_logger = logging.getLogger(name)
            child_logger.setLevel(logging.INFO)
            child_logger.propagate = True
    
    def update_log_filter(self, text):
        if text == "全部端口":
            self.text_edit_logger.set_target_port(None)
            self.log_display.clear()
            for msg in self.text_edit_logger.log_history:
                self.log_display.append_log(msg)
        else:
            self.text_edit_logger.set_target_port(text)
            self.log_display.clear()
            for msg in self.text_edit_logger.log_history:
                if f"[{text}]" in msg or f"端口 {text}" in msg or text in msg:
                    self.log_display.append_log(msg)
    
    @staticmethod
    def _state_to_chinese(state):
        from auto_fetch import GameState
        state_map = {
            GameState.MAIN_MENU: "主页",
            GameState.MODE_SELECTION_UNSELECTED: "模式",
            GameState.MODE_SELECTION_SELECTED: "开始",
            GameState.PRE_BATTLE: "战前",
            GameState.IN_BATTLE: "战斗",
            GameState.SETTLEMENT: "结算",
            GameState.FINISHED: "结束",
            GameState.UNKNOWN: "过场",
        }
        return state_map.get(state, "过场动画")
    
    def _create_port_widget(self, port):
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)
        
        label = QLabel(port)
        label.setFixedWidth(60)
        label.setFont(QFont("Consolas", 10) if sys.platform == "win32" else QFont("Courier New", 10))
        row.addWidget(label)
        
        status_label = QLabel("已停止")
        status_label.setFixedWidth(80)
        row.addWidget(status_label)
        
        detail_label = QLabel("")
        detail_label.setFont(QFont("Consolas", 9) if sys.platform == "win32" else QFont("Courier New", 9))
        row.addWidget(detail_label, 1)
        
        toggle_btn = QPushButton("启动")
        toggle_btn.setFixedWidth(50)
        toggle_btn.clicked.connect(lambda checked, p=port: self._toggle_port(p))
        row.addWidget(toggle_btn)
        
        frame = QFrame()
        frame.setLayout(row)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        
        return frame, status_label, detail_label, toggle_btn
    
    def _toggle_port(self, port):
        if port in self.instances:
            instance = self.instances[port]
            if instance.auto_fetch and instance.auto_fetch.auto_fetch_running:
                instance.stop()
                logger.info(f"端口 {port} 已单独停止")
            else:
                del self.instances[port]
        else:
            game_mode = self.game_mode_combo.currentText()
            is_invest = self.invest_check.isChecked()
            
            def do_start():
                if port in self.starting_ports:
                    return
                self.starting_ports.add(port)
                instance = DeviceInstance(port)
                self.instances[port] = instance
                try:
                    instance.start(game_mode, is_invest)
                finally:
                    self.starting_ports.discard(port)
            
            threading.Thread(target=do_start, daemon=True).start()
    
    def check_instances_crash(self):
        """检查实例是否崩溃，并尝试自动恢复"""
        current_time = time.time()
        
        # 遍历所有实例，检查崩溃情况
        for port, instance in list(self.instances.items()):
            # 如果已经设置了停止事件，就不进行崩溃检测
            if instance.stop_event.is_set():
                continue
                
            # 检查活动时间（崩溃检测）
            inactive_time = current_time - instance.last_activity_time
            
            # 如果标记为运行但超过 3 分钟没有活动，视为崩溃
            if instance.status in ["正在运行", "连接中", "登录中"] and inactive_time > 180:
                logger.warning(f"[{instance.serial}] 检测到无活动超过 {inactive_time:.0f} 秒，可能已崩溃，尝试重启")
                self._restart_crashed_instance(port, instance)
            # 检查 auto_fetch 线程是否实际还在运行
            elif instance.status == "正在运行" and instance.auto_fetch:
                if not instance.auto_fetch.auto_fetch_running:
                    logger.warning(f"[{instance.serial}] auto_fetch_running 为 False，可能已意外停止")
                    self._restart_crashed_instance(port, instance)
    
    def _restart_crashed_instance(self, port, instance):
        """重启崩溃的实例"""
        try:
            # 保存设置
            game_mode = instance.game_mode if instance.game_mode else "30人"
            is_invest = instance.is_invest if instance.is_invest is not None else False
            
            # 先停止（强制）
            instance.stop()
            
            # 从字典中清除
            if port in self.instances:
                del self.instances[port]
            
            # 延迟重新启动
            logger.info(f"[{instance.serial}] 准备重新启动...")
            
            def restart_task():
                if port in self.starting_ports:
                    logger.warning(f"端口 {port} 已在启动中，跳过重启")
                    return
                
                logger.info(f"[{instance.serial}] 正在重新启动...")
                self.starting_ports.add(port)
                
                try:
                    new_instance = DeviceInstance(port)
                    self.instances[port] = new_instance
                    success = new_instance.start(game_mode, is_invest)
                    if success:
                        logger.info(f"[{instance.serial}] 崩溃后重启成功")
                    else:
                        logger.error(f"[{instance.serial}] 崩溃后重启失败")
                finally:
                    self.starting_ports.discard(port)
            
            threading.Thread(target=restart_task, daemon=True).start()
        except Exception as e:
            logger.error(f"重启崩溃实例时出错: {str(e)}")
    
    def update_display(self):
        input_ports = self._parse_ports(self.ports_input.text())
        
        if not hasattr(self, '_last_input_ports'):
            self._last_input_ports = []
        
        ports_changed = input_ports != self._last_input_ports
        
        if ports_changed:
            current_text = self.port_combo.currentText()
            self.port_combo.clear()
            self.port_combo.addItem("全部端口")
            for port in input_ports:
                self.port_combo.addItem(port)
            
            if input_ports:
                if current_text in input_ports:
                    self.port_combo.setCurrentText(current_text)
                else:
                    self.port_combo.setCurrentIndex(1)
            self._last_input_ports = input_ports.copy()
            
            # 重建端口控件
            for w in self.port_widgets.values():
                w['frame'].setParent(None)
            self.port_widgets.clear()
            
            for i in range(self.status_layout.count() - 1):
                item = self.status_layout.itemAt(i)
                if item.widget():
                    item.widget().setParent(None)
            
            for port in input_ports:
                frame, status_label, detail_label, toggle_btn = self._create_port_widget(port)
                idx = self.status_layout.count() - 1
                self.status_layout.insertWidget(idx, frame)
                self.port_widgets[port] = {
                    'frame': frame,
                    'status_label': status_label,
                    'detail_label': detail_label,
                    'toggle_btn': toggle_btn,
                }
        
        any_running = False
        for port in input_ports:
            if port not in self.port_widgets:
                continue
            
            widgets = self.port_widgets[port]
            
            if port in self.instances:
                instance = self.instances[port]
                is_running = instance.auto_fetch and instance.auto_fetch.auto_fetch_running
                
                if is_running:
                    any_running = True
                    af = instance.auto_fetch
                    elapsed = time.time() - af.start_time if af.start_time else 0
                    hours, remainder = divmod(elapsed, 3600)
                    minutes, _ = divmod(remainder, 60)
                    
                    state_name = "过场动画"
                    if hasattr(af, 'last_state') and af.last_state:
                        state_name = self._state_to_chinese(af.last_state)
                    
                    widgets['status_label'].setText(f"运行中·{state_name}")
                    widgets['detail_label'].setText(
                        f"填写: {af.total_fill_count} | 错误: {af.incorrect_fill_count} | "
                        f"预测: {af.current_prediction:.2f} | 时长: {int(hours)}h {int(minutes)}m"
                    )
                    widgets['toggle_btn'].setText("停止")
                else:
                    widgets['status_label'].setText(instance.status)
                    widgets['detail_label'].setText("")
                    is_starting = instance.status in ("连接中", "登录中")
                    widgets['toggle_btn'].setText("停止" if is_starting else "启动")
            else:
                widgets['status_label'].setText("已停止")
                widgets['detail_label'].setText("")
                widgets['toggle_btn'].setText("启动")
        
        self.package_btn.setEnabled(not any_running)

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiInstanceManager()
    window.show()
    sys.exit(app.exec())
