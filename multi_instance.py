import sys
import threading
import time
import logging
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QPlainTextEdit, QSpinBox, QComboBox, QCheckBox,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import loadData
import auto_fetch
import data_package
from recognize import MONSTER_COUNT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='multi_instance.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

class DeviceInstance:
    def __init__(self, port):
        self.port = port
        self.serial = f"127.0.0.1:{port}"
        self.connector = loadData.AdbConnector(self.serial)
        self.auto_fetch = None
        self.status = "已停止"

    def start(self, game_mode, is_invest):
        try:
            logger.info(f"[{self.serial}] 开始启动实例，游戏模式: {game_mode}, 自动投资: {is_invest}")
            self.connector.connect()
            if not self.connector.is_connected:
                self.status = "连接失败"
                logger.error(f"[{self.serial}] 连接失败")
                return False
            
            self.auto_fetch = auto_fetch.AutoFetch(
                self.connector,
                game_mode,
                is_invest,
                update_prediction_callback=lambda x: None,
                update_monster_callback=lambda x: None,
                updater=lambda: None,
                start_callback=lambda: None,
                stop_callback=lambda: None,
                training_duration=-1
            )
            logger.info(f"[{self.serial}] 初始化 AutoFetch 成功")
            self.auto_fetch.start_auto_fetch()
            self.status = "正在运行"
            logger.info(f"[{self.serial}] 启动成功，状态: {self.status}")
            return True
        except Exception as e:
            self.status = f"错误: {str(e)}"
            logger.error(f"[{self.serial}] 启动失败: {str(e)}")
            return False

    def stop(self):
        logger.info(f"[{self.serial}] 开始停止实例")
        if self.auto_fetch:
            self.auto_fetch.stop_auto_fetch()
            logger.info(f"[{self.serial}] 停止 AutoFetch 成功")
        self.status = "已停止"
        logger.info(f"[{self.serial}] 停止成功，状态: {self.status}")

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
                f"时长: {int(hours)}小时{int(minutes)}分钟")

class MultiInstanceManager(QMainWindow):
    instance = None  # 类变量，用于在回调函数中引用当前实例
    
    def __init__(self):
        super().__init__()
        MultiInstanceManager.instance = self  # 保存当前实例的引用
        self.setWindowTitle("铁鲨鱼多开自动化工具")
        self.setGeometry(100, 100, 450, 600)  # 增加窗口高度以容纳日志显示
        
        self.instances = {}
        self.init_ui()
        self.setup_logger()
        
        # 定时更新界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)
    
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
        
        layout.addLayout(settings_layout)
        
        # 端口输入
        layout.addWidget(QLabel("ADB端口 (每行一个，例如 5555):"))
        self.ports_input = QPlainTextEdit()
        # 尝试读取历史端口
        try:
            if Path("multi_ports.txt").exists():
                self.ports_input.setPlainText(Path("multi_ports.txt").read_text())
        except:
            pass
        self.ports_input.setPlaceholderText("5555\n5557\n5559")
        self.ports_input.setFixedHeight(100)
        layout.addWidget(self.ports_input)
        
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
        
        # 状态显示
        layout.addWidget(QLabel("运行状态:"))
        self.status_display = QPlainTextEdit()
        self.status_display.setReadOnly(True)
        # 使用等宽字体以保证列对齐
        font = QFont("Courier New", 10)
        if sys.platform == "win32":
            font = QFont("Consolas", 10)
        self.status_display.setFont(font)
        layout.addWidget(self.status_display)
        
        # 日志显示
        layout.addWidget(QLabel("详细日志:"))
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(font)
        layout.addWidget(self.log_display)

    def start_all(self):
        # 保存当前端口配置
        try:
            Path("multi_ports.txt").write_text(self.ports_input.toPlainText())
            logger.info("保存端口配置到 multi_ports.txt")
        except Exception as e:
            logger.error(f"保存端口配置失败: {str(e)}")
            
        ports = [p.strip() for p in self.ports_input.toPlainText().split('\n') if p.strip()]
        game_mode = self.game_mode_combo.currentText()
        is_invest = self.invest_check.isChecked()
        
        logger.info(f"开始启动多开实例，端口列表: {ports}, 游戏模式: {game_mode}, 自动投资: {is_invest}")
        
        def run_start():
            for port in ports:
                # 检查是否已经在运行
                is_running = False
                if port in self.instances:
                    inst = self.instances[port]
                    if inst.auto_fetch and inst.auto_fetch.auto_fetch_running:
                        is_running = True
                
                if not is_running:
                    logger.info(f"启动端口 {port} 的实例")
                    instance = DeviceInstance(port)
                    if instance.start(game_mode, is_invest):
                        self.instances[port] = instance
                        logger.info(f"端口 {port} 的实例启动成功")
                    else:
                        logger.error(f"端口 {port} 的实例启动失败")
                    # 每个实例启动后延迟 2 秒，避免 ADB 冲突
                    time.sleep(2.0)
                else:
                    logger.info(f"端口 {port} 的实例已经在运行，跳过启动")
        
        threading.Thread(target=run_start, daemon=True).start()
    
    def stop_all(self):
        logger.info("开始停止所有实例")
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
        # 创建自定义日志处理器，只显示包含特定端口的日志
        class QTextEditLogger(logging.Handler):
            def __init__(self, text_edit):
                super().__init__()
                self.text_edit = text_edit
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.target_port = None  # 目标端口，只显示该端口的日志
                self.log_history = []  # 日志历史记录
            
            def set_target_port(self, port):
                self.target_port = port
            
            def emit(self, record):
                msg = self.format(record)
                
                # 保存日志消息到历史记录（无论是否过滤）
                self.log_history.append(msg)
                if len(self.log_history) > 1000:  # 最多保存1000条
                    self.log_history.pop(0)
                
                # 如果设置了目标端口，只显示包含该端口的日志
                if self.target_port:
                    # 检查日志消息中是否包含目标端口
                    port_str = str(self.target_port)
                    if f"[{port_str}]" not in msg and f"端口 {port_str}" not in msg and f"{port_str}" not in msg:
                        return
                
                # 在主线程中更新日志显示
                import threading
                if threading.current_thread() != threading.main_thread():
                    if self.text_edit is None:
                        return
                    try:
                        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                        QMetaObject.invokeMethod(
                            self.text_edit,
                            "appendPlainText",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(str, msg)
                        )
                    except:
                        pass  # 静默处理异常，避免警告
                else:
                    if self.text_edit is not None:
                        self.text_edit.appendPlainText(msg)
        
        # 获取根日志记录器
        root_logger = logging.getLogger()
        # 添加自定义处理器
        self.text_edit_logger = QTextEditLogger(self.log_display)
        self.text_edit_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.text_edit_logger)
        
        # 确保日志级别设置正确
        root_logger.setLevel(logging.INFO)
    
    def update_log_filter(self, text):
        # 更新日志过滤器
        if text == "全部端口":
            self.text_edit_logger.set_target_port(None)
            # 显示所有历史日志
            self.log_display.clear()
            for msg in self.text_edit_logger.log_history:
                self.log_display.appendPlainText(msg)
        else:
            self.text_edit_logger.set_target_port(text)
            # 只显示该端口的历史日志
            self.log_display.clear()
            for msg in self.text_edit_logger.log_history:
                if f"[{text}]" in msg or f"端口 {text}" in msg or text in msg:
                    self.log_display.appendPlainText(msg)
        # 不清空日志，保留历史记录
    
    def update_display(self):
        lines = []
        # 获取输入框中的所有端口
        input_ports = [p.strip() for p in self.ports_input.toPlainText().split('\n') if p.strip()]
        
        # 更新端口选择器的选项
        current_text = self.port_combo.currentText()
        # 保存当前的目标端口
        current_target_port = self.text_edit_logger.target_port
        
        self.port_combo.clear()
        self.port_combo.addItem("全部端口")
        for port in input_ports:
            self.port_combo.addItem(port)
        
        # 如果有输入端口，检查当前选择是否有效
        if input_ports:
            # 如果当前选择是一个有效的端口，保持该选择
            if current_text in input_ports:
                self.port_combo.setCurrentText(current_text)
            # 否则，默认选择第一个端口
            else:
                self.port_combo.setCurrentIndex(1)  # 1 是第一个端口的索引（0 是"全部端口"）
        # 否则，保持之前的选择
        elif current_text in ["全部端口"] + input_ports:
            self.port_combo.setCurrentText(current_text)
        
        # 恢复之前的目标端口
        if current_target_port:
            if current_target_port in input_ports:
                self.text_edit_logger.set_target_port(current_target_port)
                self.port_combo.setCurrentText(current_target_port)
                # 重新应用日志过滤器
                self.update_log_filter(current_target_port)
            else:
                self.text_edit_logger.set_target_port(None)
                self.port_combo.setCurrentText("全部端口")
                # 重新应用日志过滤器
                self.update_log_filter("全部端口")
        else:
            # 如果之前没有目标端口，使用当前选择的端口
            current_selected = self.port_combo.currentText()
            # 重新应用日志过滤器
            self.update_log_filter(current_selected)
        
        any_running = False
        for port in input_ports:
            if port in self.instances:
                instance = self.instances[port]
                lines.append(instance.get_status_line())
                if instance.auto_fetch and instance.auto_fetch.auto_fetch_running:
                    any_running = True
            else:
                serial = f"127.0.0.1:{port}"
                lines.append(f"[{serial:<15}] 状态: 已停止")
        
        # 只有在全部停止状态下才能打包
        self.package_btn.setEnabled(not any_running)
                
        self.status_display.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.stop_all()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiInstanceManager()
    window.show()
    sys.exit(app.exec())
