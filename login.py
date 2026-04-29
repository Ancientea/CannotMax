import os
# 设置 OpenCV 日志级别为 ERROR，减少 libpng 警告
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import logging
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class LoginManager:
    """登录管理器，处理游戏登录和页面跳转"""
    
    def __init__(self, connector, max_restart_count=3):
        self.connector = connector
        self.template_dir = Path("images") / "login"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.restart_count = 0
        self.max_restart_count = max_restart_count
        try:
            self._load_templates()
        except Exception as e:
            logger.error(f"模板加载失败: {e}")
            self.templates = {}
    
    def _log(self, level, message):
        """生成带有设备序列号的日志消息"""
        serial = getattr(self.connector, "device_serial", None)
        if serial:
            logger.log(level, f"[{serial}] {message}")
        else:
            logger.log(level, message)
    
    def reset_restart_count(self):
        """重置重启计数器"""
        self.restart_count = 0
        self._log(logging.INFO, "重启计数器已重置")
    
    def can_restart(self):
        """检查是否可以重启"""
        return self.restart_count < self.max_restart_count
    
    def _load_templates(self):
        """加载模板图片"""
        self.templates = {}
        template_files = self.template_dir.glob("*.png")
        for template_file in template_files:
            template_name = template_file.stem
            template = cv2.imread(str(template_file))
            if template is not None:
                self.templates[template_name] = template
                logger.info(f"加载模板: {template_name}")
    
    def match_template(self, screenshot, template_name, threshold=0.9):
        """匹配模板"""
        if template_name not in self.templates:
            logger.error(f"模板不存在: {template_name}")
            return False, (0, 0)
        
        template = self.templates[template_name]
        
        # 转换为灰度图像
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 多尺度模板匹配
        found = None
        for scale in np.linspace(0.5, 1.5, 10):
            # 调整模板大小
            resized = cv2.resize(template_gray, (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale)))
            if resized.shape[0] > screenshot_gray.shape[0] or resized.shape[1] > screenshot_gray.shape[1]:
                break
            
            # 匹配
            result = cv2.matchTemplate(screenshot_gray, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 更新最佳匹配
            if found is None or max_val > found[0]:
                found = (max_val, max_loc, scale)
        
        if found:
            max_val, max_loc, scale = found
            h, w = int(template.shape[0] * scale), int(template.shape[1] * scale)
            if max_val >= threshold:
                logger.info(f"匹配到模板 {template_name}，置信度: {max_val}")
                # 返回模板中心点坐标
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                return True, (center_x, center_y)
            else:
                logger.debug(f"未匹配到模板 {template_name}，最高置信度: {max_val}")
                return False, (0, 0)
        else:
            logger.debug(f"未匹配到模板 {template_name}")
            return False, (0, 0)
    


    def restart_game(self):
        """重启游戏"""
        self._log(logging.INFO, "开始重启游戏")
        
        # 确定连接类型
        is_pc = hasattr(self.connector, 'hwnd') and self.connector.hwnd
        is_adb = hasattr(self.connector, 'device_serial') and self.connector.device_serial
        
        if not is_pc and not is_adb:
            self._log(logging.ERROR, "无法确定连接类型，无法重启游戏")
            return False
        
        # 关闭游戏进程
        try:
            if is_pc:
                # 对于PC端，关闭游戏进程
                import win32gui
                import win32process
                import win32api
                _, process_id = win32process.GetWindowThreadProcessId(self.connector.hwnd)
                process = win32api.OpenProcess(1, False, process_id)
                win32api.TerminateProcess(process, 0)
                win32api.CloseHandle(process)
                self._log(logging.INFO, "关闭游戏进程成功")
            else:
                # 对于ADB端，关闭游戏进程
                adb_path = getattr(self.connector, 'adb_path', 'adb')
                subprocess.run(f"{adb_path} -s {self.connector.device_serial} shell am force-stop com.hypergryph.arknights", shell=True)
                self._log(logging.INFO, "关闭游戏进程成功")
        except Exception as e:
            self._log(logging.ERROR, f"关闭游戏进程失败: {e}")
            return False
        
        # 等待一段时间后重新启动游戏
        time.sleep(3)
        
        try:
            if is_pc:
                # 对于PC端，重新启动游戏
                # 尝试从注册表获取游戏路径
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        subkey_name = winreg.EnumKey(key, i)
                        subkey = winreg.OpenKey(key, subkey_name)
                        try:
                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            if "Arknights" in display_name:
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                game_path = Path(install_location) / "Arknights.exe"
                                if game_path.exists():
                                    subprocess.Popen(str(game_path))
                                    self._log(logging.INFO, f"从注册表获取游戏路径并启动: {game_path}")
                                    break
                        except:
                            pass
                    else:
                        # 如果从注册表获取失败，使用默认路径
                        game_path = Path("C:\\Program Files\\Arknights\\Arknights.exe")
                        if game_path.exists():
                            subprocess.Popen(str(game_path))
                            self._log(logging.INFO, f"使用默认路径启动游戏: {game_path}")
                        else:
                            self._log(logging.ERROR, "无法找到游戏可执行文件")
                            return False
                except:
                    # 如果注册表操作失败，使用默认路径
                    game_path = Path("C:\\Program Files\\Arknights\\Arknights.exe")
                    if game_path.exists():
                        subprocess.Popen(str(game_path))
                        self._log(logging.INFO, f"使用默认路径启动游戏: {game_path}")
                    else:
                        self._log(logging.ERROR, "无法找到游戏可执行文件")
                        return False
            else:
                # 对于ADB端，重新启动游戏
                adb_path = getattr(self.connector, 'adb_path', 'adb')
                subprocess.run(f"{adb_path} -s {self.connector.device_serial} shell am start -n com.hypergryph.arknights/com.u8.sdk.U8UnityContext")
                self._log(logging.INFO, "重新启动游戏成功")
        except Exception as e:
            self._log(logging.ERROR, f"重新启动游戏失败: {e}")
            return False
        
        # 重新连接
        try:
            self.connector.connect()
            if self.connector.is_connected:
                self._log(logging.INFO, "游戏重启后重新连接成功")
            else:
                self._log(logging.WARNING, "游戏重启后重新连接失败")
        except Exception as e:
            self._log(logging.ERROR, f"重新连接失败: {e}")
        
        return True

    def auto_login(self, first_start=False, stop_callback=None):
        """自动登录功能，处理服务器维护或掉线后的重新登录"""
        self._log(logging.INFO, "开始自动登录流程")
        
        def check_stop():
            if stop_callback and not stop_callback():
                self._log(logging.INFO, "检测到停止信号，中断登录流程")
                return False
            return True
        
        def sleep_with_check(seconds):
            """可中断的等待函数"""
            start_time = time.time()
            while time.time() - start_time < seconds:
                if not check_stop():
                    return False
                time.sleep(0.1)
            return True
        
        # 确保连接器已连接
        if not hasattr(self.connector, 'is_connected') or not self.connector.is_connected:
            logger.info("连接器未连接，尝试重新连接")
            try:
                self.connector.connect()
                if not self.connector.is_connected:
                    logger.error("重新连接失败，登录流程中断")
                    return False
            except Exception as e:
                logger.error(f"重新连接失败: {e}")
                return False
        
        # 首次启动时，先检测是否已经在游戏流程中
        if first_start:
            logger.info("首次启动，检测是否已在游戏流程中")
            
            # 检查是否需要停止
            if not check_stop():
                return False
            
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                # 检查是否匹配到争锋频道入口
                matched, _ = self.match_template(screenshot, "competition_page", threshold=0.7)
                if matched:
                    logger.info("已在争锋频道入口，无需登录")
                    return True
                
                # 检查是否需要停止
                if not check_stop():
                    return False
                
                # 检查是否匹配到0.png或1.png（加入赛事或开始游戏）
                try:
                    # 尝试简单匹配
                    for template_name in ["0", "1"]:
                        # 检查是否需要停止
                        if not check_stop():
                            return False
                        
                        template_path = Path(f"images/process/{template_name}.png")
                        if template_path.exists():
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
                                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                                res = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(res)
                                if max_val > 0.7:
                                    logger.info(f"已在争锋频道页面，找到模板 {template_name}.png")
                                    return True
                except Exception as e:
                    logger.debug(f"检测争锋频道页面模板失败: {e}")
                
                # 检查是否需要停止
                if not check_stop():
                    return False
                
                # 检查是否在战斗前准备阶段（PRE_BATTLE）
                # 匹配战斗前准备相关的模板（模板索引3,4,5,15）
                for template_name in ["3", "4", "5", "15"]:
                    # 检查是否需要停止
                    if not check_stop():
                        return False
                    
                    template_path = Path(f"images/process/{template_name}.png")
                    if template_path.exists():
                        try:
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
                                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                                res = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(res)
                                if max_val > 0.7:
                                    logger.info(f"已在战斗前准备阶段，找到模板 {template_name}.png，无需登录")
                                    return True
                        except Exception as e:
                            logger.debug(f"检测战斗前准备模板 {template_name}.png 失败: {e}")
        
        # 非首次启动时，等待游戏启动
        if not first_start:
            self._log(logging.INFO, "重启游戏，等待游戏启动...")
            if not sleep_with_check(40):
                return False
        
        # 点击屏幕中心跳过中转页面（点击3次，每次间隔2秒）
        self._log(logging.INFO, "点击屏幕中心跳过中转页面")
        for i in range(3):
            self.connector.click((0.5, 0.5))
            self._log(logging.INFO, f"第 {i+1} 次点击屏幕中心")
            if i < 2:
                if not sleep_with_check(2):
                    return False
        
        # 寻找并点击登录按钮
        self._log(logging.INFO, "寻找登录按钮")
        login_button_found = False
        start_time = time.time()
        while time.time() - start_time < 40:
            if not check_stop():
                return False
            screenshot = self.connector.capture_screenshot()
            if screenshot is None:
                self._log(logging.WARNING, "获取截图失败，重试")
                if not sleep_with_check(1):
                    return False
                continue
            
            h, w = screenshot.shape[:2]
            self._log(logging.DEBUG, f"截图尺寸: {w}x{h}")
            
            matched, pos = self.match_template(screenshot, "login_button", threshold=0.9)
            if matched:
                rel_x = pos[0] / w
                rel_y = pos[1] / h
                self._log(logging.INFO, f"登录按钮位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                self.connector.click((rel_x, rel_y))
                self._log(logging.INFO, "点击登录按钮")
                login_button_found = True
                break
            if not sleep_with_check(1):
                return False
        
        if not login_button_found:
            self._log(logging.ERROR, "未找到登录按钮，登录流程中断")
            return False
        
        # 等待登录完成，最多等待70秒
        self._log(logging.INFO, "等待登录完成")
        start_time = time.time()
        while time.time() - start_time < 70:
            if not check_stop():
                return False
            # 简单的等待，不进行复杂的页面检测
            if not sleep_with_check(1):
                return False
        
        # 寻找争锋频道页面，最多等待30秒
        self._log(logging.INFO, "寻找争锋频道入口")
        start_time = time.time()
        
        # 尝试识别争锋频道
        screenshot = self.connector.capture_screenshot()
        if screenshot is not None:
            if self._check_and_click_competition_page(screenshot, sleep_with_check, check_stop):
                return True
        
        # 识别失败时点击右上角两次
        click_count = 0
        while click_count < 2:
            self._log(logging.INFO, "未检测到争锋频道入口，点击屏幕右上角")
            self.connector.click((0.1, 0.1))
            if not sleep_with_check(2):
                return False
            click_count += 1
            
            # 点击右上角后，立即再次检查争锋频道入口
            if not sleep_with_check(1):
                return False
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                if self._check_and_click_competition_page(screenshot, sleep_with_check, check_stop):
                    return True
        
        # 识别失败后识别关闭按钮
        close_buttons = ["announcement_close", "event_claim_close"]
        close_button_found = False
        for button in close_buttons:
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                matched, pos = self.match_template(screenshot, button, threshold=0.9)
                if matched:
                    h, w = screenshot.shape[:2]
                    rel_x = pos[0] / w
                    rel_y = pos[1] / h
                    self._log(logging.INFO, f"{button}位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                    self.connector.click((rel_x, rel_y))
                    self._log(logging.INFO, f"关闭{button}页面")
                    close_button_found = True
                    if not sleep_with_check(1):
                        return False
                    break
        
        # 再次识别争锋频道入口
        if close_button_found:
            if not sleep_with_check(1):
                return False
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                if self._check_and_click_competition_page(screenshot, sleep_with_check, check_stop):
                    return True
        
        # 还是失败的话就重启
        self._log(logging.ERROR, "未找到争锋频道入口，登录流程失败")
        return False
    
    def _check_and_click_competition_page(self, screenshot, sleep_with_check, check_stop):
        """争锋频道入口检测和点击方法"""
        matched, pos = self.match_template(screenshot, "competition_page", threshold=0.7)
        if matched:
            h, w = screenshot.shape[:2]
            rel_x = pos[0] / w
            rel_y = pos[1] / h
            self._log(logging.INFO, f"争锋频道入口位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
            self.connector.click((rel_x, rel_y))
            self._log(logging.INFO, "点击进入争锋频道页面")
            if not sleep_with_check(2):
                return False
            self._log(logging.INFO, "自动登录流程完成")
            return True
        return False
    
    def try_login_with_retry(self, max_wait_seconds=6, stop_callback=None):
        """尝试登录，如果未找到登录按钮则等待重试"""
        for i in range(max_wait_seconds):
            # 检查是否需要停止
            if stop_callback and not stop_callback():
                logger.info("检测到停止信号，取消登录尝试")
                return False
            
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                logger.info("获取截图成功，检查是否存在登录按钮")
                matched, _ = self.match_template(screenshot, "login_button", threshold=0.9)
                if matched:
                    logger.info("找到登录按钮，执行登录流程")
                    # 传递 stop_callback 给 auto_login
                    if self.auto_login(stop_callback=stop_callback):
                        logger.info("登录成功")
                        return True
                    else:
                        return False
                else:
                    logger.info(f"第 {i+1} 次检查：未找到登录按钮，继续等待")
            
            # 可中断的等待
            start_time = time.time()
            while time.time() - start_time < 6:
                if stop_callback and not stop_callback():
                    logger.info("检测到停止信号，取消登录尝试")
                    return False
                time.sleep(0.1)
        return False
    
    def restart_and_login(self, first_start=False, stop_callback=None):
        """重启游戏并尝试登录"""
        # 检查是否需要停止
        if stop_callback and not stop_callback():
            self._log(logging.INFO, "检测到停止信号，取消重启")
            return False
        
        self._log(logging.INFO, f"尝试重启游戏 (第 {self.restart_count + 1}/{self.max_restart_count} 次)")
        self.restart_count += 1
        
        if self.restart_game():
            self._log(logging.INFO, "游戏重启成功，尝试重新登录")
            
            # 检查是否需要停止
            if stop_callback and not stop_callback():
                self._log(logging.INFO, "检测到停止信号，取消登录")
                return False
            
            # 传递 stop_callback 给 auto_login
            if self.auto_login(first_start=first_start, stop_callback=stop_callback):
                return True
            else:
                self._log(logging.ERROR, "重启后自动登录失败")
                return False
        else:
            self._log(logging.ERROR, "重启游戏失败")
            return False
    
    def auto_login_with_restart(self, first_start=False, stop_callback=None):
        """自动登录，失败时自动重启重试"""
        # 首先尝试直接登录（首次启动）
        if self.auto_login(first_start=first_start, stop_callback=stop_callback):
            self.reset_restart_count()
            return True
        
        # 登录失败，尝试重启登录
        for _ in range(self.max_restart_count - 1):
            if not self.restart_and_login(first_start=False, stop_callback=stop_callback):
                self._log(logging.WARNING, "重启登录失败，继续尝试")
                continue
            return True
        
        self._log(logging.ERROR, f"已尝试 {self.max_restart_count} 次，登录失败")
        return False