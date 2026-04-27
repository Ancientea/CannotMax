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
    
    def __init__(self, connector):
        self.connector = connector
        self.template_dir = Path("images") / "login"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._load_templates()
        except Exception as e:
            logger.error(f"模板加载失败: {e}")
            self.templates = {}
    
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
    
    def is_in_competition_page(self, match_images_func=None, running_flag=None):
        """验证是否在争锋频道页面"""
        # 确保连接器已连接
        if not hasattr(self.connector, 'is_connected') or not self.connector.is_connected:
            logger.error("连接器未连接，无法验证页面")
            return False
        
        # 尝试多次验证，确保页面完全加载
        for attempt in range(3):
            # 检查是否需要停止验证
            if running_flag is not None and not running_flag():
                logger.info("自动获取已停止，停止页面验证")
                return False
            
            # 尝试获取截图，最多尝试3次
            screenshot = None
            for i in range(3):
                # 检查是否需要停止验证
                if running_flag is not None and not running_flag():
                    logger.info("自动获取已停止，停止页面验证")
                    return False
                
                screenshot = self.connector.capture_screenshot()
                if screenshot is not None:
                    break
                logger.warning(f"获取截图失败，尝试第 {i+1} 次")
                time.sleep(1)
            
            if screenshot is None:
                logger.error("无法获取截图，无法验证页面")
                time.sleep(2)
                continue
            
            # 优先使用自定义匹配函数，检查是否匹配到0.png或1.png（加入赛事或开始游戏）
            if match_images_func:
                try:
                    results = match_images_func(screenshot)
                    if results:
                        # 检查是否匹配到0.png或1.png（加入赛事或开始游戏）
                        for template_info, confidence in results:
                            if isinstance(template_info, int):
                                if template_info in [0, 1] and confidence > 0.7:
                                    logger.info(f"确认在争锋频道页面，找到模板 {template_info}.png")
                                    return True
                except Exception as e:
                    logger.error(f"匹配图片时出错: {e}")
            
            # 如果未匹配到0/1，尝试匹配争锋频道页面入口模板
            matched, _ = self.match_template(screenshot, "competition_page")
            if matched:
                logger.info("确认在争锋频道页面入口")
                return True
            
            # 检查是否是登录下线页面
            matched, _ = self.match_template(screenshot, "login_off")
            if matched:
                logger.info("检测到登录下线页面")
                return False
            
            logger.debug(f"第{attempt+1}次验证：未在争锋频道页面")
            time.sleep(3)  # 增加等待时间，确保页面有足够的加载时间
        
        logger.warning("不在争锋频道页面")
        return False

    def restart_game(self):
        """重启游戏"""
        logger.info("开始重启游戏")
        
        # 确定连接类型
        is_pc = hasattr(self.connector, 'hwnd') and self.connector.hwnd
        is_adb = hasattr(self.connector, 'device_serial') and self.connector.device_serial
        
        if not is_pc and not is_adb:
            logger.error("无法确定连接类型，无法重启游戏")
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
                logger.info("关闭游戏进程成功")
            else:
                # 对于ADB端，关闭游戏进程
                adb_path = getattr(self.connector, 'adb_path', 'adb')
                # 假设游戏包名为com.hypergryph.arknights
                subprocess.run(f"{adb_path} -s {self.connector.device_serial} shell am force-stop com.hypergryph.arknights", shell=True)
                logger.info("关闭游戏进程成功")
        except Exception as e:
            logger.error(f"关闭游戏进程失败: {e}")
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
                                    logger.info(f"从注册表获取游戏路径并启动: {game_path}")
                                    break
                        except:
                            pass
                    else:
                        # 如果从注册表获取失败，使用默认路径
                        game_path = Path("C:\\Program Files\\Arknights\\Arknights.exe")
                        if game_path.exists():
                            subprocess.Popen(str(game_path))
                            logger.info(f"使用默认路径启动游戏: {game_path}")
                        else:
                            logger.error("无法找到游戏可执行文件")
                            return False
                except:
                    # 如果注册表操作失败，使用默认路径
                    game_path = Path("C:\\Program Files\\Arknights\\Arknights.exe")
                    if game_path.exists():
                        subprocess.Popen(str(game_path))
                        logger.info(f"使用默认路径启动游戏: {game_path}")
                    else:
                        logger.error("无法找到游戏可执行文件")
                        return False
            else:
                # 对于ADB端，重新启动游戏
                adb_path = getattr(self.connector, 'adb_path', 'adb')
                # 启动游戏
                subprocess.run(f"{adb_path} -s {self.connector.device_serial} shell am start -n com.hypergryph.arknights/com.u8.sdk.U8UnityContext")
                logger.info("重新启动游戏成功")
        except Exception as e:
            logger.error(f"重新启动游戏失败: {e}")
            return False
        
        # 等待游戏启动
        time.sleep(10)
        
        # 重新连接
        try:
            self.connector.connect()
            if self.connector.is_connected:
                logger.info("游戏重启后重新连接成功")
            else:
                logger.warning("游戏重启后重新连接失败")
        except Exception as e:
            logger.error(f"重新连接失败: {e}")
        
        return True

    def auto_login(self):
        """自动登录功能，处理服务器维护或掉线后的重新登录"""
        logger.info("开始自动登录流程")
        
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
        
        # 等待游戏启动完全
        logger.info("等待游戏启动完全")
        time.sleep(5)
        
        # 1. 点击屏幕中心跳过中转页面
        logger.info("点击屏幕中心跳过中转页面")
        self.connector.click((0.5, 0.5))
        time.sleep(2)
        
        # 2. 寻找并点击登录按钮
        logger.info("寻找登录按钮")
        login_button_found = False
        for i in range(5):
            # 尝试获取截图，最多尝试3次
            screenshot = None
            for j in range(3):
                screenshot = self.connector.capture_screenshot()
                if screenshot is not None:
                    break
                time.sleep(1)
            
            if screenshot is None:
                logger.warning(f"获取截图失败，跳过本次尝试")
                time.sleep(1)
                continue
            
            # 显示截图尺寸，用于调试
            h, w = screenshot.shape[:2]
            logger.info(f"截图尺寸: {w}x{h}")
            
            matched, pos = self.match_template(screenshot, "login_button", threshold=0.9)
            if matched:
                # 计算相对坐标
                rel_x = pos[0] / w
                rel_y = pos[1] / h
                logger.info(f"登录按钮位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                self.connector.click((rel_x, rel_y))
                logger.info("点击登录按钮")
                time.sleep(3)
                login_button_found = True
                break
            time.sleep(1)
        
        if not login_button_found:
            logger.error("未找到登录按钮，登录流程中断")
            return False
        
        # 3. 等待登录完成
        logger.info("等待登录完成")
        time.sleep(10)  # 等待10秒
        
        # 4. 寻找争锋频道页面
        logger.info("寻找争锋频道页面")
        
        # 优先检测competition_page
        for i in range(3):
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                matched, pos = self.match_template(screenshot, "competition_page")
                if matched:
                    # 计算相对坐标
                    h, w = screenshot.shape[:2]
                    rel_x = pos[0] / w
                    rel_y = pos[1] / h
                    logger.info(f"争锋频道页面位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                    self.connector.click((rel_x, rel_y))
                    logger.info("点击进入争锋频道页面")
                    time.sleep(5)  # 增加等待时间，确保页面完全加载
                    logger.info("自动登录流程完成")
                    return True
            time.sleep(2)
        
        # 如果检测不到competition_page，检测关闭按钮
        close_buttons = ["announcement_close", "announcement_close2", "event_claim_close"]
        for button in close_buttons:
            for i in range(5):  # 每个关闭按钮检测五次
                screenshot = self.connector.capture_screenshot()
                if screenshot is not None:
                    matched, pos = self.match_template(screenshot, button, threshold=0.9)
                    if matched:
                        # 计算相对坐标
                        h, w = screenshot.shape[:2]
                        rel_x = pos[0] / w
                        rel_y = pos[1] / h
                        logger.info(f"{button}位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                        self.connector.click((rel_x, rel_y))
                        logger.info(f"关闭{button}页面")
                        time.sleep(2)  # 增加等待时间，确保页面完全关闭
                        break
                time.sleep(1)
        
        # 检测完关闭按钮后，再次检测competition_page
        for i in range(3):
            screenshot = self.connector.capture_screenshot()
            if screenshot is not None:
                matched, pos = self.match_template(screenshot, "competition_page")
                if matched:
                    # 计算相对坐标
                    h, w = screenshot.shape[:2]
                    rel_x = pos[0] / w
                    rel_y = pos[1] / h
                    logger.info(f"争锋频道页面位置: ({pos[0]}, {pos[1]}), 相对坐标: ({rel_x:.2f}, {rel_y:.2f})")
                    self.connector.click((rel_x, rel_y))
                    logger.info("点击进入争锋频道页面")
                    time.sleep(5)  # 增加等待时间，确保页面完全加载
                    logger.info("自动登录流程完成")
                    return True
            time.sleep(2)
        
        logger.error("未找到争锋频道页面，登录流程失败")
        return False