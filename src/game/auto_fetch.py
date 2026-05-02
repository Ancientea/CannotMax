import os

# 设置 OpenCV 日志级别为 ERROR，减少 libpng 警告
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import csv
import datetime
from enum import Enum, auto
import logging
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from src.recognition.recognize import INTELLIGENT_WORKERS_DEBUG
from src.core.config import MONSTER_COUNT, FIELD_FEATURE_COUNT
from src.core.paths import PROJECT_ROOT, process_image_path, simulation_path
from collections.abc import Callable
from collections import deque
from .login import LoginManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GameState(Enum):
    MAIN_MENU = auto()
    MODE_SELECTION_UNSELECTED = auto()
    MODE_SELECTION_SELECTED = auto()
    PRE_BATTLE = auto()
    IN_BATTLE = auto()
    SETTLEMENT = auto()
    FINISHED = auto()
    UNKNOWN = auto()


class AutoFetch:
    def __init__(
        self,
        connector,
        game_mode,
        is_invest,
        update_prediction_callback: Callable[[float], None],
        update_monster_callback: Callable[[list], None],
        updater: Callable[[], None],
        start_callback: Callable[[], None],
        stop_callback: Callable[[], None],
        training_duration,
        recognizer=None,
        cannot_model=None,
        field_recognizer=None,
        start_timestamp=None,
    ):
        self.connector = connector
        self.game_mode = game_mode  # 游戏模式（30人或自娱自乐）
        self.is_invest = is_invest  # 是否投资
        self.current_prediction = 0.5  # 当前预测结果，初始值为0.5
        self.recognize_results = []  # 识别结果列表
        self.field_recognizer = field_recognizer  # 场地识别实例
        self.field_recognize_result = {}  # 场地识别结果
        self.incorrect_fill_count = 0  # 填写错误次数
        self.total_fill_count = 0  # 总填写次数
        self.update_prediction_callback = update_prediction_callback
        self.update_monster_callback = update_monster_callback
        self.updater = updater  # 更新统计信息的函数
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.monster_image = None  # 当前轮次怪物图片
        self.auto_fetch_running = False  # 自动获取数据的状态
        self.auto_fetch_thread = None  # 线程引用
        self.start_time = (
            start_timestamp if start_timestamp is not None else time.time()
        )  # 使用预先确定的时间戳
        self.training_duration = training_duration  # 训练时长
        self.data_folder = PROJECT_ROOT / "data"  # 数据文件夹路径
        self.image_buffer = deque(
            maxlen=5
        )  # 图片缓存队列，设置队列长短来保存结算前的图片
        self.recognizer = recognizer  # 使用传入的识别器
        self.cannot_model = cannot_model  # 使用传入的模型
        self.last_state = GameState.UNKNOWN
        self.login_manager = LoginManager(connector)
        self.state_start_time = time.time()  # 记录当前状态的开始时间

        # 初始化状态匹配模板，缩小匹配尺寸提高速度
        self.MATCH_WIDTH = 1920 // 4
        self.MATCH_HEIGHT = 1080 // 4 // 4

        # 初始化模板
        self.processed_template = []
        self._init_templates()

        # 根据 FIELD_FEATURE_COUNT 决定是否启用场地识别器（使用传入的实例）
        if FIELD_FEATURE_COUNT > 0:
            if self.field_recognizer is not None:
                logger.info(f"场地识别已启用，特征数量: {FIELD_FEATURE_COUNT}")
            else:
                logger.warning(
                    f"FIELD_FEATURE_COUNT={FIELD_FEATURE_COUNT} > 0 但未传入 field_recognizer，场地识别将被禁用"
                )
                self.field_recognizer = None
        else:
            self.field_recognizer = None
            logger.info("场地识别已禁用，仅收集怪物数据")

    def _log(self, level, message):
        """生成带有设备序列号的日志消息"""
        serial = getattr(self.connector, "device_serial", None)
        if serial:
            logger.log(level, f"[{serial}] {message}")
        else:
            logger.log(level, message)

    def _init_templates(self):
        for i in range(16):
            img = cv2.imread(str(process_image_path(i)))
            if img is not None:
                # 使用最近邻插值缩放模板，速度最快
                img_resized = cv2.resize(
                    img,
                    (self.MATCH_WIDTH, self.MATCH_HEIGHT * 4),
                    interpolation=cv2.INTER_NEAREST,
                )
                img_quarter = img_resized[self.MATCH_HEIGHT * 3 :, :]
                self.processed_template.append(img_quarter)
            else:
                self.processed_template.append(None)

    def match_images(self, screenshot):
        h, w = screenshot.shape[:2]
        # 裁剪底部 1/4 ROI
        y_start = int(h * 3 / 4)
        screenshot_quarter = screenshot[y_start:, :]
        screenshot_quarter = cv2.resize(
            screenshot_quarter,
            (self.MATCH_WIDTH, self.MATCH_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        results = []
        for idx, template in enumerate(self.processed_template):
            if template is None:
                continue
            res = cv2.matchTemplate(
                screenshot_quarter, template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(res)
            results.append((idx, max_val))
        return results

    def fill_data(
        self,
        battle_result,
        recognize_results,
        monster_image,
        result_image,
        field_recognize_result,
    ):
        # 获取队列头的图片
        if self.image_buffer:
            _, previous_image, _ = self.image_buffer[0]  # 获取队列头的图片
        else:
            logger.error("图片缓存队列为空，无法获取图片")
            previous_image = None

        if previous_image is None:
            logger.error("未找到2秒前的图片，无法保存")
            return

        image_name = self.get_image_name(
            recognize_results, battle_result
        )  # 生成图片名称

        # 确保images文件夹存在
        images_folder = self.data_folder / "images"
        try:
            images_folder.mkdir(parents=True, exist_ok=True)
            # logger.info(f"确保images文件夹存在: {images_folder}")
        except Exception as e:
            logger.error(f"创建images文件夹失败: {e}")

        if (
            INTELLIGENT_WORKERS_DEBUG
        ):  # 如果处于debug模式，保存人工审核图片到本地
            if monster_image is not None:
                try:
                    resized_monster_img = cv2.resize(
                        monster_image, (960, 540)
                    )  # 调整分辨率为 960x540
                    image_path = images_folder / (image_name + ".jpg")
                    cv2.imwrite(
                        image_path,
                        resized_monster_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    # logger.info(f"保存怪物图片到 {image_path}")
                except Exception as e:
                    logger.error(f"保存怪物图片失败: {e}")

            # 新增保存结果图片逻辑
            if image_name:
                try:
                    result_image_name = image_name + "_result.jpg"
                    # 缩放到128像素高度
                    h, w = result_image.shape[:2]
                    new_height = 128
                    resized_image = cv2.resize(
                        result_image, (int(w * (new_height / h)), new_height)
                    )
                    image_path = images_folder / result_image_name
                    cv2.imwrite(image_path, resized_image)
                    logger.info(f"保存结果图片到 {image_path}")
                except Exception as e:
                    logger.error(f"保存结果图片失败: {e}")

        # 原始怪物数据
        left_monster_data = np.zeros(MONSTER_COUNT)
        right_monster_data = np.zeros(MONSTER_COUNT)

        for res in recognize_results:
            region_id = res["region_id"]
            if "error" not in res:
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:  # 左侧怪物
                        left_monster_data[matched_id - 1] = number
                    else:  # 右侧怪物
                        right_monster_data[matched_id - 1] = number
            else:
                logger.error(f"存在错误，本次不填写")
                return

        # 组织数据格式
        data_row = []
        if self.field_recognizer is not None:
            # 准备场地特征数据
            field_feature_columns = self.field_recognizer.get_feature_columns()
            field_data_values = []
            for col in field_feature_columns:
                if col in field_recognize_result:
                    field_data_values.append(field_recognize_result[col])
                else:
                    field_data_values.append(0)  # 默认值

            # 记录场地特征到日志
            field_summary = []
            for i, col in enumerate(field_feature_columns):
                value = field_data_values[i]
                field_summary.append(f"{col}={value}")
            logger.info(f"当次场地特征: {', '.join(field_summary)}")

            # 按照data_cleaning_with_field_recognize_gpu.py的格式组织数据
            data_row.extend(left_monster_data.tolist())  # 1L-77L
            data_row.extend(field_data_values)  # 78L-83L (场地特征L)
            data_row.extend(right_monster_data.tolist())  # 1R-77R
            data_row.extend(field_data_values)  # 78R-83R (场地特征R，复制)
        else:
            # 仅收集怪物数据的格式
            logger.info("仅收集怪物数据，跳过场地特征")
            data_row.extend(left_monster_data.tolist())  # 左侧怪物数据
            data_row.extend(right_monster_data.tolist())  # 右侧怪物数据

        data_row.append(battle_result)  # Result

        # 替换所有NaN为-1
        for i, x in enumerate(data_row):
            if isinstance(x, (int, float)) and np.isnan(x):
                data_row[i] = -1

        # 保存数据
        start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(
            r"%Y_%m_%d__%H_%M_%S"
        )

        if (
            INTELLIGENT_WORKERS_DEBUG
        ):  # 如果处于debug模式，保存人工审核图片到本地
            data_row.append(image_name)

        with open(self.data_folder / "arknights.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_row)
        logger.info(f"写入csv完成")

    def build_terrain_features(self, left_counts, right_counts):
        """构建包含地形的完整特征向量"""
        # 获取场地特征列数
        field_feature_columns = self.field_recognizer.get_feature_columns()
        num_field_features = len(field_feature_columns)

        # 构建地形特征向量（基于当前场地识别结果）
        terrain_features = np.zeros(num_field_features)

        if self.field_recognize_result:
            # 将场地识别结果转换为特征向量
            for i, col in enumerate(field_feature_columns):
                if col in self.field_recognize_result:
                    terrain_features[i] = self.field_recognize_result[col]

        # 按照data_cleaning_with_field_recognize.py的格式组织数据
        full_features = np.concatenate(
            [
                left_counts,  # 1L-77L
                terrain_features,  # 78L-83L
                right_counts,  # 1R-77R
                terrain_features,  # 78R-83R
            ]
        )

        return full_features

    @staticmethod
    def calculate_average_yellow(image):
        def get_saturation(bgr):
            # 将BGR转换为0-1范围后计算饱和度
            b, g, r = [x / 255.0 for x in bgr]
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            return (
                (delta / cmax) * 255 if cmax != 0 else 0
            )  # 返回0-255范围的饱和度值

        if image is None:
            logger.error("图像加载失败")
            return None

        height, width, _ = image.shape

        # 增加多个采样点，提高识别稳定性
        sample_points = [
            (0.1, 0.1),  # 左上区域
            (0.1, 0.5),  # 左中区域
            (0.9, 0.1),  # 右上区域
            (0.9, 0.5),  # 右中区域
        ]

        sample_size = 20  # 增大采样区域
        left_saturations = []
        right_saturations = []

        for y_ratio, x_ratio in sample_points:
            y_offset = int(height * y_ratio)

            # 左侧采样点
            x_left_offset = int(width * 0.1)
            y_end = min(y_offset + sample_size, height)
            x_left_end = min(x_left_offset + sample_size, width)
            if y_end > y_offset and x_left_end > x_left_offset:
                left_region = image[y_offset:y_end, x_left_offset:x_left_end]
                left_mean = left_region.mean(axis=(0, 1))
                left_saturations.append(get_saturation(left_mean))

            # 右侧采样点
            x_right_offset = int(width * 0.9 - sample_size)
            x_right_end = min(x_right_offset + sample_size, width)
            if y_end > y_offset and x_right_end > x_right_offset:
                right_region = image[
                    y_offset:y_end, x_right_offset:x_right_end
                ]
                right_mean = right_region.mean(axis=(0, 1))
                right_saturations.append(get_saturation(right_mean))

        if not left_saturations or not right_saturations:
            logger.error("无法获取有效的采样区域")
            return None

        # 计算平均饱和度
        avg_sat_left = sum(left_saturations) / len(left_saturations)
        avg_sat_right = sum(right_saturations) / len(right_saturations)

        # 计算饱和度差值
        saturation_diff = avg_sat_left - avg_sat_right

        # 使用自适应阈值，根据整体饱和度水平调整
        base_threshold = 15
        # 如果整体饱和度较低，降低阈值
        avg_overall_sat = (avg_sat_left + avg_sat_right) / 2
        if avg_overall_sat < 50:
            threshold = 10
        elif avg_overall_sat < 100:
            threshold = 12
        else:
            threshold = base_threshold

        # 检查差值是否符合要求
        if abs(saturation_diff) <= threshold:
            logger.warning(
                f"饱和度差值不足 (左:{avg_sat_left:.1f} vs 右:{avg_sat_right:.1f}, 阈值:{threshold})"
            )
            # 尝试使用亮度差异作为备选方案
            return AutoFetch.calculate_brightness_diff(image)

        # 返回左侧是否比右侧饱和度更高
        return saturation_diff > 0

    @staticmethod
    def calculate_brightness_diff(image):
        """使用亮度差异作为胜负识别的备选方案"""
        if image is None:
            return None

        height, width, _ = image.shape

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 定义左侧和右侧区域
        left_region = gray[:, : width // 2]
        right_region = gray[:, width // 2 :]

        # 计算平均亮度
        left_brightness = left_region.mean()
        right_brightness = right_region.mean()

        # 计算亮度差值
        brightness_diff = left_brightness - right_brightness

        # 使用亮度阈值
        brightness_threshold = 10
        if abs(brightness_diff) <= brightness_threshold:
            logger.warning(
                f"亮度差值不足 (左:{left_brightness:.1f} vs 右:{right_brightness:.1f})"
            )
            return None

        # 返回左侧是否比右侧亮
        return brightness_diff > 0

    @staticmethod
    def get_image_name(recognize_results, battle_result=None):
        # 处理结果
        processed_monsters = []  # 用于存储处理的怪物 IDx数量
        for res in recognize_results:
            if "error" not in res:
                matched_id = res["matched_id"]
                if matched_id != 0:
                    number = res.get("number", 1)
                    processed_monsters.append(f"{matched_id}x{number}")
        # 生成唯一的文件名（使用日期时间字符串）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 将处理的怪物信息拼接到文件名中，格式为 IDx数量
        monsters_str = "_".join(processed_monsters)
        image_name = f"{timestamp}_{monsters_str}_{battle_result}"
        return image_name

    def save_statistics_to_log(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.total_fill_count}\n"
            f"填写×次数: {self.incorrect_fill_count}\n"
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n"
        )
        with open("log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(stats_text)

    def recognize_and_predict(self, screenshot=None):
        if screenshot is None:
            screenshot = self.connector.capture_screenshot()
        self.recognize_results = self.recognizer.process_regions(screenshot)

        # 场地识别
        if self.field_recognizer is not None:
            self.field_recognize_result = (
                self.field_recognizer.recognize_field_elements(screenshot)
            )

            # 输出场地识别结果日志
            if self.field_recognize_result:
                detected_elements = [
                    key
                    for key, value in self.field_recognize_result.items()
                    if value == 1
                ]
                partial_detected = [
                    key
                    for key, value in self.field_recognize_result.items()
                    if value == -1
                ]
                if detected_elements:
                    logger.info(
                        f"场地识别检测到元素: {', '.join(detected_elements)}"
                    )
                if partial_detected:
                    logger.info(
                        f"场地识别部分检测到元素: {', '.join(partial_detected)}"
                    )
                if not detected_elements and not partial_detected:
                    logger.info("场地识别: 未检测到任何特殊元素")
            else:
                logger.info("场地识别: 识别结果为空")
        else:
            # 场地识别被禁用，设置为空结果
            self.field_recognize_result = {}
            logger.debug("场地识别已禁用，跳过场地识别")

        # 获取预测结果
        self.update_monster_callback(self.recognize_results)
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        for res in self.recognize_results:
            if "error" not in res:
                region_id = res["region_id"]
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id == 0:
                    continue
                if region_id < 3:
                    left_counts[matched_id - 1] = number
                else:
                    right_counts[matched_id - 1] = number
            else:
                logger.error("识别结果有错误，本轮跳过")
        # 选择预测方法
        if self.cannot_model.is_model_loaded:
            if self.field_recognizer is not None:
                # 构建包含地形的完整特征向量
                full_features = self.build_terrain_features(
                    left_counts, right_counts
                )
                self.current_prediction = (
                    self.cannot_model.get_prediction_with_terrain(
                        full_features
                    )
                )
            else:
                # 仅使用怪物数据进行预测
                self.current_prediction = self.cannot_model.get_prediction(
                    left_counts, right_counts
                )
            self.update_prediction_callback(self.current_prediction)
        else:
            logger.warning("模型未加载，无法进行预测")

        # 人工审核保存测试用截图
        if INTELLIGENT_WORKERS_DEBUG:  # 如果处于debug模式且处于自动模式
            self.monster_image = screenshot

    def battle_result(self, result_image):
        result = self.calculate_average_yellow(result_image)
        if result is None:
            logger.warning("战斗结果识别失败，需要重试")
            return False

        if result:
            self.fill_data(
                "L",
                self.recognize_results,
                self.monster_image,
                result_image,
                self.field_recognize_result,
            )
            if self.current_prediction > 0.5:
                self.incorrect_fill_count += 1
            self._log(logging.INFO, "填写数据左赢")
        else:
            self.fill_data(
                "R",
                self.recognize_results,
                self.monster_image,
                result_image,
                self.field_recognize_result,
            )
            if self.current_prediction < 0.5:
                self.incorrect_fill_count += 1
            self._log(logging.INFO, "填写数据右赢")
        self.total_fill_count += 1
        self.updater()
        self._log(logging.INFO, "下一轮")
        return True

    def auto_fetch_data(self):
        relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        timea = time.time()
        screenshot = self.connector.capture_screenshot()
        if screenshot is None:
            self._log(logging.ERROR, "截图失败，尝试自动登录")

            # 使用 LoginManager 的自动登录（带重启重试）
            if not self.login_manager.auto_login_with_restart(
                lambda: self.auto_fetch_running
            ):
                self._log(logging.ERROR, "自动登录失败，无法继续操作")
                return

            # 检查是否已经收到停止信号
            if not self.auto_fetch_running:
                self._log(logging.INFO, "检测到停止信号，取消后续操作")
                return
            self._log(logging.INFO, "自动登录成功，等待页面加载...")
            if not self._sleep_with_check(3):
                return
            self._log(logging.INFO, "重新获取截图")
            screenshot = self.connector.capture_screenshot()
            if screenshot is None:
                self._log(
                    logging.ERROR, "登录后仍然无法获取截图，无法继续操作"
                )
                return

        # 保存当前截图及其信息到缓冲区
        timestamp = int(time.time())
        self.image_buffer.append((timestamp, screenshot.copy(), []))

        # 先进行状态识别
        results = self.match_images(screenshot)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        # logger.debug(f"处理图片总用时：{time.time()-timea:.3f}s")
        # logger.info("匹配结果：", results[0])

        # 状态判断：取匹配度最高的一个
        current_state = GameState.UNKNOWN
        best_idx = -1
        if results:
            best_idx, best_score = results[0]
            if best_score > 0.7:
                if best_idx == 0:
                    current_state = GameState.MAIN_MENU
                elif best_idx == 1:
                    current_state = GameState.MODE_SELECTION_UNSELECTED
                elif best_idx == 2:
                    current_state = GameState.MODE_SELECTION_SELECTED
                elif best_idx in [3, 4, 5, 15]:
                    current_state = GameState.PRE_BATTLE
                elif best_idx in [6, 7, 14]:
                    current_state = GameState.IN_BATTLE
                elif best_idx in [8, 9, 10, 11]:
                    current_state = GameState.SETTLEMENT
                elif best_idx in [12, 13]:
                    current_state = GameState.FINISHED
                if self.last_state != current_state:
                    logger.info(
                        f"匹配到状态: {self.last_state.name} -> {current_state.name}, score:{best_score:.4f}"
                    )
            else:
                # logger.info(f"状态机匹配置信度过低: idx:{best_idx}, score:{best_score:.4f}")
                pass

        # 处理状态发生变化时的逻辑
        if current_state != self.last_state:
            old_state = self.last_state
            self.last_state = current_state
            elapsed = time.time() - self.state_start_time

            # 不记录 PRE_BATTLE -> IN_BATTLE 的状态变化
            if not (
                old_state == GameState.PRE_BATTLE
                and current_state == GameState.IN_BATTLE
            ):
                self._log(
                    logging.INFO,
                    f"游戏状态变化: {old_state.name} -> {current_state.name}, 持续时间: {elapsed:.2f} 秒",
                )

            # 如果成功进入稳定状态且重启计数器非零，重置重启计数器
            _stable_states = {
                GameState.MAIN_MENU,
                GameState.IN_BATTLE,
                GameState.SETTLEMENT,
                GameState.FINISHED,
            }
            if (
                current_state in _stable_states
                and self.login_manager.restart_count > 0
            ):
                self.login_manager.reset_restart_count()

            self.state_start_time = time.time()  # 重置状态开始时间

        # 全局超时检测：无论什么状态，只要超过时间都可能触发重启
        # 非战斗状态：超过 50 秒触发重启
        # 战斗流程状态：超过 120 秒触发重启（防止战斗卡死）
        elapsed_time = time.time() - self.state_start_time
        is_battle_state = self.last_state in [
            GameState.PRE_BATTLE,
            GameState.IN_BATTLE,
            GameState.SETTLEMENT,
            GameState.FINISHED,
        ]
        timeout_threshold = 120.0 if is_battle_state else 50.0

        # 检查是否超时
        if elapsed_time > timeout_threshold:
            state_name = self.last_state.name if self.last_state else "NONE"
            self._log(
                logging.WARNING,
                f"在状态 '{state_name}' 停留超过 {elapsed_time:.2f} 秒（阈值: {timeout_threshold:.0f} 秒），触发重启",
            )

            # 使用 LoginManager 的重启登录方法
            if not self.login_manager.can_restart():
                self._log(
                    logging.ERROR,
                    f"已达到最大重启次数 {self.login_manager.max_restart_count} 次，停止运行",
                )
                self.auto_fetch_running = False
                self.stop_callback()
            elif not self.login_manager.restart_and_login(
                first_start=False,
                stop_callback=lambda: self.auto_fetch_running,
            ):
                self._log(
                    logging.WARNING, "本次重启登录失败，将在下次超时后重试"
                )

            # 检测完毕后，无论结果如何，重置计时器，避免频繁阻塞
            self.state_start_time = time.time()
            self._log(logging.INFO, "重置状态计时器")

        # 状态执行
        match current_state:
            case GameState.MAIN_MENU:
                # 活动主界面状态，点击加入赛事跳转到选择模式界面（未选择）状态
                self.connector.click(relative_points[0])
                self._log(logging.INFO, "加入赛事")
            case GameState.MODE_SELECTION_UNSELECTED:
                # 选择模式界面（未选择），点击模式跳转到已选择
                if self.game_mode == "30人":
                    self.connector.click(relative_points[1])
                    self._log(logging.INFO, "竞猜对决30人")
                    if not self._sleep_with_check(2):
                        return
                    self.connector.click(relative_points[0])
                    self._log(logging.INFO, "开始游戏")
                    time.sleep(1)
                else:
                    self.connector.click(relative_points[2])
                    self._log(logging.INFO, "自娱自乐")
                    time.sleep(1)
            case GameState.MODE_SELECTION_SELECTED:
                # 选择模式界面（已选择），点击开始游戏跳转到怪物数量界面状态
                self.connector.click(relative_points[0])
                self._log(logging.INFO, "开始游戏")
                time.sleep(1)
            case GameState.PRE_BATTLE:
                # 怪物数量界面状态，识别并开始游戏，跳转到等待结算状态
                if not self._sleep_with_check(1):
                    return
                # 识别怪物类型数量和地形
                screenshot = self.connector.capture_screenshot()
                self.recognize_and_predict(screenshot)

                # 点击下一轮
                if self.is_invest:  # 投资
                    # 根据预测结果点击投资左/右
                    if self.current_prediction > 0.5:
                        if best_idx == 4:
                            self.connector.click(relative_points[0])
                        else:
                            self.connector.click(relative_points[2])
                        self._log(logging.INFO, "投资右")
                        if not self._sleep_with_check(3):
                            return
                    else:
                        if best_idx == 4:
                            self.connector.click(relative_points[1])
                        else:
                            self.connector.click(relative_points[3])
                        self._log(logging.INFO, "投资左")
                        if not self._sleep_with_check(3):
                            return
                    if self.game_mode == "30人":
                        self._log(
                            logging.INFO, "30人模式下，投资后需要等待20秒"
                        )
                        if not self._sleep_with_check(5):
                            return
                else:  # 不投资
                    self.connector.click(relative_points[4])
                    self._log(logging.INFO, "本轮观望")
                    if not self._sleep_with_check(3):
                        return
            case GameState.IN_BATTLE:
                # 等待结算状态，战斗中界面，保持状态
                # self._log(logging.INFO, "等待战斗结束")
                pass
            case GameState.SETTLEMENT:
                if not self.battle_result(screenshot):
                    new_screenshot = self.connector.capture_screenshot()
                    if new_screenshot is not None and not self.battle_result(
                        new_screenshot
                    ):
                        self._log(logging.ERROR, "战斗结果识别失败，跳过本轮")
                if not self._sleep_with_check(5):
                    return
            case GameState.FINISHED:
                # 结束状态，所有轮次结束界面，返回主页并跳转到活动主界面状态
                self.connector.click(relative_points[0])
                self._log(logging.INFO, "返回主页")
            case _:
                # 未匹配到有效界面，保持状态
                pass

    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            try:
                # 每次循环开始时检查状态
                if not self.auto_fetch_running:
                    break

                # 刷新当前预测显示（心跳）：不要写入固定0，避免把GUI错误覆盖成“左方100%”
                self.update_prediction_callback(self.current_prediction)

                self.auto_fetch_data()

                # 每次循环结束时检查状态
                if not self.auto_fetch_running:
                    break

                elapsed_time = time.time() - self.start_time
                if (
                    self.training_duration != -1
                    and elapsed_time >= self.training_duration
                ):
                    self._log(logging.INFO, "已达到设定时长，结束自动获取")
                    break
                # 检测一次间隔时间——————————————————————————————————
                time.sleep(0.1)
            except Exception as e:
                self._log(logging.ERROR, f"自动获取数据出错:\n{e}")
                break

        else:
            self._log(
                logging.INFO, "auto_fetch_running is False, exiting loop"
            )
            return
        # 不通过按钮结束自动获取
        self._log(logging.INFO, "break auto_fetch_loop")
        self.stop_auto_fetch()

    def start_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch_running = True
            # 使用初始化时设置的时间戳，不重新获取当前时间
            start_time = datetime.datetime.fromtimestamp(
                self.start_time
            ).strftime(r"%Y_%m_%d__%H_%M_%S")
            self.data_folder = PROJECT_ROOT / "data" / start_time
            self._log(logging.INFO, f"创建文件夹: {self.data_folder}")
            self.data_folder.mkdir(parents=True, exist_ok=True)  # 创建文件夹
            (self.data_folder / "images").mkdir(parents=True, exist_ok=True)
            with open(
                self.data_folder / "arknights.csv", "w", newline=""
            ) as file:
                # 创建CSV表头
                if self.field_recognizer is not None:
                    # 获取场地特征列数
                    num_field_features = len(
                        self.field_recognizer.get_feature_columns()
                    )

                    # 按照data_cleaning_with_field_recognize_gpu.py的格式创建表头
                    header = [
                        f"{i+1}L" for i in range(MONSTER_COUNT)
                    ]  # 1L-77L
                    header += [
                        f"{i+1}LF"
                        for i in range(
                            MONSTER_COUNT, MONSTER_COUNT + num_field_features
                        )
                    ]  # 78LF-83LF (场地特征)
                    header += [
                        f"{i+1}R" for i in range(MONSTER_COUNT)
                    ]  # 1R-77R
                    header += [
                        f"{i+1}RF"
                        for i in range(
                            MONSTER_COUNT, MONSTER_COUNT + num_field_features
                        )
                    ]  # 78RF-83RF (场地特征)
                    header += ["Result", "ImgPath"]
                    logger.info(
                        f"创建包含场地特征的CSV表头，场地特征数: {num_field_features}"
                    )
                else:
                    # 仅怪物数据的格式
                    header = [
                        f"{i+1}L" for i in range(MONSTER_COUNT)
                    ]  # 左侧怪物数据
                    header += [
                        f"{i+1}R" for i in range(MONSTER_COUNT)
                    ]  # 右侧怪物数据
                    header += ["Result", "ImgPath"]
                    logger.info("创建仅包含怪物数据的CSV表头")

                writer = csv.writer(file)
                writer.writerow(header)
            self.log_file_handler = logging.FileHandler(
                self.data_folder / f"AutoFetch_{start_time}.log", "a", "utf-8"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
            )
            self.log_file_handler.setFormatter(file_formatter)
            self.log_file_handler.setLevel(logging.INFO)
            logger.addHandler(self.log_file_handler)

            # 启动自动获取数据线程
            self.auto_fetch_thread = threading.Thread(
                target=self.auto_fetch_loop
            )
            self.auto_fetch_thread.start()
            logger.info("自动获取数据已启动")
            self.start_callback()
        else:
            logger.warning("自动获取数据已在运行中，请勿重复启动。")

    def _sleep_with_check(self, seconds):
        """带停止检查的睡眠，可被中断"""
        start_time = time.time()
        while time.time() - start_time < seconds:
            if not self.auto_fetch_running:
                return False
            time.sleep(0.1)
        return True

    def stop_auto_fetch(self):
        if not self.auto_fetch_running:
            return
        # 强制设置停止标志，不等待线程退出
        self.auto_fetch_running = False
        self._log(logging.INFO, "强制停止自动获取")

        # 不等待线程退出，让线程在下一次循环时自己检测到停止标志
        self.save_statistics_to_log()
        self.stop_callback()
        if hasattr(self, "log_file_handler"):
            logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
