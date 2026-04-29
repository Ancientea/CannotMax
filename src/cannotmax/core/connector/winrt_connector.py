"""
winrt_capture.py
-----------------
Windows 10+ 上基于 **Windows.Graphics.Capture** 的屏幕捕获轻量封装。

功能特性
~~~~~~~~
- 支持通过 **window_name**（窗口标题）或 **monitor_index**（显示器索引，从 1 开始）二选一指定捕获目标；
- 以事件回调方式缓存 **最新一帧** 图像（BGR，`numpy.ndarray`），并提供线程安全的 `snapshot()`；
- 若启动失败（常见于窗口不存在），自动回退到 **主屏(1) 整屏捕获**；
- 运行时可调用 `recreate()` **切换捕获目标**（重新创建底层会话）；
- 通过锁保护 start/stop/recreate 的并发安全。
"""
from __future__ import annotations
import threading
import time
import logging
from typing import Optional

import ctypes
from ctypes import wintypes
import numpy as np
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

from PyQt6.QtWidgets import (QVBoxLayout,QListWidget,QLineEdit,QDialog)

logger = logging.getLogger(__name__)

class WinRTScreenCapture:
    """Windows.Graphics.Capture 适配器。

    参数
    ----
    window_name: Optional[str]
        要锁定的窗口标题；若提供，则忽略 `monitor_index`。
    monitor_index: Optional[int]
        显示器索引（从 1 开始）；当 `window_name` 为空时生效。
    capture_cursor: bool
        是否捕获鼠标指针。
    draw_border: Optional[bool]
        是否绘制边框（底层能力，`None` 表示使用默认）。
    minimum_update_interval_ms: Optional[int]
        帧间最小更新间隔（毫秒）。
    """

    def __init__(
        self,
        window_name: Optional[str] = None,
        monitor_index: Optional[int] = 1,
        capture_cursor: bool = False,
        draw_border: Optional[bool] = None,
        minimum_update_interval_ms: Optional[int] = None,
    ) -> None:
        # 若指定了窗口标题，则不使用显示器索引
        if window_name:
            monitor_index = None

        # 最新图像帧（BGR 格式）；通过 _lock 保护
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # 控制状态
        self._started = False
        self._control: Optional[InternalCaptureControl] = None  # 保存 CaptureControl，用于停止
        # 保护 start/stop/recreate 的可重入锁
        self._ctl_lock = threading.RLock()

        # 记录初始化参数，便于失败回退与后续重建
        self._init_kwargs = dict(
            cursor_capture=capture_cursor,
            draw_border=draw_border,
            secondary_window=None,
            minimum_update_interval=minimum_update_interval_ms,
            dirty_region=None,
            monitor_index=monitor_index,
            window_name=window_name if window_name else None,
        )

        # 创建底层捕获对象并挂接事件
        self._cap = WindowsCapture(**self._init_kwargs)
        self._attach_events()

    # ------------------------------------------------------------------
    # 事件回调：到帧/关闭
    # ------------------------------------------------------------------
    def _attach_events(self) -> None:
        """绑定底层 WindowsCapture 的事件回调。"""

        @self._cap.event
        def on_frame_arrived(frame: Frame, control: InternalCaptureControl):
            # 转成 BGR 并缓存；复制一份以避免底层复用缓冲导致的数据竞争
            bgr = frame.convert_to_bgr().frame_buffer
            with self._lock:
                self._latest = bgr.copy()

        @self._cap.event
        def on_closed():
            # 资源被关闭时，重置状态
            with self._lock:
                self._started = False
                self._latest = None

    # ------------------------------------------------------------------
    # 控制：启动/停止/重建
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动捕获。若失败，将尝试回退到主屏(1)整屏捕获。"""
        if self._started:
            return
        try:
            self._control = self._cap.start_free_threaded()
            self._started = True
        except Exception as e:
            # 启动失败：回退到主屏整屏捕获
            logger.warning(f"找不到窗口，自动回退整屏捕获（主屏 1）：{e}")
            try:
                self._cap = WindowsCapture(
                    cursor_capture=self._init_kwargs["cursor_capture"],
                    draw_border=self._init_kwargs["draw_border"],
                    secondary_window=None,
                    minimum_update_interval=self._init_kwargs["minimum_update_interval"],
                    dirty_region=None,
                    monitor_index=1,
                    window_name=None,
                )
                self._attach_events()
                self._control = self._cap.start_free_threaded()
                self._started = True
            except Exception as e2:
                raise e2

    def stop(self) -> None:
        """停止捕获并释放底层线程与资源。"""
        with self._ctl_lock:
            try:
                if self._control is not None:
                    self._control.stop()
                    # 给底层线程少许退出时间，降低并发风险
                    time.sleep(0.05)
            except Exception:
                pass
            finally:
                self._control = None
                self._started = False
                with self._lock:
                    self._latest = None

    def recreate(self, window_name: Optional[str] = None, monitor_index: Optional[int] = None) -> None:
        """运行时切换捕获目标并立即启动。

        同步步骤：stop → 以新参数创建 → 重新绑定事件 → 启动。
        """
        with self._ctl_lock:
            self.stop()
            kwargs = dict(
                cursor_capture=self._init_kwargs["cursor_capture"],
                draw_border=self._init_kwargs["draw_border"],
                secondary_window=None,
                minimum_update_interval=self._init_kwargs["minimum_update_interval"],
                dirty_region=None,
                monitor_index=monitor_index if (window_name is None) else None,
                window_name=window_name,
            )
            # 保存以便下次重建
            self._init_kwargs.update(kwargs)

            # 新建实例并启动
            self._cap = WindowsCapture(**kwargs)
            self._attach_events()
            self._control = self._cap.start_free_threaded()
            self._started = True

    # ------------------------------------------------------------------
    # 数据访问：等待首帧 / 截图
    # ------------------------------------------------------------------
    def wait_first_frame(self, timeout_sec: float = 3.0) -> bool:
        """阻塞等待首帧到达。

        参数
        ----
        timeout_sec: float
            超时时长（秒）。

        返回
        ----
        bool
            在超时前是否收到首帧。
        """
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.01)
        return False

    def snapshot(self, bbox: Optional[tuple[int, int, int, int]] = None) -> np.ndarray:
        """返回最近一帧的 **BGR** 拷贝图像，可选裁剪到指定 bbox。

        若尚未有首帧，将抛出 `RuntimeError`。
        """
        with self._lock:
            if self._latest is None:
                raise RuntimeError("WinRT capture 尚未产生首帧")
            
            frame = self._latest.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                H, W = frame.shape[:2]
                x1 = max(0, min(int(x1), W - 1))
                x2 = max(0, min(int(x2), W))
                y1 = max(0, min(int(y1), H - 1))
                y2 = max(0, min(int(y2), H))
                return frame[y1:y2, x1:x2]
            return frame

    def snapshot_once(self, bbox: Optional[tuple[int, int, int, int]] = None, timeout_sec: float = 2.0) -> np.ndarray:
        """返回最近一帧的 **BGR** 拷贝图像，可选裁剪到指定 bbox。

        若当前未启动，则会临时启动捕获，获取一帧后立即停止。
        """
        with self._ctl_lock:
            was_started = self._started
            if not was_started:
                self.start()
            
            frame = None
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                with self._lock:
                    if self._latest is not None:
                        frame = self._latest.copy()
                        break
                time.sleep(0.01)
            
            if frame is None: # 超时
                if not was_started:
                    self.stop()
                raise RuntimeError("WinRT capture 尚未产生首帧")

            if bbox:
                x1, y1, x2, y2 = bbox
                H, W = frame.shape[:2]
                x1 = max(0, min(int(x1), W - 1))
                x2 = max(0, min(int(x2), W))
                y1 = max(0, min(int(y1), H - 1))
                y2 = max(0, min(int(y2), H))
                frame = frame[y1:y2, x1:x2]

            if not was_started:
                self.stop()
            return frame

    def set_capture_target(self, window_name: Optional[str] = None, monitor_index: Optional[int] = None) -> None:
        """
        设置 WinRT 截屏目标（窗口标题或整屏），并启动捕获，等待首帧。
        若初始化失败，将抛出异常。
        """
        with self._ctl_lock:
            if self._started:
                # 如果已经启动，则重建底层会话
                self.recreate(window_name=window_name, monitor_index=monitor_index)
            else:
                # 否则，更新初始化参数并启动
                self._init_kwargs.update(
                    monitor_index=monitor_index if (window_name is None) else None,
                    window_name=window_name,
                )
                self._cap = WindowsCapture(**self._init_kwargs)
                self._attach_events()
                self.start()

            # 启动并等待首帧
            if not self.wait_first_frame(timeout_sec=2.0):
                # 首帧未就绪，清理并抛出异常
                self.stop()
                raise RuntimeError("WinRT capture 尚未产生首帧")

def list_visible_window_titles() -> list[str]:
    """列出所有**可见**窗口的标题（去重并按字典序排序）。"""
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW

    titles: list[str] = []

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
    titles = sorted(set(titles))
    logger.info(f"窗口列表：{titles}")
    return titles

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
