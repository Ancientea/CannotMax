from __future__ import annotations

import queue
import threading

from gradio_ml.infrastructure.models import LogEntry, LogLevel, ProgressUpdate, RawMetricCallback


def create_metric_queue() -> queue.Queue[RawMetricCallback]:
    return queue.Queue()


def create_log_queue() -> queue.Queue[LogEntry]:
    return queue.Queue()


def create_progress_queue() -> queue.Queue[ProgressUpdate]:
    return queue.Queue()


METRIC_QUEUE: queue.Queue[RawMetricCallback] = create_metric_queue()
LOG_QUEUE: queue.Queue[LogEntry] = create_log_queue()
PROGRESS_QUEUE: queue.Queue[ProgressUpdate] = create_progress_queue()

pause_event: threading.Event = threading.Event()
stop_event: threading.Event = threading.Event()


def reset_events() -> None:
    pause_event.clear()
    stop_event.clear()


def clear_queues() -> None:
    while not METRIC_QUEUE.empty():
        try:
            METRIC_QUEUE.get_nowait()
        except queue.Empty:
            break
    while not LOG_QUEUE.empty():
        try:
            LOG_QUEUE.get_nowait()
        except queue.Empty:
            break
    while not PROGRESS_QUEUE.empty():
        try:
            PROGRESS_QUEUE.get_nowait()
        except queue.Empty:
            break
