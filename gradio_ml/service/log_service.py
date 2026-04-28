from __future__ import annotations

import logging
import re
from datetime import datetime

from gradio_ml.infrastructure.log_buffer import LogBuffer
from gradio_ml.infrastructure.models import LogEntry, LogFilter, LogLevel

logger = logging.getLogger(__name__)


class LogService:
    def __init__(self, log_buffer: LogBuffer | None = None, sanitize_patterns: dict[str, str] | None = None):
        self._buffer = log_buffer or LogBuffer()
        self._sanitize_patterns: dict[str, re.Pattern] = {}
        if sanitize_patterns:
            for name, pattern in sanitize_patterns.items():
                try:
                    self._sanitize_patterns[name] = re.compile(pattern)
                except re.error as e:
                    logger.warning(f"无效的脱敏正则 '{name}': {e}")

    def append(self, entry: LogEntry) -> bool:
        sanitized_msg = self.sanitize(entry.message)
        sanitized_entry = LogEntry(
            level=entry.level,
            message=sanitized_msg,
            timestamp=entry.timestamp,
            source=entry.source,
            line_number=entry.line_number,
        )
        return self._buffer.append(sanitized_entry)

    def get_logs(self, log_filter: LogFilter | None = None) -> str:
        entries = self._buffer.get_entries(log_filter)
        lines: list[str] = []
        for entry in entries:
            lines.append(f"[{entry.timestamp.strftime('%H:%M:%S')}] [{entry.level.value}] {entry.message}")
        return "\n".join(lines)

    def search(self, keyword: str) -> list[int]:
        return self._buffer.search(keyword)

    def sanitize(self, message: str) -> str:
        result = message
        for name, pattern in self._sanitize_patterns.items():
            result = pattern.sub("***", result)
        return result

    def clear(self) -> None:
        self._buffer.clear()


class GradioLogHandler(logging.Handler):
    def __init__(self, log_service: LogService, level: int = logging.NOTSET):
        super().__init__(level)
        self._log_service = log_service

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level_map = {
                logging.DEBUG: LogLevel.Debug,
                logging.INFO: LogLevel.Info,
                logging.WARNING: LogLevel.Warning,
                logging.ERROR: LogLevel.Error,
                logging.CRITICAL: LogLevel.Critical,
            }
            log_level = level_map.get(record.levelno, LogLevel.Info)
            entry = LogEntry(
                level=log_level,
                message=msg,
                timestamp=datetime.fromtimestamp(record.created),
                source=record.name,
                line_number=record.lineno,
            )
            self._log_service.append(entry)
        except Exception:
            self.handleError(record)
