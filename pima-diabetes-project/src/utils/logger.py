from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def _get_project_root() -> Path:
    """
    Suy ra project root theo cấu trúc:
    pima-diabetes-project/
      └─ src/
          └─ utils/
              └─ logger.py
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _get_project_root()
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # tự tạo nếu chưa có

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Trả về logger ghi ra CẢ console lẫn file logs/app.log.

    - Đảm bảo chỉ cấu hình handler 1 lần (tránh log bị nhân đôi).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Đã cấu hình rồi → dùng lại
        return logger

    logger.setLevel(logging.INFO)

    log_file = LOG_DIR / "app.log"  # bạn có thể đổi thành data_preprocessor.log

    fmt = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    # Ghi ra file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Ghi ra console (tuỳ, có thể bỏ nếu không cần)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Không propagate lên root (tránh log trùng)
    logger.propagate = False

    return logger
