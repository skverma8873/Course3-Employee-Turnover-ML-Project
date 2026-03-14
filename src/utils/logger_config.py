"""
Centralised logging configuration for the Employee Turnover Analytics project.

Usage
-----
Call ``setup_logging()`` once at the start of a script or notebook:

    from src.utils.logger_config import setup_logging
    from pathlib import Path

    logger = setup_logging(log_dir=Path("outputs/logs"))

Every module then obtains its own named logger with:

    import logging
    logger = logging.getLogger(__name__)

Log files are written to ``outputs/logs/employee_turnover_YYYYMMDD.log``
with automatic rotation (10 MB per file, 5 backups retained).
"""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB per log file
BACKUP_COUNT: int = 5               # Keep 5 rotated backups

_logging_configured: bool = False   # Guard against double-configuration


def setup_logging(
    log_dir: Path = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure the root logger with a rotating file handler and a console handler.

    This function is idempotent — calling it multiple times has no additional
    effect after the first successful call.

    Args:
        log_dir: Directory where log files will be written.  Defaults to
            ``outputs/logs/`` relative to the project root (two levels above
            this file).
        console_level: Logging level for the console (stderr) handler.
            Defaults to ``logging.INFO``.
        file_level: Logging level for the rotating file handler.
            Defaults to ``logging.DEBUG``.

    Returns:
        The configured root :class:`logging.Logger` instance.

    Raises:
        OSError: If ``log_dir`` cannot be created or written to.
    """
    global _logging_configured
    if _logging_configured:
        return logging.getLogger()

    # ------------------------------------------------------------------
    # Resolve log directory
    # ------------------------------------------------------------------
    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "outputs" / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build log filename: employee_turnover_YYYYMMDD.log
    # ------------------------------------------------------------------
    date_stamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"employee_turnover_{date_stamp}.log"

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ------------------------------------------------------------------
    # Rotating file handler (DEBUG+)
    # ------------------------------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Console handler (INFO+)
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Root logger
    # ------------------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)           # Let handlers filter
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _logging_configured = True

    root_logger.info(
        "Logging initialised — file: %s | console level: %s",
        log_file,
        logging.getLevelName(console_level),
    )
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Convenience helper to retrieve a named logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
