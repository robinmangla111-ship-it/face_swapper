"""
Session Logging and Diagnostics Module
=======================================
Creates separate log files for each session/operation, captures exceptions
with detailed stack traces, and provides utilities to list and delete older logs.
"""

import os
import glob
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LOGS_DIR = Path(__file__).parent.resolve() / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global active log file for the current session
_current_log_file: Optional[Path] = None


def get_current_log_path() -> Path:
    """Return the active log file or create one if none exists."""
    global _current_log_file
    if _current_log_file is None or not _current_log_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _current_log_file = LOGS_DIR / f"session_{timestamp}.log"
        # Write header
        with open(_current_log_file, "a", encoding="utf-8") as f:
            f.write(f"=== FACE SWAPPER SESSION LOG [{timestamp}] ===\n")
    return _current_log_file


def start_new_session_log(prefix: str = "upload") -> Path:
    """Start a brand new separate log file for a new operation/upload."""
    global _current_log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _current_log_file = LOGS_DIR / f"{prefix}_{timestamp}.log"
    with open(_current_log_file, "w", encoding="utf-8") as f:
        f.write(f"=== FACE SWAPPER LOG: {prefix.upper()} [{timestamp}] ===\n\n")
    return _current_log_file


def write_log(message: str, level: str = "INFO", exc: Optional[Exception] = None):
    """Write a formatted entry to the current session log and console."""
    log_path = get_current_log_path()
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level.upper()}] {message}\n"
    
    if exc is not None:
        entry += f"--- EXCEPTION DETAILS ---\n"
        entry += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        entry += "-------------------------\n"
        
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        print(f"Failed to write to log file: {e}")

    # Also log to standard logger
    py_logger = logging.getLogger("face_swapper")
    if level == "ERROR":
        py_logger.error(message, exc_info=exc is not None)
    elif level == "WARNING":
        py_logger.warning(message)
    else:
        py_logger.info(message)


def list_log_files() -> List[Dict]:
    """List all log files ordered from newest to oldest."""
    files = glob.glob(str(LOGS_DIR / "*.log"))
    results = []
    for f in sorted(files, key=os.path.getmtime, reverse=True):
        p = Path(f)
        stat = p.stat()
        results.append({
            "filename": p.name,
            "path": str(p),
            "size_kb": round(stat.st_size / 1024, 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "is_current": _current_log_file is not None and p.name == _current_log_file.name,
        })
    return results


def read_log_file(filename: Optional[str] = None) -> Dict:
    """Read contents of a specific log file or the latest log."""
    if filename:
        target = LOGS_DIR / filename
    else:
        target = get_current_log_path()

    if not target.exists():
        return {
            "filename": filename or "none",
            "content": "No log file found.",
            "error_found": False
        }

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        has_error = "[ERROR]" in content or "EXCEPTION DETAILS" in content
        return {
            "filename": target.name,
            "content": content,
            "size_kb": round(target.stat().st_size / 1024, 2),
            "error_found": has_error
        }
    except Exception as e:
        return {
            "filename": target.name,
            "content": f"Failed to read log: {e}",
            "error_found": True
        }


def delete_all_logs() -> int:
    """Delete all log files in logs directory."""
    global _current_log_file
    count = 0
    for f in glob.glob(str(LOGS_DIR / "*.log")):
        try:
            os.remove(f)
            count += 1
        except Exception:
            pass
    _current_log_file = None
    return count


def delete_single_log(filename: str) -> bool:
    """Delete a specific log file by name."""
    global _current_log_file
    target = LOGS_DIR / filename
    if target.exists():
        try:
            target.unlink()
            if _current_log_file and _current_log_file.name == filename:
                _current_log_file = None
            return True
        except Exception:
            return False
    return False
