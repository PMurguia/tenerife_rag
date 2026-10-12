import logging
from pathlib import Path

logger = logging.getLogger("tenerife_rag.tools")

def log_tool_call(name: str, arguments: str, result: dict) -> None:
    logger.info("tool=%s args=%s ok=%s error=%s", name, arguments, result.get("ok"), result.get("error"))
    
def setup_logging(log_dir="logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "tool_calls.log"
    # Idempotent: re-running a notebook cell must not attach a second handler to the same file
    if any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.absolute()
        for h in logger.handlers
    ):
        return
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)