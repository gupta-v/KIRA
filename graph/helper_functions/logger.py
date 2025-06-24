import os
import logging
import re
from datetime import datetime

_logger = None

def sanitize_filename(text, max_length=60):
    # Remove invalid filename characters and truncate
    text = re.sub(r'[\\/*?:"<>|]', '', text)
    text = text.strip().replace(' ', '_')
    return text[:max_length]

def setup_logging(query=None):
    """Setup logging with logs directory creation and query-based filename"""
    global _logger
    logs_dir = "data/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")

    if query:
        query_part = sanitize_filename(query)
        log_filename = f"ai_tutor_{query_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_filename = f"ai_tutor_default.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Remove all handlers associated with the root logger object (for reconfiguration)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    _logger = logging.getLogger(__name__)
    _logger.info(f"Logging initialized. Log file: {log_filepath}")
    return _logger

def get_logger():
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger