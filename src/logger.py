import datetime
import logging
import os
import threading
from logging.handlers import TimedRotatingFileHandler

from src.file_index import LOG_STORAGE

# 创建一个锁来确保线程安全
_logger_lock = threading.Lock()


def configure_logger(name: str) -> logging.Logger:
    # 获取进程号
    process_id = os.getpid()
    # 获取启动时间
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建日志文件名前缀
    log_filename_prefix = f'run{start_time}_pid{process_id}_{name}_'

    # 获取日志记录器
    logger = logging.getLogger(name)

    # 使用锁来确保线程安全
    with _logger_lock:
        # 检查日志记录器是否已经配置过
        if not logger.hasHandlers():
            # 创建格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(thread)d - %(message)s')

            # 创建不同级别的文件处理器
            debug_log_filepath = LOG_STORAGE / f'{log_filename_prefix}_debug.log'
            info_log_filepath = LOG_STORAGE / f'{log_filename_prefix}_info.log'
            warning_log_filepath = LOG_STORAGE / f'{log_filename_prefix}_warning.log'
            error_log_filepath = LOG_STORAGE / f'{log_filename_prefix}_error.log'
            critical_log_filepath = LOG_STORAGE / f'{log_filename_prefix}_critical.log'

            # 创建文件处理器
            debug_handler = TimedRotatingFileHandler(debug_log_filepath, interval=1, backupCount=65535,
                                                     encoding='utf-8')
            info_handler = TimedRotatingFileHandler(info_log_filepath, interval=1, backupCount=65535, encoding='utf-8')
            warning_handler = TimedRotatingFileHandler(warning_log_filepath, interval=1, backupCount=65535,
                                                       encoding='utf-8')
            error_handler = TimedRotatingFileHandler(error_log_filepath, interval=1, backupCount=65535,
                                                     encoding='utf-8')
            critical_handler = TimedRotatingFileHandler(critical_log_filepath, interval=1, backupCount=65535,
                                                        encoding='utf-8')

            # 设置处理器的日志级别
            debug_handler.setLevel(logging.DEBUG)
            info_handler.setLevel(logging.INFO)
            warning_handler.setLevel(logging.WARNING)
            error_handler.setLevel(logging.ERROR)
            critical_handler.setLevel(logging.CRITICAL)

            # 将格式化器添加到处理器
            debug_handler.setFormatter(formatter)
            info_handler.setFormatter(formatter)
            warning_handler.setFormatter(formatter)
            error_handler.setFormatter(formatter)
            critical_handler.setFormatter(formatter)

            # 将处理器添加到日志记录器
            logger.addHandler(debug_handler)
            logger.addHandler(info_handler)
            logger.addHandler(warning_handler)
            logger.addHandler(error_handler)
            logger.addHandler(critical_handler)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    # 设置日志记录器级别
    logger.setLevel(logging.DEBUG)

    return logger
