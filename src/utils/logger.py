import os
from datetime import datetime
import logging
import multiprocessing


class Logger:
    def __init__(self, log_path='', name='logger', level='INFO'):
        self.log_name = name
        self.log_fname = "".join(['run', '.log'])
        self.log_path = os.path.join(log_path, self.log_fname)
        self.logger = self._setup_logger(level)

    def _setup_logger(self, level):
        # define logger
        if not multiprocessing.get_logger().hasHandlers():
            logger = multiprocessing.get_logger()
            logger.setLevel(level)

            # create formatter
            formatter = logging.Formatter(f'%(asctime)s [%(levelname)s | %(processName)s] {self.log_name}: %(message)s')

            # Use FileHandler() to log to a file
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(formatter)

            # add the file handler
            logger.addHandler(file_handler)
            
            return logger

        return multiprocessing.get_logger()

    def clear_logs(self):
        """
            Clear all the data from the log file.
        """
        with open(self.log_path, mode='w'):
            pass

    def critical(self, msg):
        self.logger.critical(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def reinitialize(self):
        # clear log file
        self.clear_logs()
