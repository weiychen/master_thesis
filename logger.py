import os
import time
import logging

import config

def get_log_time_str():
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

__logger = None
def get_logger():
    global __logger
    if __logger is None:
        # Create folder for logging
        os.makedirs(config.LOGGING_FOLDER, exist_ok=True) # create folder if not exists

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        __logger = logging.getLogger()
        __logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(config.LOG_FILE.format(datetime=get_log_time_str()))
        fileHandler.setFormatter(logFormatter)
        __logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        __logger.addHandler(consoleHandler)
    return __logger

__summary_logger = None
def get_summary_logger():
    global __summary_logger
    if __summary_logger is None:
        # Create folder for logging
        os.makedirs(config.LOGGING_FOLDER, exist_ok=True) # create folder if not exists

        logFormatter = logging.Formatter("%(message)s")
        __summary_logger = logging.getLogger("summary_logger")
        __summary_logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(config.SUMMARY_LOG_FILE.format(datetime=get_log_time_str()))
        fileHandler.setFormatter(logFormatter)
        __summary_logger.addHandler(fileHandler)

        __summary_logger.propagate = False
    return __summary_logger


def log(msg, main_logfile=True, summary=False):
    if main_logfile:
        logger = get_logger()
        logger.info(msg)
    if summary:
        sum_logger = get_summary_logger()
        sum_logger.info(msg)

def sep(main_logfile=True, summary=False):
    log("==============================================================", main_logfile, summary)
