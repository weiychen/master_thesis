import os
import logging

dataset = (
    # 'datasets/ETM_Configuration2.xes',
    'datasets/financial_log.xes',
)[0]

# Logging
LOGGING_FOLDER = "Logs"
LOG_FILE = os.path.join(LOGGING_FOLDER, "2__test.py.logs")

__logger = None
def get_logger():
    global __logger
    if __logger is not None:
        return __logger
    else:
        # Create folder for logging
        os.makedirs(LOGGING_FOLDER, exist_ok=True) # create folder if not exists

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        __logger = logging.getLogger()
        __logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler("{}".format(LOG_FILE))
        fileHandler.setFormatter(logFormatter)
        __logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        __logger.addHandler(consoleHandler)
        return __logger