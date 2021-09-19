import os
import logging

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

# Override existing checkpoints?
RETRAIN_CTGAN = False
RETRAIN_LSTM = False
OVERRIDE_EXISTING_RESULTS = True

# Global parameters
BATCH_SIZE = 20

# CTGAN
EPOCHS_CTGAN = 20
ENABLED_DP = False
RETRAIN_CTGAN = False

SAMPLING_BATCH_SIZE = 4 # Set to None to use global BATCH_SIZE
SAMPLING_MATCH_ACTIVITIES_MAX_TRIES = 2000

# LSTM
EPOCHS_DPLSTM = 40
EPSILON_LSTM_DP = 1.0

# Dataset
DATASET = (
    'datasets/ETM_Configuration2.xes',
    # 'datasets/financial_log.xes',
)[0]

# Logging
LOGGING_FOLDER = "Logs"
LOG_FILE = os.path.join(LOGGING_FOLDER, "test.py.logs")
SUMMARY_LOG_FILE = os.path.join(LOGGING_FOLDER, "summary.log")


# =============== Functions to get special configurations =================

__dataset_df = None
def get_dataset_df():
    global __dataset_df
    if __dataset_df is None:
        log = xes_importer.apply(DATASET)
        __dataset_df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return __dataset_df.copy()

def get_dataset_basename():
    return os.path.basename(DATASET).split(".")[0]


__logger = None
def get_logger():
    global __logger
    if __logger is None:
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


__summary_logger = None
def get_summary_logger():
    global __summary_logger
    if __summary_logger is None:
        # Create folder for logging
        os.makedirs(LOGGING_FOLDER, exist_ok=True) # create folder if not exists

        logFormatter = logging.Formatter("%(message)s")
        __summary_logger = logging.getLogger("summary_logger")
        __summary_logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler("{}".format(SUMMARY_LOG_FILE), mode='w')
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

def log_parameter_summary(main_logfile=True, summary=False):
    log("==============================================================", main_logfile, summary)
    # Override existing checkpoints?
    log("RETRAIN_CTGAN                              : " + str(RETRAIN_CTGAN), main_logfile, summary)
    log("RETRAIN_LSTM                               : " + str(RETRAIN_LSTM), main_logfile, summary)
    log("OVERRIDE_EXISTING_RESULTS                  : " + str(OVERRIDE_EXISTING_RESULTS), main_logfile, summary)

    # Global parameters
    log("BATCH_SIZE                                 : " + str(BATCH_SIZE), main_logfile, summary)

    # CTGAN
    log("EPOCHS_CTGAN                               : " + str(EPOCHS_CTGAN), main_logfile, summary)
    log("ENABLED_DP                                 : " + str(ENABLED_DP), main_logfile, summary)
    log("RETRAIN_CTGAN                              : " + str(RETRAIN_CTGAN), main_logfile, summary)

    log("SAMPLING_BATCH_SIZE                        : " + str(SAMPLING_BATCH_SIZE), main_logfile, summary)
    log("SAMPLING_MATCH_ACTIVITIES_MAX_TRIES        : " + str(SAMPLING_MATCH_ACTIVITIES_MAX_TRIES), main_logfile, summary)

    # LSTM
    log("EPOCHS_DPLSTM                              : " + str(EPOCHS_DPLSTM), main_logfile, summary)
    log("EPSILON_LSTM_DP                            : " + str(EPSILON_LSTM_DP), main_logfile, summary)

    # Dataset
    log("DATASET                                    : " + str(DATASET), main_logfile, summary)
    
    log("==============================================================", main_logfile, summary)