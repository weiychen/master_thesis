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
EPOCHS_CTGAN = 2
ENABLED_DP = False
RETRAIN_CTGAN = False

SAMPLING_MATCH_ACTIVITIES_MAX_TRIES = 200

# LSTM
EPOCHS_DPLSTM = 40
EPSILON_LSTM_DP = 1.0

# Dataset
DATASET = (
    'datasets/ETM_Configuration2.xes',
    # 'datasets/financial_log.xes',
)[0]

__dataset_df = None
def get_dataset_df():
    global __dataset_df
    if __dataset_df is not None:
        return __dataset_df
    else:
        log = xes_importer.apply(DATASET)
        __dataset_df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        return __dataset_df


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
