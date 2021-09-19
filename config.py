import os

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
LOG_FILE = os.path.join(LOGGING_FOLDER, "{datetime}_test.py.logs")
SUMMARY_LOG_FILE = os.path.join(LOGGING_FOLDER, "{datetime}_summary.log")


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