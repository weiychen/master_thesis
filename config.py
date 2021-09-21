""" 
The file config.py contains all major configuration parameters for running the
DPCTGAN model.
"""

import os

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

import logger

# Parameters for the checkpoints. These settings control, wether pre-trained
# models are used (if existing) or if they are trained again.
RETRAIN_CTGAN = False
RETRAIN_LSTM = False
OVERRIDE_EXISTING_RESULTS = True
CHECKPOINTS_ROOT = "checkpoints"

# Global parameters used in all parts of the model (LSTM and CTGAN)
BATCH_SIZE = 20

# CTGAN
EPOCHS_CTGAN = 100
ENABLED_DP = True
EPSILON_CTGAN = 2.0

SAMPLING_BATCH_SIZE = None # Set to None to use global BATCH_SIZE
SAMPLING_MATCH_ACTIVITIES_MAX_TRIES = 250

# LSTM
EPOCHS_DPLSTM = 100
EPSILON_LSTM_DP = 2.0

# Dataset
DATASET = (
    # 'datasets/ETM_Configuration2.xes',
    'datasets/financial_log.xes',
    # 'datasets/Sepsis Cases - Event Log.xes',
)[0]

# Logging
LOGGING_FOLDER = "Logs"
LOG_FILE = os.path.join(LOGGING_FOLDER, "{datetime}_test.py.logs")
SUMMARY_LOG_FILE = os.path.join(LOGGING_FOLDER, "{datetime}_summary.log")


# =============== Functions =================

__dataset_df = None
def get_dataset_df():
    """ Get a copy of the dataframe with the original dataset. If the dataset
    was not loaded into a dataframe yet that step will be done first.
    If this function is called multiple times, it will not load the dataset
    again, but just copy the dataframe created during the first run.
    """
    global __dataset_df
    if __dataset_df is None:
        log = xes_importer.apply(DATASET)
        __dataset_df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return __dataset_df.copy()[0:2000]

def get_dataset_basename():
    """ Get the name of the dataset used, without path information and
    without file type extension.
    """
    return os.path.basename(DATASET).split(".")[0]


def log_parameter_summary(main_logfile=True, summary=False):
    logger.log("Main parameter summary:", main_logfile, summary)
    logger.sep(main_logfile=main_logfile, summary=summary)
    # Override existing checkpoints?
    logger.log("RETRAIN_CTGAN                              : " + str(RETRAIN_CTGAN), main_logfile, summary)
    logger.log("RETRAIN_LSTM                               : " + str(RETRAIN_LSTM), main_logfile, summary)
    logger.log("OVERRIDE_EXISTING_RESULTS                  : " + str(OVERRIDE_EXISTING_RESULTS), main_logfile, summary)

    # Global parameters
    logger.log("BATCH_SIZE                                 : " + str(BATCH_SIZE), main_logfile, summary)

    # CTGAN
    logger.log("EPOCHS_CTGAN                               : " + str(EPOCHS_CTGAN), main_logfile, summary)
    logger.log("ENABLED_DP                                 : " + str(ENABLED_DP), main_logfile, summary)
    logger.log("RETRAIN_CTGAN                              : " + str(RETRAIN_CTGAN), main_logfile, summary)
    logger.log("EPSILON_CTGAN                              : " + str(EPSILON_CTGAN), main_logfile, summary)

    logger.log("SAMPLING_BATCH_SIZE                        : " + str(SAMPLING_BATCH_SIZE), main_logfile, summary)
    logger.log("SAMPLING_MATCH_ACTIVITIES_MAX_TRIES        : " + str(SAMPLING_MATCH_ACTIVITIES_MAX_TRIES), main_logfile, summary)

    # LSTM
    logger.log("EPOCHS_DPLSTM                              : " + str(EPOCHS_DPLSTM), main_logfile, summary)
    logger.log("EPSILON_LSTM_DP                            : " + str(EPSILON_LSTM_DP), main_logfile, summary)

    # Dataset
    logger.log("DATASET                                    : " + str(DATASET), main_logfile, summary)
    
    logger.sep(main_logfile=main_logfile, summary=summary)