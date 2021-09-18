import os
import logging

import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from pm4py.objects.log.importer.xes import importer as xes_importer
from sdv.constraints import Positive

from checkpoint import Checkpoint, DataframeSaveLoad, CTGANSaveLoad

# Create folder for logging
LOGGING_FOLDER = "Logs"
os.makedirs(LOGGING_FOLDER, exist_ok=True) # create folder if not exists
LOGGING_FILE = os.path.join(LOGGING_FOLDER, "2__test.py.logs")

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

fileHandler = logging.FileHandler("{}".format(LOGGING_FILE))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

dataset = (
    # 'datasets/ETM_Configuration2.xes',
    'datasets/financial_log.xes',
)[0]

# import datetime
rootLogger.info(f"Load data from file '{dataset}'")
log = xes_importer.apply(dataset)
dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
def infer_time(dataframe):
    return -dataframe['time:timestamp'].diff(-1).dt.total_seconds()
rootLogger.info("Len dataframe:" + str(len(dataframe)))
rootLogger.info("Calculate durations.")
duration= dataframe.groupby('case:concept:name').apply(infer_time)
dataframe['duration'] = duration.droplevel(0).reset_index(drop = True)
rootLogger.info("Resetting index.")
dataframe = dataframe.reset_index(drop = True)
data = dataframe[['concept:name','duration']]
rootLogger.info("Fill na.")
data = data.fillna(0) ## maybe before training
rootLogger.info("Finished data loading.")


# Override existing checkpoints?
RETRAIN_CTGAN = False
OVERRIDE_EXISTING_RESULTS = True

# Settings for training and sampling
BATCH_SIZE = 20
# EPOCHS_DPLSTM = 40
EPOCHS_CTGAN = 10
ENABLED_DP = False


def get_fitted_model():
    """ Load an already fitted model from checkpoint or fit a new one. """
    cp = Checkpoint("fitted_models", CTGANSaveLoad(), "ctgan", ".mdl")
    cp.add_info("dataset", os.path.basename(dataset).split(".")[0])
    cp.add_info("epochs", EPOCHS_CTGAN)
    cp.add_info("dp", ENABLED_DP)

    if cp.exists() and not RETRAIN_CTGAN:
        rootLogger.info("Loading trained model from '{}'".format(cp.save_file))
        ctgan = cp.load()
    else:
        rootLogger.info("Retraining model...")
        pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
        ctgan = CTGAN(epochs=EPOCHS_CTGAN, batch_size=BATCH_SIZE, constraints=[pos_constraint])
        ctgan.fit(
            data, 
            dataframe[['concept:name','duration','case:concept:name','time:timestamp']],
            disabled_dp=not ENABLED_DP
        )
        cp.save(ctgan)
    return ctgan


def is_concept_names_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    len1 = len(df1['concept:name'])
    len2 = len(df2['concept:name'])

    # print names for debugging
    same = len1 == len2
    for i in range(min(len1, len2)):
        try:
            if df1.iloc[i]['concept:name'] != df2.iloc[i]['concept:name']:
                rootLogger.info("Activity not the same (i={}): {:<2} != {:<2}".format(i, df1.iloc[i]['concept:name'], df2.iloc[i]['concept:name']))
                same = False
        except IndexError:
            # End of one array reached
            break
    rootLogger.info("Lengths (df1-df2): {}-{}".format(len1, len2))
    return same


def save_results(results_df: pd.DataFrame):
    cp = Checkpoint("results", DataframeSaveLoad(), "sampled", ".csv")
    cp.add_info("dataset", os.path.basename(dataset).split(".")[0])
    cp.add_info("epochs", EPOCHS_CTGAN)
    cp.add_info("dp", ENABLED_DP)
    cp.save(results_df, override=OVERRIDE_EXISTING_RESULTS)


def main():
    
    discrete_columns = ['concept:name']
    ctgan = get_fitted_model()

    rootLogger.info("\n\tSampling model.\n")
    sampled, activities = ctgan.sample(len(data))

    # TODO: Make sure they have the same activities
    if is_concept_names_equal(activities, sampled):
        rootLogger.info("equal --> inner join by key with concept:name")
        sampled['traces'] = activities['traces'].values
    else:
        # Activities don't match. Insert the lstm activities for
        # comparison
        rootLogger.info('Activities don\'t match --> make sampled conc')
        sampled['lstm_activities'] = activities['concept:name'].values

    # Save the sampled data result
    save_results(sampled)

    rootLogger.info(activities)
    rootLogger.info(data)
    rootLogger.info(sampled)

    pass


if __name__ == "__main__":
    main()
