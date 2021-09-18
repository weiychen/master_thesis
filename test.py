import os

import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from pm4py.objects.log.importer.xes import importer as xes_importer
from sdv.constraints import Positive

import logging

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


# Create folder for saved pre-fitted models
MODEL_FOLDER = "fitted_models"
os.makedirs(MODEL_FOLDER, exist_ok=True) # create folder if not exists
MODEL_FILE_PATTERN = os.path.join(MODEL_FOLDER, "{}_{}-epochs_dp-{}.mdl")
RETRAIN = False

# Create folder for saved sampled data (result) for later evaluation
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE_PATTERN = os.path.join(RESULTS_FOLDER, "{}_sampled_{}-epochs_dp-{}.csv")
OVERRIDE_EXISTING_CSV = True

# Settings for training and sampling
BATCH_SIZE = 20
# EPOCHS_DPLSTM = 40
EPOCHS_CTGAN = 100
ENABLED_DP = False


def save_model(model: CTGAN, path: str, override=False):
    if not os.path.exists(path) or override:
        model.save(path)


def get_fitted_model():
    """ Load an already fitted model from file or fit a new one. """
    dataset_name = os.path.basename(dataset).split(".")[0]
    model_file = MODEL_FILE_PATTERN.format(dataset_name, EPOCHS_CTGAN, ENABLED_DP)
    if os.path.exists(model_file) and not RETRAIN:
        rootLogger.info("Loading trained model from '{}'".format(model_file))
        ctgan = CTGAN.load(model_file)
    else:
        rootLogger.info("Retraining model...")
        pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
        ctgan = CTGAN(epochs=EPOCHS_CTGAN, batch_size=BATCH_SIZE, constraints=[pos_constraint])
        ctgan.fit(
            data, 
            dataframe[['concept:name','duration','case:concept:name','time:timestamp']],
            disabled_dp=not ENABLED_DP
        )
        save_model(ctgan, model_file, override=True)
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
    dataset_name = os.path.basename(dataset).split(".")[0]
    csv_file = RESULTS_FILE_PATTERN.format(dataset_name, EPOCHS_CTGAN, ENABLED_DP)
    if not os.path.exists(csv_file) or OVERRIDE_EXISTING_CSV:
        # Save only if file doesn't already exist or override flag set
        results_df.to_csv(csv_file)
    

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
        sampled['lstm_activities'] = activities['concept:name'].copy()

    # Save the sampled data result to the file specified in the settings
    save_results(sampled)

    rootLogger.info(data)
    rootLogger.info(sampled)

    pass


if __name__ == "__main__":
    main()
