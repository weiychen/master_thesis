import os

import numpy as np
import pandas as pd
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from sdv.constraints import Positive

from checkpoint import CTGANCheckpoint, DataframeCheckpoint
import config
import logger

logger.log_parameter_summary(True, True)

# import datetime
logger.log(f"Load data from file '{config.DATASET}'", summary=True)

dataframe = config.get_dataset_df()
def infer_time(dataframe):
    return -dataframe['time:timestamp'].diff(-1).dt.total_seconds()
logger.log("Len dataframe:" + str(len(dataframe)), summary=True)
logger.log("Calculate durations.")
duration= dataframe.groupby('case:concept:name').apply(infer_time)
dataframe['duration'] = duration.droplevel(0).reset_index(drop = True)
logger.log("Resetting index.")
dataframe = dataframe.reset_index(drop = True)
data = dataframe[['concept:name','duration']]
logger.log("Fill na.")
data = data.fillna(0) ## maybe before training
logger.log("Finished data loading.")


def get_fitted_model():
    """ Load an already fitted model from checkpoint or fit a new one. """
    cp = CTGANCheckpoint(
        config.get_dataset_basename(), config.EPOCHS_CTGAN, config.ENABLED_DP)

    if cp.exists() and not config.RETRAIN_CTGAN:
        logger.log("Loading trained CTGAN model from '{}'".format(cp.save_file), summary=True)
        ctgan = cp.load()
    else:
        logger.log("Retraining model...", summary=True)
        pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
        ctgan = CTGAN(epochs=config.EPOCHS_CTGAN, batch_size=config.BATCH_SIZE, constraints=[pos_constraint])
        ctgan.fit(
            data, 
            dataframe[['concept:name','duration','case:concept:name','time:timestamp']],
            disabled_dp=not config.ENABLED_DP
        )
        cp.save(ctgan)
    return ctgan


def is_concept_names_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    len1 = len(df1['concept:name'])
    len2 = len(df2['concept:name'])

    num_match = 0
    num_no_match = 0

    # print names for debugging
    same = len1 == len2
    for i in range(min(len1, len2)):
        try:
            if df1.iloc[i]['concept:name'] != df2.iloc[i]['concept:name']:
                logger.log("Activity not the same (i={}): {:<2} != {:<2}".format(i, df1.iloc[i]['concept:name'], df2.iloc[i]['concept:name']), summary=True)
                num_no_match += 1
                same = False
            else:
                num_match += 1
        except IndexError:
            # End of one array reached
            break
    logger.log("Lengths (df1-df2): {}-{}".format(len1, len2))
    total = num_match+num_no_match
    perc_match = num_match / total * 100
    logger.sep(True, True)
    logger.log("{} ({:.1f}%) of {} activities matched. {} didn't match.".format(num_match, perc_match, total, num_no_match), summary=True)
    logger.sep(True, True)
    return same


def save_results(results_df: pd.DataFrame):
    cp = DataframeCheckpoint(
        config.get_dataset_basename(), config.EPOCHS_CTGAN, config.ENABLED_DP
    )
    cp.save(results_df, override=config.OVERRIDE_EXISTING_RESULTS)


def main():
    ctgan = get_fitted_model()

    logger.log("\n\tSampling model.\n")
    sampled, activities = ctgan.sample(len(data))

    # TODO: Make sure they have the same activities
    if is_concept_names_equal(activities, sampled):
        logger.log("equal --> inner join by key with concept:name")
        sampled['traces'] = activities['traces'].values
    else:
        # Activities don't match. Insert the lstm activities for
        # comparison
        logger.log('Activities don\'t match --> make sampled conc')
        sampled['lstm_activities'] = activities['concept:name'].values

    # Save the sampled data result
    save_results(sampled)

    logger.log(activities, summary=True)
    logger.log(data, summary=True)
    logger.log(sampled, summary=True)

    pass


if __name__ == "__main__":
    main()
