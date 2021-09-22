import os

import numpy as np
import pandas as pd
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from sdv.constraints import Positive

from checkpoint import CTGANCheckpoint, ResultsCheckpoint
import config
import logger

config.log_parameter_summary(True, True)

def load_data():
    """ The function load_data loads the data of the dataset specified in config.DATASET and
    takes care of some pre-processing.
    This pre-processing consists of calculating the duration of each activity, resetting the
    index and filling missing data (nan) with 0.
    Some statistics of the dataset and the steps executed is logged.
    """
    logger.log(f"Load data from file '{config.DATASET}'")
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
    return data, dataframe


def _fit_ctgan(data: pd.DataFrame, dataframe: pd.DataFrame):
    pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
    ctgan = CTGAN(epochs=config.EPOCHS_CTGAN, batch_size=config.BATCH_SIZE, constraints=[pos_constraint])
    ctgan.fit(
        data, 
        dataframe[['concept:name','duration','case:concept:name','time:timestamp']],
        disabled_dp=not config.ENABLED_DP_CTGAN
    )
    return ctgan


def get_fitted_model(data: pd.DataFrame, dataframe: pd.DataFrame) -> CTGAN:
    """ The function get_fitted_model uses a CTGAN Checkpoint (see chapter about checkpoints),
    to load a trained CTGAN model if one is available with the desired hyperparameters, or
    train a new one if none is available. The function then returns the trained CTGAN model.

    The CTGAN model created here uses a 'Positive' constraint for the dataframe column 'duration',
    which contains the duration of each activity. The 'reject_sampling' strategy is used as
    handling strategy for this constraint.

    The function logs wether a pre-trained model was loaded or a new one was generated.
    """
    cp = CTGANCheckpoint(
        config.get_dataset_basename(), config.EPOCHS_CTGAN, config.ENABLED_DP_CTGAN, "{:.1f}".format(config.EPSILON_CTGAN))
    return cp.load_if_exists_else_generate(config.RETRAIN_CTGAN, _fit_ctgan, data, dataframe)

def is_concept_names_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """ The function is_concept_names_equal compares the column 'concept:name' of two dataframes
    for equality and returns True if they are equal, or False otherwise. The columns are considered
    equal, if the number of rows of both columns is the same and if the values in both rows are equal.
    This method is used after sampling to compare the activities generated with the LSTM model with the
    activities generated with the (DP)CTGAN  model.
    This function includes logging to show the values of rows that didn't match as well as to give an
    overview of the final result of the comparison. This summary includes the total number of how many
    rows matched, both as a percentage of the total number of rows as well as in total numbers.
    """
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
    """ This function saves the dataframe with the final results to a checkpoint file. For later evaluation,
    this file can then be opened using the utilities in 'evaluate.py'.
    """
    cp = ResultsCheckpoint(
        config.get_dataset_basename(), config.EPOCHS_CTGAN, config.ENABLED_DP_CTGAN
    )
    cp.save(results_df, override=config.OVERRIDE_EXISTING_RESULTS)


def main():
    """ The main function is the main entry point into the program. It contains the whole sequence of creating
    the CTGAN model and fitting it (see function 'get_fitted_model'), sampling the model, comparing the result
    to see if the activities match (see function 'is_concept_names_equal') and saving the result as a csv file
    (see function 'save_results').
    The result of the sequence is also logged and printed to the console.

    All hyperparameters of the models as well as basic setup settings are controlled through the file 'config.py'.
    """
    data, dataframe = load_data()
    ctgan = get_fitted_model(data, dataframe)

    
    logger.log("Generating activities...")
    org_data = dataframe[['concept:name','duration','case:concept:name','time:timestamp']]
    global_condition_vec, activities = ctgan._model._data_sampler.generate_cond_from_condition_column_info(
                                                                config.BATCH_SIZE, data, org_data, config.EPOCHS_DPLSTM)

    logger.log("\n\tSampling model.\n")
    sampled, activities = ctgan.sample(global_condition_vec, activities, num_rows=len(activities))
    # Make sure they have the same activities
    if is_concept_names_equal(activities, sampled):
        logger.log("equal --> inner join by key with concept:name")
        sampled['traces'] = activities['traces'].values
    else:
        # Activities don't match. Insert the lstm activities for
        # comparison
        logger.log('Activities don\'t match --> make sampled conc')
        sampled['lstm_activities'] = activities.iloc[:len(sampled)]['concept:name'].values

    # Save the sampled data result
    save_results(sampled)

    # Log the results
    logger.log(activities, summary=True)
    logger.log(data, summary=True)
    logger.log(sampled, summary=True)

    pass


if __name__ == "__main__":
    main()
