import os

import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from pm4py.objects.log.importer.xes import importer as xes_importer
from sdv.constraints import Positive

# import datetime
log = xes_importer.apply('ETM_Configuration2.xes')#('financial_log.xes')
dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
def infer_time(dataframe):
    return -dataframe['time:timestamp'].diff(-1).dt.total_seconds()
print(len(dataframe))
df = dataframe.copy()
df['case:concept:name'] = dataframe['case:concept:name'].astype(str) + '_1'
df = dataframe.append(df)
df_2 = df.copy()
df_2['case:concept:name'] = df['case:concept:name'].astype(str) + '_2'
df = dataframe.append(df_2)
df_3 = df.copy()
df_3['case:concept:name'] = df['case:concept:name'].astype(str) + '_3'
df = dataframe.append(df_3)
df_2 = df.copy()
df_2['case:concept:name'] = df['case:concept:name'].astype(str) + '_4'
df = dataframe.append(df_2)
df_2 = df.copy()
df_2['case:concept:name'] = df['case:concept:name'].astype(str) + '_5'
dataframe = dataframe.append(df_2)
duration= dataframe.groupby('case:concept:name').apply(infer_time)
dataframe['duration'] =duration.droplevel(0).reset_index(drop = True)
dataframe = dataframe.reset_index(drop = True)
data = dataframe[['concept:name','duration']]
data = data.fillna(0) ## maybe before training
# batch_size=50

# Create folder for saved pre-fitted models
MODEL_FOLDER = "fitted_models"
os.makedirs(MODEL_FOLDER, exist_ok=True) # create folder if not exists
MODEL_FILE_PATTERN = os.path.join(MODEL_FOLDER, "mdl_{}-epochs_dp-{}.mdl")
RETRAIN = False

# Create folder for saved sampled data (result) for later evaluation
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
RESULTS_FILE_PATTERN = os.path.join(RESULTS_FOLDER, "sampled_{}-epochs_dp-{}.csv")
OVERRIDE_EXISTING_CSV = True

# Settings for training and sampling
EPOCHS = 10
DISABLED_DP = True


def save_model(model: CTGAN, path: str, override=False):
    if not os.path.exists(path) or override:
        model.save(path)


def get_fitted_model():
    """ Load an already fitted model from file or fit a new one. """
    model_file = MODEL_FILE_PATTERN.format(EPOCHS, not DISABLED_DP)
    if os.path.exists(model_file) and not RETRAIN:
        print("Loading trained model from '{}'".format(model_file))
        ctgan = CTGAN.load(model_file)
    else:
        print("Retraining model...")
        pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
        ctgan = CTGAN(epochs=EPOCHS, batch_size=20, constraints=[pos_constraint])
        ctgan.fit(
            data, 
            dataframe[['concept:name','duration','case:concept:name','time:timestamp']],
            disabled_dp=DISABLED_DP
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
                print("Activity not the same (i={}): {:<2} != {:<2}".format(i, df1.iloc[i]['concept:name'], df2.iloc[i]['concept:name']))
                same = False
        except IndexError:
            # End of one array reached
            break
    print("Lengths (df1-df2): {}-{}".format(len1, len2))
    return same


def save_results(results_df: pd.DataFrame):
    csv_file = RESULTS_FILE_PATTERN.format(EPOCHS, not DISABLED_DP)
    if not os.path.exists(csv_file) or OVERRIDE_EXISTING_CSV:
        # Save only if file doesn't already exist or override flag set
        results_df.to_csv(csv_file)
    

def main():
    
    discrete_columns = ['concept:name']
    ctgan = get_fitted_model()

    print("\nSampling model.\n")
    sampled, activities = ctgan.sample(len(data))

    # TODO: Make sure they have the same activities
    if is_concept_names_equal(activities, sampled):
        print("equal --> inner join by key with concept:name")
        sampled['traces'] = activities['traces'].values
    else:
        print('not equal --> make sampled conc')

    save_results(sampled)

    print(data)
    print(sampled)

    pass


if __name__ == "__main__":
    main()
