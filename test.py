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

MODEL_FILE = "ctgan_trained_model.mdl"
RETRAIN = True

def save_model(model, path, override=False):
    if not os.path.exists(path) or override:
        model.save(path)

def get_fitted_model():
    """ Load an already fitted model from file or fit a new one. """
    if os.path.exists(MODEL_FILE) and not RETRAIN:
        print("Loading trained model from '{}'".format(MODEL_FILE))
        ctgan = CTGAN.load(MODEL_FILE)
    else:
        print("Retraining model...")
        pos_constraint = Positive(columns='duration', strict=False, handling_strategy='reject_sampling')
        ctgan = CTGAN(epochs=1, batch_size=20, constraints=[pos_constraint])
        ctgan.fit(data, dataframe[['concept:name','duration','case:concept:name','time:timestamp']])
        save_model(ctgan, MODEL_FILE, override=True)
    return ctgan


def is_concept_names_equal(df1, df2, ignore_length=True) -> bool:
    len1 = len(df1['concept:name'])
    len2 = len(df2['concept:name'])

    if len1 != len2 and not ignore_length:
        return False

    # print names for debugging
    for i in range(min(len1, len2)):
        try:
            print("{:<2} ?= {:<2}".format(df1.iloc[i]['concept:name'], df2.iloc[i]['concept:name']))
        except IndexError:
            # End of one array reached
            break
    print("Lengths (df1-df2): {}-{}".format(len1, len2))
    return len(df1[df1['concept:name'] != df2['concept:name'].values]) == 0


def main():
    
    discrete_columns = ['concept:name']
    ctgan = get_fitted_model()

    print("\nSampling model.\n")
    sampled, activities = ctgan.sample(len(data))

    # TODO: Make sure they have the same activities
    if is_concept_names_equal(activities, sampled):
        print("inner join by key with concept:name")
    else:
        print('not equal--> make sampled conc')

    print(data)
    print(sampled)

    pass


if __name__ == "__main__":
    main()
