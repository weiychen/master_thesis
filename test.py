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
        # ctgan = DPCTGAN(epochs=50, batch_size=10)#epochs=50, batch_size=10
        # ctgan.fit(data, discrete_columns)
        save_model(ctgan, MODEL_FILE, override=True)
    return ctgan

def main():

    # size = 10_000
    # data = pd.DataFrame({
    #     'continuous': np.random.random(size),
    #     'discrete': np.random.choice(['a', 'b', 'c'], size),
    #     #'discrete2': np.random.choice(['e', 'f', 'g'], 100)
    # })
    discrete_columns = ['concept:name']
    ctgan = get_fitted_model()

    print("\nSampling model.\n")
    sampled = ctgan.sample(len(data))#, discrete_columns[0], "A")#, discrete_columns[0], "b"

    # TODO:
    if activities['concept:name'] == sampled['concept:name']:
           print( "inner join by key with concept:name")
    else:
        print('not equal--> make sampled conc')

    print(data)
    print(sampled)

    pass


if __name__ == "__main__":
    main()
