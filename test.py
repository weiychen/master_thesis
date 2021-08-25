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


def main():

    # size = 10_000
    # data = pd.DataFrame({
    #     'continuous': np.random.random(size),
    #     'discrete': np.random.choice(['a', 'b', 'c'], size),
    #     #'discrete2': np.random.choice(['e', 'f', 'g'], 100)
    # })
    discrete_columns = ['concept:name']
    pos_constraint = Positive(columns='duration',strict=False, handling_strategy='reject_sampling')
    ctgan = CTGAN(epochs=3,batch_size=20,constraints=[pos_constraint])
    # ctgan = DPCTGAN(epochs=50, batch_size=10)#epochs=50, batch_size=10
    # ctgan.fit(data, discrete_columns)
    
    ctgan.fit(data, dataframe[['concept:name','duration','case:concept:name','time:timestamp']])
    sampled = ctgan.sample(len(data))#, discrete_columns[0], "A")#, discrete_columns[0], "b"

    print(data)
    print(sampled)

    pass


if __name__ == "__main__":
    main()
