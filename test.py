import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from sdv.tabular.ctgan import CTGAN
from DPCTGAN import DPCTGAN
from pm4py.objects.log.importer.xes import importer as xes_importer


# import datetime
log = xes_importer.apply('ETM_Configuration2.xes')#('financial_log.xes')
dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
def infer_time(dataframe):
    return -dataframe['time:timestamp'].diff(-1).dt.total_seconds()

duration= dataframe.groupby('case:concept:name').apply(infer_time)
dataframe['duration'] =duration.droplevel(0)
data = dataframe[['concept:name','duration']]
data = data.fillna(0) ## maybe before training
batch_size=10
def main(data):

    # size = 10_000
    # data = pd.DataFrame({
    #     'continuous': np.random.random(size),
    #     'discrete': np.random.choice(['a', 'b', 'c'], size),
    #     #'discrete2': np.random.choice(['e', 'f', 'g'], 100)
    # })
    discrete_columns = ['concept:name']
    # ctgan = CTGANModel()
    ctgan = CTGAN(epochs=50, batch_size=10)
    ctgan.fit(data, discrete_columns)
    sampled = ctgan.sample(len(data), discrete_columns[0], "A")#, discrete_columns[0], "b"

    print(data)
    print(sampled)

    pass


if __name__ == "__main__":
    main(data)

    