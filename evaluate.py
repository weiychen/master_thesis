import os
import pandas as pd

from checkpoint import ResultsCheckpoint

# Settings of data to load. This is used to determine the
# file name
EPOCHS_CTGAN = 10
ENABLED_DP = False
dataset = (
    'datasets/ETM_Configuration2.xes',
    # 'datasets/financial_log.xes',
)[0]



def load_sampled_data():
    cp = ResultsCheckpoint(
        os.path.basename(dataset).split(".")[0], EPOCHS_CTGAN, ENABLED_DP)
    
    if cp.exists():
        return cp.load()
    else:
        print("File not found:", cp.save_file)

def evaluate():
    sampled = load_sampled_data()
    
    # Now you can do something with the data
    print(sampled)


if __name__ == "__main__":
    evaluate()