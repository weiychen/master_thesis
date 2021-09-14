import os
import pandas as pd

# Settings of data to load. This is used to determine the
# file name
EPOCHS = 500
DISABLED_DP = False
dataset = (
    'datasets/ETM_Configuration2.xes',
    # 'datasets/financial_log.xes',
)[0]

def load_sampled_data():
    dataset_name = os.path.basename(dataset).split(".")[0]
    RESULTS_FILE_PATTERN = os.path.join("results", "{}_sampled_{}-epochs_dp-{}.csv")
    csv_file = RESULTS_FILE_PATTERN.format(dataset_name, EPOCHS, not DISABLED_DP)
    
    try:
        return pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        print("File not found:", csv_file)


def evaluate():
    sampled = load_sampled_data()
    
    # Now you can do something with the data
    print(sampled)


if __name__ == "__main__":
    evaluate()