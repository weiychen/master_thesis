import os
import pandas as pd

# Settings of data to load. This is used to determine the
# file name
EPOCHS = 10
DISABLED_DP = True

def load_sampled_data():
    RESULTS_FILE_PATTERN = os.path.join("results", "sampled_{}-epochs_dp-{}.csv")
    csv_file = RESULTS_FILE_PATTERN.format(EPOCHS, not DISABLED_DP)
    
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