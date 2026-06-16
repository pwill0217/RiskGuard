import pandas as pd 
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'/"Loan_Default.csv"

#below is the function to load the data from the specified path and return a pandas DataFrame. It also checks if the file exists before attempting to load it, and raises a FileNotFoundError if it does not.
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df

#print the first few rows of the DataFrame, the summary information about the DataFrame, and the count of missing values in each column to verify that the data has been loaded correctly and to understand its structure.
if __name__ == "__main__":
    df = load_data()
    print(df.head()) # this basically means to stop after the first 5 rows of the data frame
    print(df.info()) # this will give us a summary of the data frame, including the number of non-null entries in each column and the data types of each column
    print(df.isnull().sum())