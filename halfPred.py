from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Have to load the environment variable prior to loading the kagglehub and kaggle libraries in python
load_dotenv()
import kagglehub
import kaggle
kaggle.api.authenticate()

# Setting script variables
ACCT_STRING = "saife245/english-premier-league"
DF_STRING = "final_dataset.csv"

def kaggleDfLoader(acctStr: str, dfStr: str):
    # Load a DataFrame with a specific version of a CSV
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        acctStr,
        dfStr
    )

    return df

def dfDescriber(df) -> None:
    print(df.head())
    print("-"*60)
    print()

    shape = df.shape
    print(f"This dataframe has {shape[0]} rows and {shape[1]} columns.")
    print("-"*60)
    print()

    print(df.isna().sum())
    print("-"*60)
    print()

    print(df.dtypes)
    print("-"*60)
    print()

    for dtype in df.dtypes.drop_duplicates():
        print(f"There are {sum([1 if dt==dtype else 0 for dt in df.dtypes])} columns of the {dtype} data-type")
    print("-"*60)
    print()


    

def main():
    # loading descriptive statistics
    df = kaggleDfLoader(ACCT_STRING, DF_STRING)
    dfDescriber(df)

    # looking into the string columns


if __name__ == "__main__":
    main()