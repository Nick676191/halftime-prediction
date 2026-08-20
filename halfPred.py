from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv
import pandas as pd

# Have to load the environment variable prior to loading the kagglehub and kaggle libraries in python
load_dotenv()
import kagglehub
import kaggle
kaggle.api.authenticate()

# kagglehub.login()

# Load a DataFrame with a specific version of a CSV
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "saife245/english-premier-league",
    "final_dataset.csv"
)

print(df.head())
print(df.shape)