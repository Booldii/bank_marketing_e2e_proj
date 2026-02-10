import pandas as pd
from sklearn.datasets import fetch_openml

def fetch_dataset():
    dataset = fetch_openml(data_id=1461)
    df = dataset.frame
    return df

