
import pandas as pd
import seaborn as sns


def load_data(filepath="ames_house_data.csv"):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)


df = load_data()

print(df.head())