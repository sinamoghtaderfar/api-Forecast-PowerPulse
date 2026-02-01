import pandas as pd
from pathlib import Path


class EnergyDataService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.data_path = base_dir / "data" / "germany-energy-clean.csv"

    def load_data(self):
        df = pd.read_csv(self.data_path)

        print("Data loaded successfully")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())

        return df
