import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from utils.config import load_config

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

class NYCCrimePreprocessor:
    def __init__(self, config_filename):
        self.config = project_root / config_filename
        self.config_data = load_config(self.config)  # Load yaml config once and store it

    def data_to_pandas_df(self):
        path = project_root / self.config_data.get("raw_path")
        return pd.read_csv(path)

    def select_and_clean_spacetime_columns(self, df):
        df_spacetime_cols = df[self.config_data.get("columns_to_keep")]
        df_spacetime_cols = df_spacetime_cols.dropna()
        date_col, time_col = self.config_data.get("columns_to_keep")[:2]  # first two are date and time
        df_spacetime_cols['datetime'] = pd.to_datetime(df_spacetime_cols[date_col] + ' ' +
                                    df_spacetime_cols[time_col], format=self.config_data.get('datetime_format'),
                                    errors='coerce')
        no_null_df = df_spacetime_cols.dropna()
        sorted_df = no_null_df.sort_values(by="datetime").reset_index(drop=True)
        return sorted_df

    def remove_outer_coordinates(self, df):
        geocoords_bounds = self.config_data.get("geo_bounds")
        df_spacetime_cols = self.config_data.get("columns_to_keep")
        latitude_col, longitude_col = df_spacetime_cols[2:]

        bounded_df = df[
            (df[latitude_col] >= geocoords_bounds["lat_min"]) &
            (df[latitude_col] <= geocoords_bounds["lat_max"]) &
            (df[longitude_col] >= geocoords_bounds["lon_min"]) &
            (df[longitude_col] <= geocoords_bounds["lon_max"])
        ]
        return bounded_df

    def df_included_grid(self, df):
        GRID_SIZE = self.config_data.get("grid_size")
        NYC_contours = self.config_data.get("geo_bounds")
        lat_bins = np.linspace(NYC_contours['lat_min'], NYC_contours['lat_max'], GRID_SIZE + 1)
        lon_bins = np.linspace(NYC_contours['lon_min'], NYC_contours['lon_max'], GRID_SIZE + 1)
        df_spacetime_cols = self.config_data.get("columns_to_keep")
        latitude_col, longitude_col = df_spacetime_cols[2:]
        df['lat_bin'] = pd.cut(df[latitude_col], bins=lat_bins, labels=False)
        df['lon_bin'] = pd.cut(df[longitude_col], bins=lon_bins, labels=False)
        return df

    def generate_tensor(self, df):
        df['day'] = df['datetime'].dt.floor(self.config_data.get("tensor_frequency"))
        crime_cells = df[['day', 'lat_bin', 'lon_bin']].drop_duplicates()
        days = sorted(crime_cells['day'].unique())
        time_index = {d: i for i, d in enumerate(days)}
        tensor = np.zeros((len(days), self.config_data.get("grid_size"),
                           self.config_data.get("grid_size")))
        for _, row in crime_cells.iterrows():
            t = time_index[row['day']]
            x = int(row['lat_bin'])
            y = int(row['lon_bin'])
            tensor[t, x, y] = 1
        return tensor

