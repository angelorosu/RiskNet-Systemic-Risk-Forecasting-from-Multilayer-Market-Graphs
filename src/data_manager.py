import os
import pandas as pd
import numpy as np

class DataManager:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.sectors_data = {}  # Dictionary to hold DataFrames for each sector
        self.industry_names = [
            'basic_industries', 'capital_goods', 'consumer_durables',
            'consumer_non_durables', 'energy', 'equity_final_finance',
            'finance', 'health_care', 'miscellaneous', 'public_utilities',
            'technology', 'transportation'
        ]
        self.data = pd.DataFrame()         # Merged DataFrame with all tickers
        self.sector_mapping = pd.DataFrame() # DataFrame mapping each ticker to its sector
        self.log_returns = pd.DataFrame()
        self.training_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.training_data_percent = 0.8

    def read_all_files(self):
        """
        Reads all CSV files for each sector from the data_directory.
        Each file is expected to be named as <sector>.csv.
        Renames ticker columns to include the sector name to ensure uniqueness.
        """
        for industry in self.industry_names:
            file_path = os.path.join(self.data_directory, f'{industry}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Convert 'Date' column to datetime and set as index.
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                # Rename columns to ensure uniqueness.
                new_cols = [f"{industry}_{col}" for col in df.columns]
                df.columns = new_cols
                self.sectors_data[industry] = df
            else:
                print(f"Warning: {file_path} does not exist.")

    def check_date_range(self, verbose=False):
        """
        Checks if all sector dataframes have the same date range.
        Returns True if they are consistent; otherwise, False.
        """
        date_ranges = {}
        for industry, df in self.sectors_data.items():
            date_ranges[industry] = (df.index.min(), df.index.max())
        if verbose:
            print("Date ranges per sector:")
            for industry, dr in date_ranges.items():
                print(f"{industry}: {dr}")
        if len(set(date_ranges.values())) == 1:
            return True
        else:
            if verbose:
                print("Date ranges are not consistent across sectors.")
            return False

    def fix_date_range(self):
        """
        Fixes date ranges for all sector dataframes to the common overlapping period.
        """
        min_dates = [df.index.min() for df in self.sectors_data.values()]
        max_dates = [df.index.max() for df in self.sectors_data.values()]
        common_start = max(min_dates)
        common_end = min(max_dates)
        for industry, df in self.sectors_data.items():
            df_fixed = df.loc[(df.index >= common_start) & (df.index <= common_end)].copy()
            df_fixed.dropna(inplace=True)
            df_fixed.sort_index(inplace=True)
            self.sectors_data[industry] = df_fixed
        print(f"Date range fixed to: {common_start} - {common_end}")

    def merge_dataframes(self):
        """
        Merges all sector dataframes on their Date index to form a single DataFrame.
        """
        if self.sectors_data:
            self.data = pd.concat(self.sectors_data.values(), axis=1, join='inner')
            self.data.dropna(inplace=True)
        else:
            print("No sector data available to merge.")

    def map_sectors(self):
        """
        Creates a DataFrame mapping each ticker to its corresponding sector.
        """
        mapping_list = []
        for industry, df in self.sectors_data.items():
            for ticker in df.columns:
                mapping_list.append({'Ticker': ticker, 'Sector': industry})
        self.sector_mapping = pd.DataFrame(mapping_list)

    def compute_log_returns(self):
        """
        Computes log returns for the merged data.
        Log return = log(price_t) - log(price_{t-1})
        """
        if not self.data.empty:
            self.log_returns = np.log(self.data).diff().dropna()
        else:
            print("Merged data is empty. Cannot compute log returns.")

    def split_data(self):
        """
        Splits the log returns data into training and test sets based on training_data_percent.
        """
        if not self.log_returns.empty:
            split_index = int(len(self.log_returns) * self.training_data_percent)
            self.training_data = self.log_returns.iloc[:split_index]
            self.test_data = self.log_returns.iloc[split_index:]
        else:
            print("Log returns data is empty. Cannot split data.")

    def load_data(self):
        """
        Loads the data by reading CSV files, ensuring date ranges are aligned,
        merging the sector data, and mapping tickers to sectors.
        """
        self.read_all_files()
        if not self.check_date_range(verbose=True):
            self.fix_date_range()
        self.merge_dataframes()
        self.map_sectors()

    def get_data(self):
        """
        Returns a dictionary containing the processed data components.
        """
        return {
            'raw_data': self.data,
            'log_returns': self.log_returns,
            'sector_mapping': self.sector_mapping,
            'training_data': self.training_data,
            'test_data': self.test_data,
            'tickers': list(self.data.columns) if not self.data.empty else []
        }

    def run_pipeline(self):
        """
        Executes the full data management pipeline.
        """
        self.load_data()
        self.compute_log_returns()
        self.split_data()
        print("Data management pipeline executed successfully.")
