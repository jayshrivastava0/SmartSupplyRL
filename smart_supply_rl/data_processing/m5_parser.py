import pandas as pd
import numpy as np
import os
from pathlib import Path
from ..utils.config_loader import Config
from ..utils.logger import setup_logger

# logger = setup_logger(__name__) # Module-level logger

class M5DataParser:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(f"{self.__class__.__name__}") # Instance-specific logger
        self._calendar_df = None
        self._sell_prices_df = None
        self._sales_df_raw = None # To hold the raw sales_train_evaluation.csv

    def _load_raw_sales_data(self, use_polars: bool = False):
        """Loads the main sales data file (sales_train_evaluation.csv)."""
        if self._sales_df_raw is None:
            sales_path = self.config.get_m5_file_path('sales')
            self.logger.info(f"Loading raw sales data from {sales_path}...")
            
            id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            total_days = self.config.main_config['total_days_in_dataset']
            sales_cols = [f'd_{i}' for i in range(1, total_days + 1)]
            
            if use_polars:
                import polars as pl
                # Polars is faster for large CSVs if memory allows
                # Select specific columns with Polars
                # This requires knowing all column names if not loading all.
                # For simplicity in this refactor, using Pandas as in notebook.
                # self._sales_df_raw = pl.read_csv(sales_path, columns=id_cols + sales_cols)
                self.logger.warning("Polars loading for sales data not fully implemented here, using Pandas.")
                use_polars = False # Fallback

            if not use_polars: # Pandas path
                cols_to_load = id_cols + sales_cols
                try:
                    self._sales_df_raw = pd.read_csv(sales_path, usecols=cols_to_load)
                except FileNotFoundError:
                    self.logger.error(f"ERROR: Input CSV file not found at {sales_path}")
                    raise
                except Exception as e:
                    self.logger.error(f"An error occurred during CSV loading: {e}")
                    raise
            
            self.logger.info(f"Loaded raw sales data. Shape: {self._sales_df_raw.shape}")

    def get_product_timeseries(self, product_id_full: str) -> np.ndarray:
        """ 
        Extracts and processes the demand time series for a single product_id_full
        (e.g., FOODS_3_090_CA_3_evaluation) from the raw sales data.
        This logic is from notebook cell 35ea31cb.
        """
        if self._sales_df_raw is None:
            self._load_raw_sales_data()
        
        self.logger.info(f"Filtering for product_id: {product_id_full}")
        # Assuming self._sales_df_raw is a Pandas DataFrame
        product_df_filtered = self._sales_df_raw[self._sales_df_raw['id'] == product_id_full]

        if product_df_filtered.empty:
            self.logger.error(f"Product ID '{product_id_full}' not found in the sales data.")
            # Optionally, list some available IDs
            # self.logger.info(f"Available IDs sample: {self._sales_df_raw['id'].head().tolist()}")
            raise ValueError(f"Product ID {product_id_full} not found.")
        
        self.logger.info(f"Found product. Shape: {product_df_filtered.shape}")

        total_days = self.config.main_config['total_days_in_dataset']
        sales_cols = [f'd_{i}' for i in range(1, total_days + 1)]
        
        # Extract sales values
        # .iloc[0] selects the first (and only) row after filtering
        # .values converts the resulting Pandas Series to a NumPy array
        demand_raw = product_df_filtered[sales_cols].iloc[0].values
        self.logger.debug(f"Raw demand array shape: {demand_raw.shape}, type: {demand_raw.dtype}")

        # Handle potential NaNs and ensure numeric type
        demand_numeric = pd.to_numeric(demand_raw, errors='coerce') # Convert non-numeric to NaN
        demand_filled = np.nan_to_num(demand_numeric, nan=0.0)     # Replace NaN with 0.0
        demand_final = demand_filled.astype(np.int64)              # Convert to integer
        
        self.logger.info(f"Processed demand for {product_id_full}, shape: {demand_final.shape}, type: {demand_final.dtype}")
        return demand_final

    def split_and_save_demand(self, product_id_full: str, demand_series: np.ndarray, train_days: int):
        """
        Splits the demand data for a product into training and testing sets and saves them as .npy files.
        Logic from notebook cell 35ea31cb.
        """
        output_dir = self.config.get_path('output_dir_abs', 'processed_data')
        # output_dir.mkdir(parents=True, exist_ok=True) # Config loader handles this

        # Standardized naming convention
        output_train_path = output_dir / f"{product_id_full}_demand_train.npy"
        output_test_path = output_dir / f"{product_id_full}_demand_test.npy"
        
        self.logger.info(f"Splitting data: train ({train_days} days) and test ({len(demand_series) - train_days} days).")

        if train_days < 0 or train_days > len(demand_series):
            self.logger.error(f"Invalid train_days ({train_days}) for series length {len(demand_series)}.")
            raise ValueError("train_days is out of bounds for the demand series length.")

        if train_days == len(demand_series):
            self.logger.warning("Train days is equal to total days. Test set will be empty.")
            demand_train = demand_series
            demand_test = np.array([], dtype=demand_series.dtype)
        else:
            demand_train = demand_series[:train_days]
            demand_test = demand_series[train_days:]

        self.logger.info(f"Train data shape: {demand_train.shape}, Test data shape: {demand_test.shape}")

        np.save(output_train_path, demand_train)
        np.save(output_test_path, demand_test)
        self.logger.info(f"Saved train data to: {output_train_path}")
        self.logger.info(f"Saved test data to: {output_test_path}")
        return demand_train, demand_test

    def get_or_process_demand_splits(self, product_id_full: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads pre-split train/test demand for a product. If not found,
        it processes the raw sales data, splits, saves, and then returns the splits.
        Uses `train_days_split` from product-specific or default config.
        """
        train_days = self.config.get_train_days_for_product(product_id_full)
        
        output_dir = self.config.get_path('output_dir_abs', 'processed_data')
        train_path = output_dir / f"{product_id_full}_demand_train.npy"
        test_path = output_dir / f"{product_id_full}_demand_test.npy"

        if train_path.exists() and test_path.exists():
            self.logger.info(f"Loading preprocessed demand splits for {product_id_full} from {output_dir}")
            demand_train = np.load(train_path)
            demand_test = np.load(test_path)
            # Sanity check against configured train_days (optional)
            if len(demand_train) != train_days :
                 self.logger.warning(f"Loaded train data for {product_id_full} has length {len(demand_train)}, "
                                     f"but config expects {train_days}. Using loaded data.")
            return demand_train, demand_test
        else:
            self.logger.info(f"Processed demand splits not found for {product_id_full}. Processing now...")
            full_demand_series = self.get_product_timeseries(product_id_full)
            return self.split_and_save_demand(product_id_full, full_demand_series, train_days)

    def load_calendar(self) -> pd.DataFrame:
        """Loads and caches the calendar.csv file."""
        if self._calendar_df is None:
            calendar_path = self.config.get_m5_file_path('calendar')
            self.logger.info(f"Loading calendar data from {calendar_path}")
            try:
                self._calendar_df = pd.read_csv(calendar_path, parse_dates=['date'])
            except FileNotFoundError:
                self.logger.error(f"Calendar file not found: {calendar_path}")
                raise
            self.logger.info(f"Calendar data loaded. Shape: {self._calendar_df.shape}")
        return self._calendar_df.copy() # Return copy to prevent accidental modification

    def load_sell_prices(self) -> pd.DataFrame:
        """Loads and caches the sell_prices.csv file."""
        if self._sell_prices_df is None:
            prices_path = self.config.get_m5_file_path('prices')
            self.logger.info(f"Loading sell prices data from {prices_path}")
            try:
                self._sell_prices_df = pd.read_csv(prices_path)
            except FileNotFoundError:
                self.logger.error(f"Sell prices file not found: {prices_path}")
                raise
            self.logger.info(f"Sell prices data loaded. Shape: {self._sell_prices_df.shape}")
        return self._sell_prices_df.copy() # Return copy