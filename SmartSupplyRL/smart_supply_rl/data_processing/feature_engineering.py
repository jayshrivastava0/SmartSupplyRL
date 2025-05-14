import pandas as pd
import numpy as np
from ..utils.config_loader import Config # Not strictly needed if params are passed directly
from ..utils.logger import setup_logger

# logger = setup_logger(__name__) # Module-level logger

class FeatureEngineer:
    def __init__(self, calendar_df: pd.DataFrame, sell_prices_df: pd.DataFrame):
        """
        Initializes the FeatureEngineer with necessary dataframes.
        Args:
            calendar_df: Loaded calendar.csv DataFrame.
            sell_prices_df: Loaded sell_prices.csv DataFrame.
        """
        self.calendar_df = calendar_df
        self.sell_prices_df = sell_prices_df
        self.logger = setup_logger(f"{self.__class__.__name__}")


    def _add_price_regressor(self, df: pd.DataFrame, item_id: str, store_id: str) -> pd.DataFrame:
        """Helper to add sell_price regressor to a DataFrame with 'ds' and 'wm_yr_wk'."""
        item_prices = self.sell_prices_df[
            (self.sell_prices_df['item_id'] == item_id) &
            (self.sell_prices_df['store_id'] == store_id)
        ][['wm_yr_wk', 'sell_price']].copy()

        if item_prices.empty:
            self.logger.warning(f"No price data found for {item_id}_{store_id}. 'sell_price' will be NaN.")
            df['sell_price'] = np.nan
        else:
            # Deduplicate prices if multiple entries for the same week (e.g., take mean or first)
            item_prices = item_prices.groupby('wm_yr_wk')['sell_price'].first().reset_index()
            df = pd.merge(df, item_prices, on='wm_yr_wk', how='left')
        
        # Fill missing prices (e.g., for future weeks or if item price wasn't available)
        # Notebook used ffill for future, then mean for any remaining NaNs.
        # For training data, bfill might also be needed if prices start late.
        df['sell_price'] = df['sell_price'].ffill()
        if 'y' in df.columns: # If it's training data, can also bfill
             df['sell_price'] = df['sell_price'].bfill()
        
        if df['sell_price'].isna().any():
            mean_price = df['sell_price'].mean() # Mean of available prices for this item
            if pd.isna(mean_price) and not item_prices.empty: # If all NaNs but prices exist, use overall mean from prices
                 mean_price = item_prices['sell_price'].mean()
            
            if pd.isna(mean_price): # Still NaN, maybe item has no price data at all
                self.logger.warning(f"Could not determine a fallback mean price for {item_id}_{store_id}. Filling NaNs with 0.")
                mean_price = 0.0

            df['sell_price'].fillna(mean_price, inplace=True)
            self.logger.info(f"Filled NaNs in 'sell_price' for {item_id}_{store_id} using ffill/bfill and mean ({mean_price:.2f}).")
        
        return df

    def _add_calendar_features(self, df: pd.DataFrame, include_event_regressors: bool) -> pd.DataFrame:
        """Helper to merge calendar features needed for Prophet (wm_yr_wk, events if custom)."""
        # Select necessary columns from calendar_df
        # 'wm_yr_wk' is needed for price merging.
        # Event columns are for custom regressors if `include_event_regressors` is true
        # and Prophet isn't handling holidays internally via `holidays` DataFrame.
        calendar_cols_to_merge = ['date', 'wm_yr_wk']
        if include_event_regressors:
            # Example: if you made custom dummy variables from event_name_1, event_type_1 etc.
            # For now, assume Prophet handles holidays via its own mechanism or no custom event regressors.
            # calendar_cols_to_merge.extend(['event_name_1', 'event_type_1']) 
            self.logger.info("Custom event regressors from calendar not implemented by default. "
                             "Prophet can handle holidays internally if a holidays_df is provided at init.")
            pass

        calendar_subset = self.calendar_df[calendar_cols_to_merge].copy()
        # Ensure 'ds' column exists in df for merging with 'date'
        if 'ds' not in df.columns:
            self.logger.error("'ds' column missing from input DataFrame for calendar feature merge.")
            return df # Or raise error

        df = pd.merge(df, calendar_subset, left_on='ds', right_on='date', how='left')
        if 'date' in df.columns and 'ds' in df.columns and df['date'].equals(df['ds']): # Remove duplicate date column
             df = df.drop(columns=['date'])
        
        return df


    def prepare_prophet_train_df(self,
                                 demand_series: pd.Series, # Indexed by date ('ds'), values are 'y'
                                 item_id: str, # e.g., FOODS_3_090
                                 store_id: str, # e.g., CA_3
                                 prophet_model_params: dict # From config
                                ) -> pd.DataFrame:
        """ 
        Prepares the DataFrame for training Prophet (ds, y, and any regressors).
        Logic adapted from notebook cell 53667a13 (Prophet model training section)
        and cell 158e8861 (regressor preparation logic).
        """
        self.logger.info(f"Preparing training DataFrame for Prophet: {item_id}_{store_id}")
        
        train_df = pd.DataFrame({'ds': demand_series.index, 'y': demand_series.values})

        # Add calendar features like wm_yr_wk (needed for price) or custom event features
        # prophet_model_params can indicate if custom event regressors are to be built
        use_custom_event_regressors = prophet_model_params.get('regressors', {}).get('custom_events', False)
        train_df = self._add_calendar_features(train_df, include_event_regressors=use_custom_event_regressors)

        # Add price regressor if specified in params
        if prophet_model_params.get('regressors', {}).get('price'):
            if 'wm_yr_wk' not in train_df.columns:
                self.logger.error("Cannot add price regressor: 'wm_yr_wk' column is missing. Ensure calendar features were added.")
            else:
                train_df = self._add_price_regressor(train_df, item_id, store_id)
                self.logger.info("Added 'sell_price' regressor to training data.")
        
        # Select final columns: 'ds', 'y', and any explicitly added regressor columns
        final_cols = ['ds', 'y']
        if 'sell_price' in train_df.columns and prophet_model_params.get('regressors', {}).get('price'):
            final_cols.append('sell_price')
        # Add other custom regressor column names here if they were created

        train_df_final = train_df[final_cols].copy()
        
        # Check for NaNs in critical columns
        if train_df_final['ds'].isna().any() or train_df_final['y'].isna().any():
            self.logger.error("NaNs found in 'ds' or 'y' columns of training data. This is critical.")
            raise ValueError("NaNs in 'ds' or 'y' for Prophet training data.")
        if 'sell_price' in train_df_final and train_df_final['sell_price'].isna().any():
            self.logger.error("NaNs found in 'sell_price' regressor column of training data. This is critical.")
            raise ValueError("NaNs in 'sell_price' regressor for Prophet training data.")

        self.logger.info(f"Prepared training DataFrame for Prophet for {item_id}_{store_id}. Shape: {train_df_final.shape}, Columns: {train_df_final.columns.tolist()}")
        return train_df_final


    def create_prophet_future_df(self,
                                 prediction_dates: pd.DatetimeIndex, # Future dates to predict
                                 item_id: str,
                                 store_id: str,
                                 prophet_model_params: dict # From config
                                ) -> pd.DataFrame:
        """
        Creates the 'future' DataFrame with 'ds' and necessary regressor columns for Prophet prediction.
        Logic from notebook cell 158e8861.
        """
        self.logger.info(f"Creating future DataFrame for Prophet prediction: {item_id}_{store_id}")
        future_df = pd.DataFrame({'ds': prediction_dates})

        # Add calendar features (wm_yr_wk for price, custom events if any)
        use_custom_event_regressors = prophet_model_params.get('regressors', {}).get('custom_events', False)
        future_df = self._add_calendar_features(future_df, include_event_regressors=use_custom_event_regressors)

        # Add price regressor if specified
        if prophet_model_params.get('regressors', {}).get('price'):
            if 'wm_yr_wk' not in future_df.columns:
                self.logger.error("Cannot add price regressor to future_df: 'wm_yr_wk' is missing.")
            else:
                future_df = self._add_price_regressor(future_df, item_id, store_id)
                self.logger.info("Added 'sell_price' regressor to future data.")

        # Select final columns: 'ds' and any regressor columns model was trained with
        final_cols = ['ds']
        if 'sell_price' in future_df.columns and prophet_model_params.get('regressors', {}).get('price'):
            final_cols.append('sell_price')
        # Add other custom regressor column names here

        future_df_final = future_df[final_cols].copy()

        # Check for NaNs in regressor columns (critical for prediction)
        if 'sell_price' in future_df_final and future_df_final['sell_price'].isna().any():
            self.logger.error("NaNs found in 'sell_price' regressor column of future data. Prediction might fail or be inaccurate.")
            # Depending on strictness, could raise error or just warn.
            # Prophet might handle NaNs in regressors depending on version/settings, but best to avoid.
            # raise ValueError("NaNs in 'sell_price' regressor for Prophet future data.")

        self.logger.info(f"Created future DataFrame for Prophet. Shape: {future_df_final.shape}, Columns: {future_df_final.columns.tolist()}")
        return future_df_final