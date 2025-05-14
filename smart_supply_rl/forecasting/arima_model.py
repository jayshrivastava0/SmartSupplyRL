from pmdarima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA # If using manual ARIMA
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .base_forecaster import BaseForecaster
from .evaluation import perform_adf_test # Import from evaluation module

class ARIMAForecaster(BaseForecaster):
    def _get_model_save_path(self) -> Path:
        model_dir = self.config.get_path('output_dir_abs', 'models', 'forecasting')
        return model_dir / f"arima_model_{self.product_id_full}.joblib"

    def train(self, train_df: pd.DataFrame, holidays_df: pd.DataFrame = None): # train_df has 'ds', 'y'
        self.logger.info(f"Training ARIMA model using auto_arima for {self.product_id_full}...")
        if 'y' not in train_df.columns:
            self.logger.error("Training DataFrame must contain 'y' column for ARIMA.")
            raise ValueError("Missing 'y' in train_df for ARIMA.")

        y_train = train_df['y'].values

        # Determine 'd' using ADF test from self.model_specific_params or dynamic
        # This matches logic from notebook cell 53667a13
        use_adf_d = self.model_specific_params.get('determine_d_adf', True)
        arima_d_order = self.model_specific_params.get('d_order', None) # Allow fixed d

        if use_adf_d and arima_d_order is None:
            is_stationary_orig, _, _ = perform_adf_test(train_df['y'], "Training Data (Original)")
            if is_stationary_orig:
                arima_d_order = 0
            else:
                is_stationary_diff1, _, _ = perform_adf_test(train_df['y'].diff().dropna(), "Training Data (1st Diff)")
                if is_stationary_diff1:
                    arima_d_order = 1
                else:
                    # Could test 2nd diff, or default to 2 as in notebook
                    is_stationary_diff2, _, _ = perform_adf_test(train_df['y'].diff().diff().dropna(), "Training Data (2nd Diff)")
                    arima_d_order = 2 if is_stationary_diff2 else 1 # Fallback to 1 if 2nd still not stationary
                    self.logger.info(f"ADF Test on 2nd diff stationary: {is_stationary_diff2}")
            self.logger.info(f"Auto-determined differencing order (d) for ARIMA: {arima_d_order}")
        elif arima_d_order is None: # Not using ADF and no d_order provided
            arima_d_order = 1 # Default d_order
            self.logger.info(f"Using default differencing order (d): {arima_d_order}")
        else:
            self.logger.info(f"Using pre-configured differencing order (d): {arima_d_order}")


        # Exogenous variables (regressors) for ARIMA
        # auto_arima uses `exogenous`. statsmodels.tsa.arima.model.ARIMA uses `exog`.
        # For now, no exogenous variables for ARIMA to match notebook simplicity.
        # If used, train_df would need these columns, and predict would need future values.
        exog_train = None
        if self.model_specific_params.get('regressors', {}).get('price') and 'sell_price' in train_df.columns:
            # exog_train = train_df[['sell_price']].values # pmdarima expects 2D array for exog
            self.logger.warning("Exogenous variables for ARIMA (e.g. price) are not fully implemented in this example. Training without.")
            # TODO: If using exog, ensure future_df in predict also gets them.
            pass

        try:
            self.model = auto_arima(
                y_train,
                exogenous=exog_train,
                start_p=self.model_specific_params.get('start_p', 1),
                start_q=self.model_specific_params.get('start_q', 1),
                max_p=self.model_specific_params.get('max_p', 5),
                max_q=self.model_specific_params.get('max_q', 5),
                d=arima_d_order,
                seasonal=self.model_specific_params.get('seasonal', False),
                m=self.model_specific_params.get('m', 1), # Seasonal period (e.g., 7 for weekly)
                test=self.model_specific_params.get('test', 'adf'), # ADF test for seasonality if seasonal=True
                stepwise=self.model_specific_params.get('stepwise', True), # Faster search
                suppress_warnings=True,
                error_action='ignore' # Skip models that fail to fit
            )
            self.logger.info(f"Auto ARIMA Best Order: {self.model.order}, Seasonal Order: {self.model.seasonal_order}")
            self.save_model()
        except Exception as e:
            self.logger.error(f"Auto ARIMA training failed: {e}. Model not trained.", exc_info=True)
            self.model = None
            raise # Re-raise

    def predict(self, future_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
        if self.model is None:
            self.logger.error("ARIMA model not trained or loaded. Call train() or load_model() first.")
            return np.array([]), None
        
        n_periods = len(future_df)
        self.logger.info(f"Generating ARIMA predictions for {n_periods} periods...")

        exog_future = None
        # if self.model_specific_params.get('regressors', {}).get('price') and 'sell_price' in future_df.columns:
            # exog_future = future_df[['sell_price']].values
            # self.logger.warning("Exogenous variables for ARIMA prediction not fully implemented.")

        try:
            predictions, conf_int = self.model.predict(
                n_periods=n_periods, 
                exogenous=exog_future,
                return_conf_int=True,
                alpha=0.05 # For 95% confidence interval
            )
        except Exception as e:
            self.logger.error(f"Exception during ARIMA model prediction: {e}", exc_info=True)
            return np.array([]), None

        predictions[predictions < 0] = 0 # Ensure non-negative sales
        self.logger.info("ARIMA predictions generated.")
        
        # Create a CI DataFrame similar to Prophet's output
        ci_df = pd.DataFrame(conf_int, columns=['yhat_lower', 'yhat_upper'])
        ci_df['ds'] = future_df['ds'].values # Add ds column
        ci_df['yhat_lower'][ci_df['yhat_lower'] < 0] = 0
        ci_df['yhat_upper'][ci_df['yhat_upper'] < 0] = 0

        return predictions, ci_df

    def save_model(self):
        if self.model is None:
            self.logger.error("No model to save (ARIMA).")
            return
        try:
            joblib.dump(self.model, self.model_save_path)
            self.logger.info(f"ARIMA model saved to {self.model_save_path}")
        except Exception as e:
            self.logger.error(f"Error saving ARIMA model: {e}", exc_info=True)

    def load_model(self) -> bool:
        if not self.model_save_path.exists():
            self.logger.warning(f"ARIMA model file not found: {self.model_save_path}")
            self.model = None
            return False
        try:
            self.model = joblib.load(self.model_save_path)
            self.logger.info(f"ARIMA model loaded from {self.model_save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ARIMA model: {e}", exc_info=True)
            self.model = None
            return False