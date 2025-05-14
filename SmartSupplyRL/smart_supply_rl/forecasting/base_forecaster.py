from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.config_loader import Config
from ..utils.logger import setup_logger

class BaseForecaster(ABC):
    def __init__(self, product_id_full: str, config: Config, model_specific_params: dict):
        """
        Abstract base class for forecasting models.

        Args:
            product_id_full: The full product identifier (e.g., FOODS_3_090_CA_3_evaluation).
            config: The main configuration object.
            model_specific_params: Dictionary of parameters specific to the forecasting model (e.g., Prophet's seasonality).
        """
        self.product_id_full = product_id_full
        self.config = config
        self.model_specific_params = model_specific_params
        self.model = None # This will hold the trained model instance
        self.logger = setup_logger(f"{self.__class__.__name__}-{self.product_id_full.replace('_evaluation', '')}")
        self.model_save_path = self._get_model_save_path()

    @abstractmethod
    def _get_model_save_path(self) -> Path:
        """Determines the save path for the model file."""
        pass

    @abstractmethod
    def train(self, train_df: pd.DataFrame, holidays_df: pd.DataFrame = None):
        """
        Trains the forecasting model.
        Args:
            train_df: DataFrame containing 'ds' (datetime), 'y' (target), and any regressors.
            holidays_df: Optional DataFrame for holidays (specific to models like Prophet).
        """
        pass

    @abstractmethod
    def predict(self, future_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
        """
        Generates predictions for future dates.
        Args:
            future_df: DataFrame with 'ds' and any required regressor columns.
        Returns:
            A tuple containing:
                - NumPy array of predictions ('yhat').
                - Optional: DataFrame with confidence intervals (e.g., 'ds', 'yhat_lower', 'yhat_upper').
                  Returns None if the model doesn't provide CI in this format.
        """
        pass

    @abstractmethod
    def save_model(self):
        """Saves the trained model to the path defined by _get_model_save_path()."""
        pass

    @abstractmethod
    def load_model(self) -> bool:
        """
        Loads a pre-trained model from the path defined by _get_model_save_path().
        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        pass

    def evaluate(self, actuals: np.ndarray, predictions: np.ndarray) -> dict:
        """
        Evaluates the model's predictions against actual values.
        Args:
            actuals: NumPy array of actual target values.
            predictions: NumPy array of predicted values.
        Returns:
            Dictionary with evaluation metrics (e.g., MAE, RMSE).
        """
        if len(actuals) != len(predictions):
            self.logger.error(f"Length mismatch for evaluation: actuals ({len(actuals)}), predictions ({len(predictions)})")
            return {"error": "Length mismatch"}
        if len(actuals) == 0:
            self.logger.warning("Cannot evaluate on empty actuals/predictions array.")
            return {"mae": np.nan, "rmse": np.nan}

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Ensure no NaNs which would break sklearn metrics
        mask = ~np.isnan(actuals) & ~np.isnan(predictions)
        if not np.all(mask):
            self.logger.warning(f"NaNs found in actuals or predictions during evaluation. Evaluating on {np.sum(mask)} non-NaN pairs.")
            actuals_clean = actuals[mask]
            predictions_clean = predictions[mask]
            if len(actuals_clean) == 0:
                self.logger.warning("No non-NaN pairs left after cleaning for evaluation.")
                return {"mae": np.nan, "rmse": np.nan}
        else:
            actuals_clean = actuals
            predictions_clean = predictions

        mae = mean_absolute_error(actuals_clean, predictions_clean)
        rmse = np.sqrt(mean_squared_error(actuals_clean, predictions_clean))
        
        self.logger.info(f"Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f} (on {len(actuals_clean)} points)")
        return {"mae": mae, "rmse": rmse, "eval_points": len(actuals_clean)}
            
    def get_holidays_df(self, calendar_df: pd.DataFrame) -> pd.DataFrame | None:
        """ 
        Prepares a holidays DataFrame suitable for Prophet, based on calendar events.
        This is from notebook cell 53667a13.
        """
        if not self.model_specific_params.get('holidays', False): # Check if model configured to use holidays
            self.logger.info("Holiday processing skipped as per model_specific_params.")
            return None
        
        holidays = calendar_df[
            calendar_df['event_name_1'].notna() | calendar_df['event_name_2'].notna()
        ][['date', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']].copy()
        
        holidays['event_name_1'] = holidays['event_name_1'].fillna('')
        holidays['event_name_2'] = holidays['event_name_2'].fillna('')
        
        # Combine event names; ensure no leading/trailing spaces affect uniqueness if one is empty
        holidays['holiday'] = (holidays['event_name_1'].astype(str) + " " + holidays['event_name_2'].astype(str)).str.strip()
        
        # Filter out rows where the combined holiday name is empty (i.e., both events were originally NaN)
        holidays = holidays[holidays['holiday'] != ''][['date', 'holiday']].rename(columns={'date': 'ds'})
        
        # Prophet requires lower_window and upper_window for holidays
        holidays['lower_window'] = 0 # Point events (can be configured)
        holidays['upper_window'] = 0
        
        holidays.drop_duplicates(subset=['ds', 'holiday'], inplace=True) # Handle cases where a date might have multiple identical combined events listed
        
        self.logger.info(f"Prepared {len(holidays)} unique holiday/event instances for Prophet.")
        return holidays if not holidays.empty else None