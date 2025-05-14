from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .base_forecaster import BaseForecaster
# Config import not needed if passed to __init__

class ProphetForecaster(BaseForecaster):
    def _get_model_save_path(self) -> Path:
        model_dir = self.config.get_path('output_dir_abs', 'models', 'forecasting')
        # model_dir.mkdir(parents=True, exist_ok=True) # Handled by Config loader
        # Standardized naming: prophet_model_PRODUCTIDFULL.joblib
        return model_dir / f"prophet_model_{self.product_id_full}.joblib"

    def train(self, train_df: pd.DataFrame, holidays_df: pd.DataFrame = None):
        self.logger.info(f"Training Prophet model for {self.product_id_full}...")
        if 'ds' not in train_df.columns or 'y' not in train_df.columns:
            self.logger.error("Training DataFrame must contain 'ds' and 'y' columns.")
            raise ValueError("Missing 'ds' or 'y' in train_df for Prophet.")

        # model_specific_params from config drives seasonality, holidays, regressors
        prophet_init_params = {
            'yearly_seasonality': self.model_specific_params.get('seasonality', {}).get('yearly', True),
            'weekly_seasonality': self.model_specific_params.get('seasonality', {}).get('weekly', True),
            'daily_seasonality': self.model_specific_params.get('seasonality', {}).get('daily', False),
            'holidays': holidays_df if self.model_specific_params.get('holidays') and holidays_df is not None else None,
            'growth': self.model_specific_params.get('growth', 'linear'),
            'changepoint_prior_scale': self.model_specific_params.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': self.model_specific_params.get('seasonality_prior_scale', 10.0),
            'holidays_prior_scale': self.model_specific_params.get('holidays_prior_scale', 10.0),
            # Add other Prophet parameters as needed from self.model_specific_params
        }
        self.logger.debug(f"Prophet initialization parameters: {prophet_init_params}")
        
        self.model = Prophet(**prophet_init_params)

        # Add regressors if specified and present in train_df
        if self.model_specific_params.get('regressors', {}).get('price') and 'sell_price' in train_df.columns:
            if train_df['sell_price'].isnull().any():
                self.logger.warning("NaNs found in 'sell_price' regressor column during training. Prophet might error or produce unexpected results.")
            self.model.add_regressor('sell_price')
            self.logger.info("Added 'sell_price' regressor to Prophet model.")
        # Add other custom event regressors if they were engineered into train_df

        try:
            self.model.fit(train_df)
            self.logger.info("Prophet model training complete.")
            self.save_model()
        except Exception as e:
            self.logger.error(f"Exception during Prophet model fitting: {e}", exc_info=True)
            self.model = None # Ensure model is None if fit failed
            raise # Re-raise the exception after logging

    def predict(self, future_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
        if self.model is None:
            self.logger.error("Model not trained or loaded. Call train() or load_model() first.")
            return np.array([]), None # Return empty array and None for CI
        if 'ds' not in future_df.columns:
            self.logger.error("Future DataFrame must contain 'ds' column.")
            raise ValueError("Missing 'ds' in future_df for Prophet prediction.")

        self.logger.info(f"Generating Prophet predictions for {len(future_df)} periods...")
        
        # Ensure regressor columns match training if they exist
        if self.model_specific_params.get('regressors', {}).get('price') and 'sell_price' not in future_df.columns:
            self.logger.error("Price regressor was used in training, but 'sell_price' column is missing in future_df.")
            raise ValueError("'sell_price' regressor column missing in future_df.")
        if self.model_specific_params.get('regressors', {}).get('price') and 'sell_price' in future_df.columns and future_df['sell_price'].isnull().any():
             self.logger.warning("NaNs found in 'sell_price' regressor column during prediction. Prophet might error or produce unexpected results.")


        try:
            forecast_df = self.model.predict(future_df)
        except Exception as e:
            self.logger.error(f"Exception during Prophet model prediction: {e}", exc_info=True)
            return np.array([]), None
            
        predictions = forecast_df['yhat'].values
        predictions[predictions < 0] = 0 # Ensure non-negative sales
        
        self.logger.info("Prophet predictions generated.")
        # Return predictions and confidence interval DataFrame
        ci_df = forecast_df[['ds', 'yhat_lower', 'yhat_upper']].copy()
        ci_df['yhat_lower'][ci_df['yhat_lower'] < 0] = 0
        ci_df['yhat_upper'][ci_df['yhat_upper'] < 0] = 0 # Should not happen if yhat is >=0

        return predictions, ci_df

    def save_model(self):
        if self.model is None:
            self.logger.error("No model to save (Prophet).")
            return
        try:
            # self.model_save_path is defined in BaseForecaster
            joblib.dump(self.model, self.model_save_path)
            self.logger.info(f"Prophet model saved to {self.model_save_path}")
        except Exception as e:
            self.logger.error(f"Error saving Prophet model: {e}", exc_info=True)

    def load_model(self) -> bool:
        if not self.model_save_path.exists():
            self.logger.warning(f"Prophet model file not found: {self.model_save_path}")
            self.model = None
            return False
        try:
            self.model = joblib.load(self.model_save_path)
            self.logger.info(f"Prophet model loaded from {self.model_save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Prophet model: {e}", exc_info=True)
            self.model = None
            return False