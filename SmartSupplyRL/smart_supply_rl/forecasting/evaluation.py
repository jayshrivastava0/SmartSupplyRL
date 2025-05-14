from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import matplotlib.pyplot as plt
from ..utils.logger import setup_logger

logger = setup_logger(__name__) # Module-level logger

def perform_adf_test(series: pd.Series, series_name: str) -> tuple[bool, int, dict]:
    """
    Performs Augmented Dickey-Fuller test for stationarity.
    From notebook cell 53667a13.

    Args:
        series: Pandas Series containing the time series data.
        series_name: Name of the series for logging.

    Returns:
        A tuple: (is_stationary (bool), num_lags_used (int), critical_values (dict)).
    """
    logger.info(f'Performing ADF Test for: {series_name}')
    if series.empty or series.nunique() <= 1: # Handle empty or constant series
        logger.warning(f"ADF Test not performed for {series_name} due to empty or constant data.")
        return True, 0, {} # Consider constant series stationary for differencing purposes

    result = adfuller(series.dropna()) # Drop NA if differencing created them
    
    p_value = result[1]
    num_lags = result[2]
    critical_values = result[4]
    is_stationary = p_value <= 0.05

    logger.debug(f'   Test Statistic: {result[0]:.4f}')
    logger.debug(f'   p-value: {p_value:.4f}')
    logger.debug(f'   Num Lags Used: {num_lags}')
    logger.debug(f'   Critical Values:')
    for key, value in critical_values.items():
        logger.debug(f'      {key}: {value:.4f}')
    
    if is_stationary:
        logger.info(f'   Conclusion for {series_name}: Likely Stationary (Reject H0, p-value: {p_value:.4f})')
    else:
        logger.info(f'   Conclusion for {series_name}: Likely Non-Stationary (Fail to reject H0, p-value: {p_value:.4f})')
        
    return is_stationary, num_lags, critical_values

def plot_acf_pacf_summary(series: pd.Series, series_name: str, lags: int = 40, save_path_str: str = None):
    """
    Plots ACF and PACF for a given series.
    From notebook cell 53667a13. (Now uses the plotting utility for saving)
    """
    from ..utils.plotting import _save_plot # Import plotting helper
    from pathlib import Path

    if series.empty or series.nunique() <= 1:
        logger.warning(f"ACF/PACF plot not generated for {series_name} due to empty or constant data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    try:
        plot_acf(series.dropna(), ax=axes[0], lags=lags, title=f'ACF - {series_name}')
        plot_pacf(series.dropna(), ax=axes[1], lags=lags, title=f'PACF - {series_name}', method='ywm') # 'ywm' is often preferred
    except Exception as e:
        logger.error(f"Error plotting ACF/PACF for {series_name}: {e}", exc_info=True)
        plt.close(fig) # Close the figure if an error occurs
        return

    fig.suptitle(f"Autocorrelation Analysis for {series_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    
    filename = f"{series_name.replace(' ', '_').replace('/', '_')}_acf_pacf.png"
    _save_plot(fig, filename, save_path_str, output_dir=Path("output/plots")) # Example output dir
    logger.info(f"ACF/PACF plot for {series_name} generated.")