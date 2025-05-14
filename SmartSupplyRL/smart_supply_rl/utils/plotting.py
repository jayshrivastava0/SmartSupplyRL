import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
# from ..utils.config_loader import Config # If paths for saving plots are from config
# from ..utils.logger import setup_logger

# logger = setup_logger(__name__)

# Placeholder for saving plots, can be enhanced using Config
def _save_plot(fig, default_filename: str, save_path_str: str = None, output_dir: Path = None):
    if save_path_str:
        save_loc = Path(save_path_str)
    elif output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_loc = output_dir / default_filename
    else: # Don't save, just show
        plt.show()
        return

    fig.savefig(save_loc)
    # logger.info(f"Plot saved to {save_loc}")
    print(f"Plot saved to {save_loc}") # Use logger in real version
    plt.close(fig) # Close to free memory if generating many plots


def plot_forecast_vs_actual(
    ts_full_pd: pd.Series, 
    ts_test_pd: pd.Series,
    test_dates: pd.DatetimeIndex, 
    prophet_preds_test: np.ndarray,
    arima_preds_test: np.ndarray, 
    forecast_test_prophet_ci: pd.DataFrame, # DataFrame with ds, yhat_lower, yhat_upper
    product_id_full: str, 
    forecast_horizon: int, 
    rmse_prophet: float, 
    rmse_arima: float,
    train_days: int, 
    history_plot_days: int = 100, 
    save_path: str = None,
    output_plot_dir: Path = None
):
    """Plots forecasts vs actuals for the test set. From notebook cell 53667a13."""
    fig, ax = plt.subplots(figsize=(18, 7))

    # Plot historical data (last part of training for context + test data)
    # Ensure ts_full_pd is indexed by date for proper plotting
    plot_start_index = max(0, train_days - history_plot_days)
    ax.plot(ts_full_pd.index[plot_start_index:], ts_full_pd.values[plot_start_index:], 
            label='Historical Demand', color='gray', alpha=0.8)

    # Plot actual test data
    ax.plot(ts_test_pd.index, ts_test_pd.values, label='Actual Demand (Test)', color='black', linewidth=2)

    # Plot Prophet forecast
    ax.plot(test_dates, prophet_preds_test, 
            label=f'Prophet Forecast (RMSE: {rmse_prophet:.2f})', color='blue', linestyle='--')
    if forecast_test_prophet_ci is not None:
        ax.fill_between(test_dates, forecast_test_prophet_ci['yhat_lower'].values, 
                        forecast_test_prophet_ci['yhat_upper'].values, 
                        color='blue', alpha=0.1, label='Prophet 95% CI')

    # Plot ARIMA forecast
    if arima_preds_test is not None:
        ax.plot(test_dates, arima_preds_test, 
                label=f'ARIMA Forecast (RMSE: {rmse_arima:.2f})', color='red', linestyle=':')

    ax.set_title(f'Demand Forecast vs Actuals for {product_id_full} (Test: {forecast_horizon} days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    _save_plot(fig, f"{product_id_full}_forecast_vs_actual_test.png", save_path, output_plot_dir)


def plot_validation_forecast(
    train_val: pd.Series, # Validation training data
    validation_target: pd.Series, # Actuals for validation period
    prophet_preds_val: np.ndarray, 
    arima_preds_val: np.ndarray,
    product_id_full: str, 
    gap_days: int, 
    rmse_prophet_val: float, 
    rmse_arima_val: float,
    history_plot_days: int = 100, 
    save_path: str = None,
    output_plot_dir: Path = None
):
    """Plots forecasts vs actuals for the validation set. From notebook cell 53667a13."""
    fig, ax = plt.subplots(figsize=(18, 6))

    # Plot validation training data (last part for context)
    plot_start_index = max(0, len(train_val) - history_plot_days)
    ax.plot(train_val.index[plot_start_index:], train_val.values[plot_start_index:], 
            label='Validation Training Data', color='gray', alpha=0.8)
    
    # Plot validation actuals
    ax.plot(validation_target.index, validation_target.values, 
            label='Validation Actual', color='black', linewidth=2)

    # Plot Prophet validation predictions
    ax.plot(validation_target.index, prophet_preds_val, 
            label=f'Prophet Validation Pred (RMSE: {rmse_prophet_val:.2f})', color='blue', linestyle='--')

    # Plot ARIMA validation predictions
    if arima_preds_val is not None:
        ax.plot(validation_target.index, arima_preds_val, 
                label=f'ARIMA Validation Pred (RMSE: {rmse_arima_val:.2f})', color='red', linestyle=':')
    
    ax.set_title(f'Model Performance on Validation Set for {product_id_full} (Gap={gap_days} days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    _save_plot(fig, f"{product_id_full}_forecast_vs_actual_validation.png", save_path, output_plot_dir)


def plot_acf_pacf(series: pd.Series, series_name: str, lags: int = 40, save_path: str = None, output_plot_dir: Path = None):
    """Plots ACF and PACF. From notebook cell 53667a13 (evaluation.py in plan)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags, title=f'ACF - {series_name}')
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, title=f'PACF - {series_name}', method='ywm')
    fig.suptitle(f"Autocorrelation Analysis for {series_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    _save_plot(fig, f"{series_name.replace(' ', '_')}_acf_pacf.png", save_path, output_plot_dir)


def plot_rl_vs_baseline_inventory(
    test_demand: np.ndarray,
    baseline_inventory_trajectory: list,
    rl_inventory_trajectory: list,
    max_inventory: int,
    baseline_label: str, 
    rl_label: str = "DQN Agent",
    product_id_full: str = "Product", 
    save_path: str = None,
    output_plot_dir: Path = None
):
    """Plots inventory levels for RL agent vs baseline. From notebook cell 97463152."""
    fig, ax = plt.subplots(figsize=(18, 7))
    time_axis = np.arange(len(test_demand))

    ax.plot(time_axis, test_demand, label='Demand', color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.plot(time_axis, baseline_inventory_trajectory, label=f'{baseline_label} Inv', color='orange', linewidth=1.5)
    ax.plot(time_axis, rl_inventory_trajectory, label=f'{rl_label} Inv', color='blue', linewidth=1.5)
    
    ax.axhline(y=max_inventory, color='red', linestyle='--', label='Max Inventory', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_title(f'Inventory Levels: {rl_label} vs. {baseline_label} for {product_id_full}')
    ax.set_xlabel('Day in Test Period')
    ax.set_ylabel('Inventory Level')
    ax.set_ylim(bottom=min(0, np.min(baseline_inventory_trajectory)-5, np.min(rl_inventory_trajectory)-5) ) # Adjust based on actual data
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    _save_plot(fig, f"{product_id_full}_inventory_comparison.png", save_path, output_plot_dir)


def plot_rl_vs_baseline_orders(
    baseline_actions: list, # List of order quantities
    rl_actions: list,       # List of order quantities
    baseline_label: str, 
    rl_label: str = "DQN Agent",
    product_id_full: str = "Product", 
    save_path: str = None,
    output_plot_dir: Path = None
):
    """Plots orders placed over time for RL agent vs baseline. From notebook cell 97463152."""
    fig, ax = plt.subplots(figsize=(18, 5))
    time_axis = np.arange(len(baseline_actions)) # Assuming same length as rl_actions

    ax.plot(time_axis, baseline_actions, label=f'{baseline_label} Orders', color='orange', marker='.', linestyle='None', markersize=5)
    ax.plot(time_axis, rl_actions, label=f'{rl_label} Orders', color='blue', marker='x', linestyle='None', markersize=5)
    
    ax.set_title(f'Orders Placed Over Time: {rl_label} vs. {baseline_label} for {product_id_full}')
    ax.set_xlabel('Day in Test Period')
    ax.set_ylabel('Order Quantity')
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    _save_plot(fig, f"{product_id_full}_orders_comparison.png", save_path, output_plot_dir)