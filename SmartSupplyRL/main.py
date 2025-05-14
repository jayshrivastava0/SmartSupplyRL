# main.py
import argparse
import sys
import os
import numpy as np
import pandas as pd # For type hints or direct use in command functions

# Ensure the smart_supply_rl package is discoverable
# This assumes main.py is in the project root, and smart_supply_rl is a subdirectory.
# If your CWD is already the project root when you run `python main.py`,
# this sys.path modification might not be strictly necessary, but it's good practice.
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except NameError:
    # __file__ is not defined, happens in some interactive environments
    # Fallback assuming CWD is project root
    PROJECT_ROOT = os.getcwd()
    if PROJECT_ROOT not in sys.path:
         sys.path.insert(0, PROJECT_ROOT)


# Import project modules
from smart_supply_rl.utils.config_loader import Config
from smart_supply_rl.utils.logger import setup_logger
from smart_supply_rl.data_processing.m5_parser import M5DataParser
from smart_supply_rl.data_processing.feature_engineering import FeatureEngineer
from smart_supply_rl.forecasting.prophet_model import ProphetForecaster
from smart_supply_rl.forecasting.arima_model import ARIMAForecaster
from smart_supply_rl.forecasting.evaluation import plot_acf_pacf_summary, perform_adf_test
from smart_supply_rl.rl_agents.dqn_agent import DQNAgent
from smart_supply_rl.baselines.heuristic_policies import run_baseline_policy
from smart_supply_rl.api.recommendation_service import RecommendationService # For get_recommendation
from smart_supply_rl.utils import plotting # For evaluation plots

# Initialize configuration and logger globally for the main script
try:
    config = Config()
    logger = setup_logger("SmartSupplyPipeline", config_instance=config)
except Exception as e:
    print(f"CRITICAL: Failed to initialize Config or Logger: {e}", file=sys.stderr)
    print("Ensure configuration files (main_config.yaml, etc.) are present in the 'config' directory "
          "and paths within them are correct relative to the project root.", file=sys.stderr)
    sys.exit(1)

# --- Command Functions ---

def cmd_preprocess_data(args):
    """Handles the 'preprocess_data' command."""
    logger.info(f"Running data preprocessing for product: {args.product_id_full}")
    parser = M5DataParser(config)
    # Determine train_days: use arg if provided, else from config
    train_days = args.train_days if args.train_days is not None else config.get_train_days_for_product(args.product_id_full)
    
    try:
        full_demand_series = parser.get_product_timeseries(args.product_id_full)
        if full_demand_series.size == 0:
            logger.error(f"No demand data found or processed for {args.product_id_full}.")
            return

        train_demand, test_demand = parser.split_and_save_demand(args.product_id_full, full_demand_series, train_days)
        logger.info(f"Data preprocessing complete for {args.product_id_full}. Train shape: {train_demand.shape}, Test shape: {test_demand.shape}")
    except ValueError as ve: # Catch specific error from get_product_timeseries if product_id not found
        logger.error(f"Error during preprocessing for {args.product_id_full}: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing for {args.product_id_full}: {e}", exc_info=True)


def cmd_analyze_timeseries(args):
    """Handles the 'analyze_timeseries' command."""
    logger.info(f"Analyzing time series for product: {args.product_id_full}")
    parser = M5DataParser(config)
    
    try:
        # This will load or process+load the splits based on config
        train_demand_arr, _ = parser.get_or_process_demand_splits(args.product_id_full)

        if train_demand_arr.size == 0:
            logger.error(f"Train demand data is empty for {args.product_id_full}. Cannot analyze.")
            return
            
        train_demand_series = pd.Series(train_demand_arr)
        
        # Define where to save plots
        plot_output_dir = config.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
        plot_output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        logger.info("--- Original Series Analysis ---")
        perform_adf_test(train_demand_series, f"{args.product_id_full} Original")
        # Note: plot_acf_pacf_summary in evaluation.py saves the plot.
        # Pass the directory string to save_path_str or handle Path object if modified.
        plot_acf_pacf_summary(train_demand_series, f"{args.product_id_full}_Original", output_plot_dir=plot_output_dir)


        logger.info("--- First Difference Analysis ---")
        diff1_series = train_demand_series.diff().dropna()
        if not diff1_series.empty:
            perform_adf_test(diff1_series, f"{args.product_id_full} 1st Diff")
            plot_acf_pacf_summary(diff1_series, f"{args.product_id_full}_1st_Diff", output_plot_dir=plot_output_dir)
        else:
            logger.warning("First difference series is empty. Skipping analysis.")
        logger.info(f"Time series analysis plots saved to: {plot_output_dir}")
    except Exception as e:
        logger.error(f"Error during time series analysis for {args.product_id_full}: {e}", exc_info=True)


def get_item_and_store_ids(product_id_full: str) -> tuple[str | None, str | None]:
    """Helper to extract item_id and store_id from product_id_full. Returns (None, None) on failure."""
    # Example: FOODS_3_090_CA_3_evaluation -> item_id=FOODS_3_090, store_id=CA_3
    parts = product_id_full.replace("_evaluation", "").replace("_validation", "").split('_')
    if len(parts) >= 4: # Need at least CAT_DEPT_ITEM_STATE_STORE
        item_id = "_".join(parts[:-2])
        store_id = "_".join(parts[-2:])
        return item_id, store_id
    else:
        logger.error(f"Cannot reliably parse item_id and store_id from '{product_id_full}'. Expected format like 'CAT_DEPT_ITEM_STATE_STORE'.")
        return None, None

def cmd_train_forecaster(args):
    """Handles the 'train_forecaster' command."""
    logger.info(f"Training forecaster {args.model_type} for product: {args.product_id_full}")
    parser = M5DataParser(config)

    try:
        train_demand_arr, test_demand_arr = parser.get_or_process_demand_splits(args.product_id_full)
        if train_demand_arr.size == 0:
            logger.error(f"Training demand data is empty for {args.product_id_full}. Cannot train forecaster.")
            return

        calendar_df = parser.load_calendar()
        # Ensure dates align with the length of the demand array
        train_dates = calendar_df['date'][:len(train_demand_arr)]
        train_demand_series = pd.Series(train_demand_arr, index=pd.to_datetime(train_dates), name='y')

        item_id, store_id = get_item_and_store_ids(args.product_id_full)
        if not item_id or not store_id: # Parsing failed
            return

        forecaster = None
        if args.model_type == 'prophet':
            prophet_params_config = config.get_prophet_params(args.product_id_full)
            forecaster = ProphetForecaster(args.product_id_full, config, prophet_params_config)
            
            sell_prices_df = parser.load_sell_prices()
            feature_eng = FeatureEngineer(calendar_df, sell_prices_df) # FeatureEngineer needs calendar and prices
            
            # Prepare DataFrame for Prophet (ds, y, and regressors)
            prophet_train_df = feature_eng.prepare_prophet_train_df(train_demand_series, item_id, store_id, prophet_params_config)
            
            holidays_df = forecaster.get_holidays_df(calendar_df) # Get holidays if configured
            forecaster.train(prophet_train_df, holidays_df=holidays_df)

        elif args.model_type == 'arima':
            # Assuming arima_config might be nested under prophet_params or needs its own config file
            # For now, using a similar structure to prophet_params for model_specific_params
            arima_params_config = config.prophet_params_all.get(args.product_id_full, {}).get("arima_config", {}) # Example
            if not arima_params_config: # Fallback to default if no specific arima_config found
                arima_params_config = config.prophet_params_all.get("default", {}).get("arima_config", {})

            forecaster = ARIMAForecaster(args.product_id_full, config, arima_params_config)
            # ARIMA typically just needs 'y' and optionally 'ds' for indexing, no complex regressors by default here
            arima_train_df = pd.DataFrame({'ds': train_demand_series.index, 'y': train_demand_series.values})
            forecaster.train(arima_train_df) # holidays_df not typically used by basic ARIMA
        else:
            logger.error(f"Unsupported forecaster type: {args.model_type}")
            return
        
        if forecaster and forecaster.model:
            logger.info(f"Forecaster training complete for {args.product_id_full} using {args.model_type}. Model saved.")
            # Optional: Evaluate on test set if it exists
            if test_demand_arr.size > 0:
                logger.info("Evaluating forecaster on test set...")
                test_start_index = len(train_demand_arr)
                test_end_index = test_start_index + len(test_demand_arr)
                test_dates = calendar_df['date'][test_start_index:test_end_index]
                test_demand_series = pd.Series(test_demand_arr, index=pd.to_datetime(test_dates), name='y_test')

                # Create future_df for prediction
                future_df_for_pred = pd.DataFrame({'ds': pd.to_datetime(test_dates)})
                if args.model_type == 'prophet':
                    sell_prices_df = parser.load_sell_prices() # Reload if not in scope
                    feature_eng = FeatureEngineer(calendar_df, sell_prices_df)
                    # Use the forecaster's stored model_specific_params
                    future_df_for_pred = feature_eng.create_prophet_future_df(pd.to_datetime(test_dates), item_id, store_id, forecaster.model_specific_params)


                predictions, ci_df = forecaster.predict(future_df_for_pred)
                if predictions.size > 0:
                    eval_metrics = forecaster.evaluate(test_demand_series.values, predictions)
                    logger.info(f"Test set evaluation for {args.model_type} on {args.product_id_full}: {eval_metrics}")
                    
                    # Plotting
                    plot_output_dir = config.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
                    plot_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    full_demand_series_for_plot = np.concatenate([train_demand_arr, test_demand_arr])
                    full_dates_for_plot = calendar_df['date'][:len(full_demand_series_for_plot)]
                    ts_full_pd_for_plot = pd.Series(full_demand_series_for_plot, index=pd.to_datetime(full_dates_for_plot))

                    plotting.plot_forecast_vs_actual(
                        ts_full_pd=ts_full_pd_for_plot, 
                        ts_test_pd=test_demand_series,
                        test_dates=pd.to_datetime(test_dates), 
                        prophet_preds_test=predictions if args.model_type == 'prophet' else None,
                        arima_preds_test=predictions if args.model_type == 'arima' else None,
                        forecast_test_prophet_ci=ci_df if args.model_type == 'prophet' else None,
                        product_id_full=args.product_id_full, 
                        forecast_horizon=len(test_dates), 
                        rmse_prophet=eval_metrics.get('rmse', np.nan) if args.model_type == 'prophet' else np.nan,
                        rmse_arima=eval_metrics.get('rmse', np.nan) if args.model_type == 'arima' else np.nan,
                        train_days=len(train_demand_arr), 
                        output_plot_dir=plot_output_dir
                    )
                else:
                    logger.warning(f"No predictions generated for test set evaluation of {args.product_id_full}.")
        else:
            logger.error(f"Forecaster training failed for {args.product_id_full} using {args.model_type}.")

    except Exception as e:
        logger.error(f"Error during forecaster training for {args.product_id_full}, type {args.model_type}: {e}", exc_info=True)


def cmd_train_rl(args):
    """Handles the 'train_rl' command."""
    logger.info(f"Starting RL agent training: {args.agent_type} for product: {args.product_id_full}")
    parser = M5DataParser(config)
    
    try:
        train_demand, _ = parser.get_or_process_demand_splits(args.product_id_full)
        if train_demand.size == 0:
            logger.error(f"Training demand is empty for {args.product_id_full}. Cannot train RL agent.")
            return

        if args.agent_type == 'dqn':
            dqn_agent_params = config.get_dqn_params(args.product_id_full)
            env_params_for_agent = config.get_env_params(args.product_id_full) # Gets 'env_config'
            
            # Ensure env_params_for_agent is not empty if 'env_config' was missing
            if not env_params_for_agent:
                logger.error(f"Critical: 'env_config' not found in DQN parameters for {args.product_id_full} or default. Cannot create environment.")
                return

            agent = DQNAgent(args.product_id_full, config, dqn_agent_params, env_params_for_agent)
            agent.train(train_demand) # The train method in DQNAgent handles model saving
        # elif args.agent_type == 'ppo': # Placeholder for PPO
            # logger.warning("PPO agent training not yet fully implemented in CLI.")
            # ppo_params = {} # Load PPO params
            # agent = PPOAgent(args.product_id_full, config, ppo_params, env_params_for_agent)
            # agent.train(train_demand)
        else:
            logger.error(f"Unsupported RL agent type: {args.agent_type}")
            return
        logger.info(f"RL training process initiated for {args.product_id_full} using {args.agent_type}. Check agent logs for completion and model saving.")
    except Exception as e:
        logger.error(f"Error during RL training for {args.product_id_full}, agent {args.agent_type}: {e}", exc_info=True)

def cmd_evaluate_rl(args):
    """Handles the 'evaluate_rl' command."""
    logger.info(f"Evaluating RL agent {args.agent_type} for product: {args.product_id_full}")
    parser = M5DataParser(config)

    try:
        _, test_demand = parser.get_or_process_demand_splits(args.product_id_full)
        if test_demand.size == 0:
            logger.error(f"Test demand is empty for {args.product_id_full}. Cannot evaluate RL agent.")
            return

        if args.agent_type == 'dqn':
            dqn_agent_params = config.get_dqn_params(args.product_id_full)
            env_params_for_agent = config.get_env_params(args.product_id_full)
            if not env_params_for_agent:
                logger.error(f"Critical: 'env_config' not found for {args.product_id_full} or default. Cannot create environment for evaluation.")
                return

            agent = DQNAgent(args.product_id_full, config, dqn_agent_params, env_params_for_agent)
            
            model_suffix_to_load = args.model_suffix if args.model_suffix else "final_trained"
            if not agent.load_model(suffix=model_suffix_to_load):
                 logger.error(f"Could not load RL model (suffix: {model_suffix_to_load}) for evaluation of {args.product_id_full}. Ensure it's trained.")
                 return
            
            # Run RL agent evaluation
            rl_results = agent.evaluate_agent_performance(test_demand, verbose=args.verbose)

            # Run Baseline evaluation
            # Create a non-vectorized env for baseline
            baseline_env = agent.create_env(test_demand, vectorized=False) 
            baseline_policy_params = {'level': args.baseline_level} 
            baseline_results = run_baseline_policy(baseline_env, policy_type='order_up_to', policy_params=baseline_policy_params, verbose=args.verbose)
            baseline_env.close() # Important to close env

            logger.info(f"--- Evaluation Summary for {args.product_id_full} ---")
            logger.info(f"RL Agent ({args.agent_type.upper()}, Model: {model_suffix_to_load}): Total Reward = {rl_results.get('total_reward', 'N/A'):.2f}, AvgInv = {rl_results.get('average_eod_inventory', 'N/A'):.2f}")
            logger.info(f"Baseline (Order-up-to L{args.baseline_level}): Total Reward = {baseline_results.get('total_reward', 'N/A'):.2f}, AvgInv = {baseline_results.get('average_eod_inventory', 'N/A'):.2f}")
            
            # Plotting results
            plot_output_dir = config.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
            plot_output_dir.mkdir(parents=True, exist_ok=True)

            if rl_results and baseline_results and 'inventory_trajectory' in rl_results and 'inventory_trajectory' in baseline_results:
                plotting.plot_rl_vs_baseline_inventory(
                    test_demand=test_demand, 
                    baseline_inventory_trajectory=baseline_results['inventory_trajectory'],
                    rl_inventory_trajectory=rl_results['inventory_trajectory'], 
                    max_inventory=env_params_for_agent['max_inventory'],
                    baseline_label=f"Baseline L{args.baseline_level}", 
                    rl_label=f"{args.agent_type.upper()} Agent",
                    product_id_full=args.product_id_full, 
                    output_plot_dir=plot_output_dir
                )
            if rl_results and baseline_results and 'action_qty_trajectory' in rl_results and 'action_trajectory' in baseline_results:
                 plotting.plot_rl_vs_baseline_orders(
                    baseline_actions=baseline_results['action_trajectory'], # Baseline uses 'action_trajectory' for quantities
                    rl_actions=rl_results['action_qty_trajectory'], # DQNAgent eval uses 'action_qty_trajectory'
                    baseline_label=f"Baseline L{args.baseline_level}", 
                    rl_label=f"{args.agent_type.upper()} Agent",
                    product_id_full=args.product_id_full, 
                    output_plot_dir=plot_output_dir
                )
        else:
            logger.error(f"Unsupported RL agent type for evaluation: {args.agent_type}")
    except Exception as e:
        logger.error(f"Error during RL evaluation for {args.product_id_full}, agent {args.agent_type}: {e}", exc_info=True)


def cmd_get_recommendation(args):
    """Handles the 'get_recommendation' command using RecommendationService."""
    logger.info(f"Getting order recommendation for product: {args.product_id_full}, current inventory: {args.inventory}")
    model_suffix_to_use = args.model_suffix if args.model_suffix else "final_trained"
    try:
        # RecommendationService is designed to be straightforward to call
        recommendation = RecommendationService.get_order_recommendation(
            product_id_full=args.product_id_full, 
            current_inventory=args.inventory,
            model_suffix=model_suffix_to_use
        )
        logger.info(f"Recommended order quantity for {args.product_id_full} (Inv: {args.inventory}, Model: {model_suffix_to_use}): {recommendation}")
        print(f"Recommendation: {recommendation}") # Also print to stdout for CLI user
    except FileNotFoundError as fnf_error:
        logger.error(f"Failed to get recommendation for {args.product_id_full}: {fnf_error}")
        print(f"Error: Model file not found. {fnf_error}")
    except NotImplementedError as ni_error:
        logger.error(f"Failed to get recommendation for {args.product_id_full}: {ni_error}")
        print(f"Error: Service logic not implemented. {ni_error}")
    except Exception as e:
        logger.error(f"Failed to get recommendation for {args.product_id_full}: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


# --- Main CLI Parser Setup ---
def main():
    """Main function to parse arguments and call command functions."""
    cli_parser = argparse.ArgumentParser(
        description="SmartSupplyRL: Inventory Optimization Pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    subparsers = cli_parser.add_subparsers(dest="command", required=True, help="Available commands. Use <command> --help for more details.")

    # --- Common arguments ---
    product_arg_name = "--product_id_full"
    product_arg_help = "Full product ID (e.g., FOODS_3_090_CA_3_evaluation or HOBBIES_1_001_CA_1_evaluation)"

    # --- Subparser for 'preprocess_data' ---
    p_preprocess = subparsers.add_parser("preprocess_data", help="Load and preprocess M5 sales data for a specific product.")
    p_preprocess.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    p_preprocess.add_argument("--train_days", type=int, help="Number of days for training set split. Overrides config if set.")
    p_preprocess.set_defaults(func=cmd_preprocess_data)

    # --- Subparser for 'analyze_timeseries' ---
    p_analyze = subparsers.add_parser("analyze_timeseries", help="Perform ADF tests and plot ACF/PACF for a product's training data.")
    p_analyze.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    # train_days might be relevant if one wants to analyze a specific split not yet default in config
    # p_analyze.add_argument("--train_days", type=int, help="Number of train days if different from config (used to load correct split).")
    p_analyze.set_defaults(func=cmd_analyze_timeseries)

    # --- Subparser for 'train_forecaster' ---
    p_train_fc = subparsers.add_parser("train_forecaster", help="Train a forecasting model (Prophet or ARIMA).")
    p_train_fc.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    p_train_fc.add_argument("--model_type", choices=['prophet', 'arima'], required=True, help="Type of forecasting model to train.")
    p_train_fc.set_defaults(func=cmd_train_forecaster)

    # --- Subparser for 'train_rl' ---
    p_train_rl = subparsers.add_parser("train_rl", help="Train an RL agent (e.g., DQN).")
    p_train_rl.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    p_train_rl.add_argument("--agent_type", choices=['dqn'], required=True, help="Type of RL agent to train (currently only 'dqn' supported).") # Add 'ppo' when ready
    p_train_rl.set_defaults(func=cmd_train_rl)

    # --- Subparser for 'evaluate_rl' ---
    p_eval_rl = subparsers.add_parser("evaluate_rl", help="Evaluate a trained RL agent against a baseline heuristic policy.")
    p_eval_rl.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    p_eval_rl.add_argument("--agent_type", choices=['dqn'], required=True, help="Type of RL agent to evaluate.")
    p_eval_rl.add_argument("--model_suffix", type=str, help="Suffix of the RL model file to load (e.g., 'final_trained', 'best_model'). Default: 'final_trained'.")
    p_eval_rl.add_argument("--baseline_level", type=int, default=60, help="Order-up-to level for the baseline comparison (default: 60).")
    p_eval_rl.add_argument("--verbose", action="store_true", help="Print detailed step-by-step logs during evaluation simulation.")
    p_eval_rl.set_defaults(func=cmd_evaluate_rl)

    # --- Subparser for 'get_recommendation' ---
    p_recommend = subparsers.add_parser("get_recommendation", help="Get an order quantity recommendation from a trained RL model.")
    p_recommend.add_argument(product_arg_name, type=str, required=True, help=product_arg_help)
    p_recommend.add_argument("--inventory", type=float, required=True, help="Current on-hand inventory level for the product.")
    p_recommend.add_argument("--model_suffix", type=str, help="Suffix of the RL model file to use (e.g., 'final_trained', 'best_model'). Default: 'final_trained'.")
    p_recommend.set_defaults(func=cmd_get_recommendation)

    # Parse arguments
    args_parsed = cli_parser.parse_args()

    # Execute the function associated with the chosen command
    if hasattr(args_parsed, 'func'):
        logger.info(f"Executing command: {args_parsed.command} with arguments: {vars(args_parsed)}")
        args_parsed.func(args_parsed)
        logger.info(f"Command {args_parsed.command} finished.")
    else:
        # Should not happen if command is required, but as a fallback:
        cli_parser.print_help()

if __name__ == "__main__":
    main()
