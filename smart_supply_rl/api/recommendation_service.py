import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys # For traceback in MCP logs if needed
import traceback # For traceback

# Imports for RecommendationService class
from smart_supply_rl.utils.config_loader import Config
from smart_supply_rl.utils.logger import setup_logger
from smart_supply_rl.rl_agents.dqn_agent import DQNAgent
# Add any other specific imports your RecommendationService class needs

# Imports for the CLI part of this file (many are already here)
from smart_supply_rl.utils import plotting, helpers # plotting might not be needed by service
from smart_supply_rl.data_processing.m5_parser import M5DataParser
from smart_supply_rl.data_processing.feature_engineering import FeatureEngineer
from smart_supply_rl.forecasting.prophet_model import ProphetForecaster
from smart_supply_rl.forecasting.arima_model import ARIMAForecaster
from smart_supply_rl.forecasting.evaluation import plot_acf_pacf_summary, perform_adf_test
from smart_supply_rl.baselines.heuristic_policies import run_baseline_policy

# CRITICAL: REMOVE THE SELF-IMPORT LINE THAT WAS CAUSING THE CIRCULAR DEPENDENCY
# from smart_supply_rl.api.recommendation_service import RecommendationService # <-- REMOVE THIS LINE

# Initialize Config and Logger globally for this module.
# The RecommendationService class can use these, or be made more independent.
config_module = Config()
logger_module = setup_logger("SmartSupplyModule", config_instance=config_module)


class RecommendationService:
    """
    Provides order recommendations using a trained RL agent.
    """
    # The class can use the module-level config and logger, or you can pass them explicitly.
    # For simplicity, this example will use module-level config_module and logger_module.

    @staticmethod
    def get_order_recommendation(product_id_full: str, current_inventory: float, model_suffix: str = "final_trained") -> int:
        """
        Gets an order recommendation for a given product and current inventory.

        Args:
            product_id_full (str): The full product ID.
            current_inventory (float): The current on-hand inventory.
            model_suffix (str): Suffix for the RL model file (e.g., "final_trained").

        Returns:
            int: The recommended order quantity.

        Raises:
            FileNotFoundError: If the RL model cannot be loaded.
            NotImplementedError: If crucial prediction logic is missing.
            Exception: For other errors during recommendation.
        """
        logger_module.info(f"RecommendationService: Request for {product_id_full}, inv: {current_inventory}, model: {model_suffix}")

        try:
            # 1. Get necessary parameters from config
            env_params = config_module.get_env_params(product_id_full)
            dqn_params = config_module.get_dqn_params(product_id_full)

            # 2. Initialize the RL agent
            agent = DQNAgent(product_id_full, config_module, dqn_params, env_params)

            # 3. Load the trained model
            if not agent.load_model(suffix=model_suffix):
                error_msg = f"RecommendationService: Failed to load RL model '{model_suffix}' for product '{product_id_full}'."
                logger_module.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 4. Construct the observation for the model based on current_inventory.
            #    This is CRITICAL and depends heavily on your RL environment's observation space.
            #    You need a way to create the same observation structure your model was trained on.
            #    If your DQNAgent or environment has a helper method, use it.
            #    Example: obs = agent.env.get_observation_for_inventory(current_inventory)
            #    Example: obs = agent.construct_observation_from_state_data(current_inventory, other_features_if_any)
            
            # --- Placeholder for observation construction ---
            # --- YOU MUST IMPLEMENT THIS PART BASED ON YOUR ENVIRONMENT ---
            if not hasattr(agent, 'create_observation_for_prediction'):
                 # Add this method to your DQNAgent, or implement logic here.
                 # It needs to return a NumPy array matching the model's expected input shape.
                 logger_module.error("DQNAgent is missing 'create_observation_for_prediction' method.")
                 raise NotImplementedError("Observation construction logic for prediction is missing in RecommendationService/DQNAgent.")
            
            observation = agent.create_observation_for_prediction(current_inventory)
            # --- End of placeholder ---

            # 5. Get the action (recommendation) from the agent's model
            action, _states = agent.model.predict(observation, deterministic=True)
            
            # 6. Convert action to order quantity
            #    This depends on how your action space is defined (e.g., direct quantity, action index).
            #    Assuming 'action' is the direct order quantity or can be easily converted.
            recommended_order_qty = int(np.round(action[0])) if isinstance(action, np.ndarray) else int(np.round(action))


            logger_module.info(f"RecommendationService: Generated recommendation for {product_id_full}: {recommended_order_qty} units.")
            return recommended_order_qty

        except FileNotFoundError: # Re-raise specifically
            raise
        except NotImplementedError: # Re-raise specifically
            raise
        except Exception as e:
            logger_module.error(f"RecommendationService: Error during recommendation generation for {product_id_full}: {e}", exc_info=False) # Set exc_info=True for full stack in dev
            # For MCP, printing to stderr is also good for debugging
            print(f"ERROR in RecommendationService.get_order_recommendation: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            # Depending on desired behavior, either raise a custom exception or return a fallback (e.g., 0 or -1)
            raise # Re-raise the original exception to signal failure


# --- The rest of your CLI utility code (functions and __main__) ----
# Make sure they use `config_module` and `logger_module` if they were using the old `config` and `logger`.

def get_item_and_store_ids(product_id_full: str) -> tuple[str, str]:
    """Helper to extract item_id and store_id from product_id_full."""
    parts = product_id_full.replace("_evaluation", "").replace("_validation", "").split('_')
    if len(parts) < 4:
        logger_module.error(f"Cannot reliably parse item_id and store_id from '{product_id_full}'. Expected format like 'CAT_DEPT_ITEM_STATE_STORE'.")
        raise ValueError(f"Invalid product_id_full format: {product_id_full}")
    item_id = "_".join(parts[:-2])
    store_id = "_".join(parts[-2:])
    return item_id, store_id

def cmd_preprocess_data(args):
    logger_module.info(f"Running data preprocessing for product: {args.product_id_full}")
    parser = M5DataParser(config_module) # Use module-level config
    train_days = args.train_days if args.train_days is not None else config_module.get_train_days_for_product(args.product_id_full)
    
    full_demand_series = parser.get_product_timeseries(args.product_id_full)
    train_demand, test_demand = parser.split_and_save_demand(args.product_id_full, full_demand_series, train_days)
    logger_module.info(f"Data preprocessing complete for {args.product_id_full}. Train shape: {train_demand.shape}, Test shape: {test_demand.shape}")

def cmd_analyze_timeseries(args):
    logger_module.info(f"Analyzing time series for product: {args.product_id_full}")
    parser = M5DataParser(config_module)
    train_days = args.train_days if args.train_days is not None else config_module.get_train_days_for_product(args.product_id_full)
    train_demand_arr, _ = parser.get_or_process_demand_splits(args.product_id_full)

    if train_demand_arr.size == 0:
        logger_module.error(f"Train demand data is empty for {args.product_id_full}. Cannot analyze.")
        return
        
    train_demand_series = pd.Series(train_demand_arr)
    
    plot_output_dir = config_module.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    logger_module.info("--- Original Series Analysis ---")
    perform_adf_test(train_demand_series, f"{args.product_id_full} Original")
    plot_acf_pacf_summary(train_demand_series, f"{args.product_id_full}_Original", output_plot_dir=plot_output_dir)

    logger_module.info("--- First Difference Analysis ---")
    diff1_series = train_demand_series.diff().dropna()
    if not diff1_series.empty:
        perform_adf_test(diff1_series, f"{args.product_id_full} 1st Diff")
        plot_acf_pacf_summary(diff1_series, f"{args.product_id_full}_1st_Diff", output_plot_dir=plot_output_dir)
    else:
        logger_module.warning("First difference series is empty. Skipping analysis.")
    logger_module.info(f"Time series analysis plots saved to: {plot_output_dir}")


def cmd_train_forecaster(args):
    logger_module.info(f"Training forecaster {args.model_type} for product: {args.product_id_full}")
    parser = M5DataParser(config_module)
    train_demand_arr, test_demand_arr = parser.get_or_process_demand_splits(args.product_id_full)
    
    calendar_df = parser.load_calendar()
    train_dates = calendar_df['date'][:len(train_demand_arr)]
    train_demand_series = pd.Series(train_demand_arr, index=train_dates, name='y')

    item_id, store_id = get_item_and_store_ids(args.product_id_full)
    
    forecaster = None
    if args.model_type == 'prophet':
        prophet_params = config_module.get_prophet_params(args.product_id_full)
        forecaster = ProphetForecaster(args.product_id_full, config_module, prophet_params)
        
        sell_prices_df = parser.load_sell_prices()
        feature_eng = FeatureEngineer(calendar_df, sell_prices_df)
        prophet_train_df = feature_eng.prepare_prophet_train_df(train_demand_series, item_id, store_id, prophet_params)
        
        holidays_df = forecaster.get_holidays_df(calendar_df)
        forecaster.train(prophet_train_df, holidays_df)

    elif args.model_type == 'arima':
        arima_params = config_module.prophet_params_all.get(args.product_id_full, {}).get("arima_config", {})
        forecaster = ARIMAForecaster(args.product_id_full, config_module, arima_params)
        arima_train_df = pd.DataFrame({'ds': train_demand_series.index, 'y': train_demand_series.values})
        forecaster.train(arima_train_df)
    else:
        logger_module.error(f"Unsupported forecaster type: {args.model_type}")
        return
    
    if forecaster and forecaster.model:
        logger_module.info(f"Forecaster training complete for {args.product_id_full} using {args.model_type}. Model saved.")
        if test_demand_arr.size > 0:
            logger_module.info("Evaluating forecaster on test set...")
            test_dates = calendar_df['date'][len(train_demand_arr) : len(train_demand_arr) + len(test_demand_arr)]
            test_demand_series = pd.Series(test_demand_arr, index=test_dates, name='y_test')

            future_df_for_pred = pd.DataFrame({'ds': test_dates})
            if args.model_type == 'prophet':
                 sell_prices_df = parser.load_sell_prices()
                 feature_eng = FeatureEngineer(calendar_df, sell_prices_df)
                 future_df_for_pred = feature_eng.create_prophet_future_df(test_dates, item_id, store_id, forecaster.model_specific_params)

            predictions, ci_df = forecaster.predict(future_df_for_pred)
            if predictions.size > 0:
                eval_metrics = forecaster.evaluate(test_demand_series.values, predictions)
                logger_module.info(f"Test set evaluation for {args.model_type}: {eval_metrics}")
                plot_output_dir = config_module.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
                plot_output_dir.mkdir(parents=True, exist_ok=True)
                full_demand_series = np.concatenate([train_demand_arr, test_demand_arr])
                full_dates = calendar_df['date'][:len(full_demand_series)]
                ts_full_for_plot = pd.Series(full_demand_series, index=full_dates)
                plotting.plot_forecast_vs_actual(
                    ts_full_pd=ts_full_for_plot, ts_test_pd=test_demand_series, test_dates=test_dates,
                    prophet_preds_test=predictions if args.model_type == 'prophet' else None,
                    arima_preds_test=predictions if args.model_type == 'arima' else None,
                    forecast_test_prophet_ci=ci_df if args.model_type == 'prophet' else None,
                    product_id_full=args.product_id_full, forecast_horizon=len(test_dates),
                    rmse_prophet=eval_metrics.get('rmse', np.nan) if args.model_type == 'prophet' else np.nan,
                    rmse_arima=eval_metrics.get('rmse', np.nan) if args.model_type == 'arima' else np.nan,
                    train_days=len(train_demand_arr), output_plot_dir=plot_output_dir
                )
            else:
                logger_module.warning("No predictions generated for test set evaluation.")
    else:
        logger_module.error(f"Forecaster training failed for {args.product_id_full} using {args.model_type}.")


def cmd_train_rl(args):
    logger_module.info(f"Training RL agent {args.agent_type} for product: {args.product_id_full}")
    parser = M5DataParser(config_module)
    train_demand, _ = parser.get_or_process_demand_splits(args.product_id_full)
    if train_demand.size == 0:
        logger_module.error(f"Training demand is empty for {args.product_id_full}. Cannot train RL agent.")
        return
    if args.agent_type == 'dqn':
        dqn_agent_params = config_module.get_dqn_params(args.product_id_full)
        env_params_for_agent = config_module.get_env_params(args.product_id_full)
        agent = DQNAgent(args.product_id_full, config_module, dqn_agent_params, env_params_for_agent)
        agent.train(train_demand)
    else:
        logger_module.error(f"Unsupported RL agent type: {args.agent_type}")
        return
    logger_module.info(f"RL training possibly complete for {args.product_id_full} using {args.agent_type}. Check logs.")

def cmd_evaluate_rl(args):
    logger_module.info(f"Evaluating RL agent {args.agent_type} for product: {args.product_id_full}")
    parser = M5DataParser(config_module)
    _, test_demand = parser.get_or_process_demand_splits(args.product_id_full)
    if test_demand.size == 0:
        logger_module.error(f"Test demand is empty for {args.product_id_full}. Cannot evaluate RL agent.")
        return
    if args.agent_type == 'dqn':
        dqn_agent_params = config_module.get_dqn_params(args.product_id_full)
        env_params_for_agent = config_module.get_env_params(args.product_id_full)
        agent = DQNAgent(args.product_id_full, config_module, dqn_agent_params, env_params_for_agent)
        model_suffix = args.model_suffix if args.model_suffix else "final_trained"
        if not agent.load_model(suffix=model_suffix):
             logger_module.error(f"Could not load RL model (suffix: {model_suffix}) for evaluation. Ensure it's trained.")
             return
        rl_results = agent.evaluate_agent_performance(test_demand, verbose=args.verbose)
        baseline_env = agent.create_env(test_demand, vectorized=False)
        baseline_policy_params = {'level': args.baseline_level} 
        baseline_results = run_baseline_policy(baseline_env, policy_type='order_up_to', policy_params=baseline_policy_params, verbose=args.verbose)
        baseline_env.close()
        logger_module.info(f"--- Evaluation Summary for {args.product_id_full} ---")
        logger_module.info(f"RL Agent ({args.agent_type}, model: {model_suffix}): Total Reward = {rl_results.get('total_reward', 'N/A'):.2f}")
        logger_module.info(f"Baseline (Order-up-to L{args.baseline_level}): Total Reward = {baseline_results.get('total_reward', 'N/A'):.2f}")
        plot_output_dir = config_module.get_path('output_dir_abs', 'results', 'plots', args.product_id_full)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        if rl_results and baseline_results:
            plotting.plot_rl_vs_baseline_inventory(
                test_demand=test_demand, baseline_inventory_trajectory=baseline_results['inventory_trajectory'],
                rl_inventory_trajectory=rl_results['inventory_trajectory'], max_inventory=env_params_for_agent['max_inventory'],
                baseline_label=f"Baseline L{args.baseline_level}", rl_label=f"{args.agent_type.upper()} Agent",
                product_id_full=args.product_id_full, output_plot_dir=plot_output_dir
            )
            plotting.plot_rl_vs_baseline_orders(
                baseline_actions=baseline_results['action_trajectory'], rl_actions=rl_results['action_qty_trajectory'],
                baseline_label=f"Baseline L{args.baseline_level}", rl_label=f"{args.agent_type.upper()} Agent",
                product_id_full=args.product_id_full, output_plot_dir=plot_output_dir
            )
    else:
        logger_module.error(f"Unsupported RL agent type for evaluation: {args.agent_type}")


def cmd_get_recommendation(args):
    logger_module.info(f"Getting order recommendation for product: {args.product_id_full}, inventory: {args.inventory}")
    model_suffix = args.model_suffix if args.model_suffix else "final_trained"
    try:
        recommendation = RecommendationService.get_order_recommendation(
            args.product_id_full, 
            args.inventory,
            model_suffix=model_suffix
        )
        logger_module.info(f"Recommended order quantity for {args.product_id_full} (Inv: {args.inventory}): {recommendation}")
    except Exception as e:
        logger_module.error(f"Failed to get recommendation for {args.product_id_full}: {e}")
        # Error already logged by RecommendationService, this is just for CLI feedback.


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="SmartSupplyRL: Inventory Optimization CLI")
    subparsers = cli_parser.add_subparsers(dest="command", required=True, help="Available commands")
    product_arg = "--product_id_full"
    product_help = "Full product ID (e.g., FOODS_3_090_CA_3_evaluation)"
    p_preprocess = subparsers.add_parser("preprocess_data", help="Load and preprocess M5 data for a product.")
    p_preprocess.add_argument(product_arg, type=str, required=True, help=product_help)
    p_preprocess.add_argument("--train_days", type=int, help="Number of days for training set split. Overrides config if set.")
    p_preprocess.set_defaults(func=cmd_preprocess_data)
    p_analyze = subparsers.add_parser("analyze_timeseries", help="Perform ADF tests and plot ACF/PACF for a product's training data.")
    p_analyze.add_argument(product_arg, type=str, required=True, help=product_help)
    p_analyze.add_argument("--train_days", type=int, help="Number of train days if different from config (used to load correct split).")
    p_analyze.set_defaults(func=cmd_analyze_timeseries)
    p_train_fc = subparsers.add_parser("train_forecaster", help="Train a forecasting model.")
    p_train_fc.add_argument(product_arg, type=str, required=True, help=product_help)
    p_train_fc.add_argument("--model_type", choices=['prophet', 'arima'], required=True, help="Type of forecasting model to train.")
    p_train_fc.set_defaults(func=cmd_train_forecaster)
    p_train_rl = subparsers.add_parser("train_rl", help="Train an RL agent.")
    p_train_rl.add_argument(product_arg, type=str, required=True, help=product_help)
    p_train_rl.add_argument("--agent_type", choices=['dqn'], required=True, help="Type of RL agent to train.")
    p_train_rl.set_defaults(func=cmd_train_rl)
    p_eval_rl = subparsers.add_parser("evaluate_rl", help="Evaluate a trained RL agent against a baseline.")
    p_eval_rl.add_argument(product_arg, type=str, required=True, help=product_help)
    p_eval_rl.add_argument("--agent_type", choices=['dqn'], required=True, help="Type of RL agent to evaluate.")
    p_eval_rl.add_argument("--model_suffix", type=str, help="Suffix of the model file to load. Default: 'final_trained'.")
    p_eval_rl.add_argument("--baseline_level", type=int, default=60, help="Order-up-to level for baseline comparison (default: 60).")
    p_eval_rl.add_argument("--verbose", action="store_true", help="Print detailed step-by-step logs during evaluation.")
    p_eval_rl.set_defaults(func=cmd_evaluate_rl)
    p_recommend = subparsers.add_parser("get_recommendation", help="Get order recommendation from a trained RL model.")
    p_recommend.add_argument(product_arg, type=str, required=True, help=product_help)
    p_recommend.add_argument("--inventory", type=float, required=True, help="Current on-hand inventory level.")
    p_recommend.add_argument("--model_suffix", type=str, help="Suffix of the RL model file to use. Default: 'final_trained'.")
    p_recommend.set_defaults(func=cmd_get_recommendation)
    args_parsed = cli_parser.parse_args()
    logger_module.info(f"Executing command: {args_parsed.command}")
    args_parsed.func(args_parsed)
    logger_module.info(f"Command {args_parsed.command} finished.")