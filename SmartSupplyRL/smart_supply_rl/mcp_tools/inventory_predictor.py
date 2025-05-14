import os
from pathlib import Path
import sys
import traceback # Keep for the CWD block
import warnings # Import warnings module
import argparse
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

# --- CWD and sys.path modification (keep as is, ensure no prints to stdout/stderr for MCP) ---
# (Assuming the CWD block from previous responses is here and silent for MCP)
try:
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent.parent
    os.chdir(project_root)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
except Exception:
    # This part should not print to stdout/stderr if MCP is running
    # If it fails, subsequent imports will fail, and the tool won't load/run.
    pass
# --- END OF CWD AND SYS.PATH MODIFICATION ---


# MCP Server specific import
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    class FastMCP: # Dummy for local testing
        def __init__(self, server_name, dependencies): pass
        def tool(self):
            def decorator(func): return func
            return decorator
        def run(self): pass

from smart_supply_rl.utils.config_loader import Config
from smart_supply_rl.rl_agents.dqn_agent import DQNAgent
from smart_supply_rl.rl_environment.inventory_env import InventoryEnv

# --- MCP Server Initialization ---
mcp = FastMCP(
    server_name="SmartSupplyRLInventoryPredictor",
    dependencies=["numpy", "PyYAML", "gymnasium", "stable-baselines3"]
)

_RL_MODEL_CACHE: dict[tuple[str, str], BaseAlgorithm | None] = {}
_ACTION_MAP_CACHE: dict[tuple[str, str], dict[int, int] | None] = {}
_MODEL_LOAD_ATTEMPT_MARKER: set[tuple[str, str]] = set()

# --- Constants for error codes ---
ERROR_CRITICAL_FAILURE = -1 # e.g. model load failed, config error
ERROR_INVALID_ACTION_FROM_MODEL = -2 # Model predicted action_idx not in map
# A successful prediction of "order 0" will be returned as 0.

def _load_rl_model_and_map(
    product_id_full: str, model_suffix: str, tool_config: Config
) -> bool:
    cache_key = (product_id_full, model_suffix)
    # Filter warnings specifically for this loading part if necessary,
    # though DQNAgent and InventoryEnv should ideally not print warnings critical for flow.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress warnings locally if they are the source of JSON corruption

        if cache_key in _RL_MODEL_CACHE and _RL_MODEL_CACHE.get(cache_key) is not None:
            return True
        if cache_key in _MODEL_LOAD_ATTEMPT_MARKER and _RL_MODEL_CACHE.get(cache_key) is None:
            return False

        _MODEL_LOAD_ATTEMPT_MARKER.add(cache_key)
        _RL_MODEL_CACHE[cache_key] = None
        _ACTION_MAP_CACHE[cache_key] = None

        if not tool_config:
            return False

        try:
            env_params_for_agent = tool_config.get_env_params(product_id_full)
            dqn_agent_params = tool_config.get_dqn_params(product_id_full)

            if not env_params_for_agent or not dqn_agent_params: # Basic check
                return False # Config parameters missing

            agent = DQNAgent(product_id_full, tool_config, dqn_agent_params, env_params_for_agent)
            
            model_path = agent._get_full_model_path(model_suffix)
            if not model_path.exists():
                return False # Model file does not exist

            if not agent.load_model(suffix=model_suffix):
                return False
            
            _RL_MODEL_CACHE[cache_key] = agent.model

            dummy_demand_data = np.array([0] * (env_params_for_agent.get('lead_time', 1) + 5))
            temp_env = InventoryEnv(demand_data=dummy_demand_data, env_params=env_params_for_agent)
            _ACTION_MAP_CACHE[cache_key] = temp_env._action_to_quantity
            temp_env.close()
            return True
        except Exception:
            return False

def _predict_rl_action(
    product_id_full: str, model_suffix: str, current_inventory: float, tool_config: Config
) -> tuple[int | None, int | None]: # Returns (action_index, quantity) or (None, None) or (action_index, None)
    cache_key = (product_id_full, model_suffix)
    if not _load_rl_model_and_map(product_id_full, model_suffix, tool_config):
        return None, None 

    model = _RL_MODEL_CACHE.get(cache_key)
    action_map = _ACTION_MAP_CACHE.get(cache_key)

    if model is None or action_map is None:
        return None, None 

    if current_inventory < 0:
        current_inventory = 0

    try:
        observation = np.array([current_inventory], dtype=np.float32)
        action_array, _ = model.predict(observation, deterministic=True)
        action_index = int(action_array.item())
        actual_order_quantity = action_map.get(action_index) # Will be None if action_index is not in map
        return action_index, actual_order_quantity
    except Exception:
        return None, None


@mcp.tool()
def predict_order_quantity(
    product_id_full: str, current_inventory: float, model_suffix: str = "final_trained"
) -> int:
    """
    Predicts the quantity of a product to order based on current inventory.
    This tool uses the project's Config, DQNAgent for model loading, and InventoryEnv
    to determine action mappings, similar to the user's main.py POC logic.

    Args:
        product_id_full (str): MANDATORY. The full product ID (e.g., "FOODS_3_090_CA_3_evaluation").
        current_inventory (float): MANDATORY. The current on-hand inventory for the product.
        model_suffix (str): Optional. Suffix for the RL model file (e.g., "final_trained", "best_model").
                            Defaults to "final_trained".

    Returns:
        int: The recommended order quantity.
             Returns -1 on critical failure (e.g., model load failure or invalid action).
             Returns 0 if the best action is to order 0 or a non-critical prediction error occurs.
    """
    # Suppress warnings for the duration of this tool call to prevent JSON corruption
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tool_config = Config() 
        except Exception:
            return ERROR_CRITICAL_FAILURE

        action_idx, recommended_quantity = _predict_rl_action(
            product_id_full, model_suffix, current_inventory, tool_config
        )

        if recommended_quantity is not None: # Successfully got a quantity (could be 0)
            return int(recommended_quantity)
        elif action_idx is not None and recommended_quantity is None:
            # Model predicted an action_idx, but it's not valid in the map
            return ERROR_INVALID_ACTION_FROM_MODEL 
        else:
            # Covers model load failure or other prediction errors from _predict_rl_action
            # where action_idx might also be None.
            return ERROR_CRITICAL_FAILURE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory Predictor MCP Tool - Local Test")
    parser.add_argument("product_id_full", type=str, help="Full product ID")
    parser.add_argument("current_inventory", type=float, help="Current on-hand inventory")
    parser.add_argument("--model_suffix", type=str, default="final_trained", help="RL model file suffix")
    args = parser.parse_args()

    print(f"Local Test: CWD is {os.getcwd()}") # For visibility in local test
    
    # Temporarily allow warnings for local test if needed for debugging setup
    # with warnings.catch_warnings():
        # warnings.simplefilter("default") # Show warnings for local test
    recommendation = predict_order_quantity(
        product_id_full=args.product_id_full,
        current_inventory=args.current_inventory,
        model_suffix=args.model_suffix
    )
    
    print(f"\nLocal Test Result for {args.product_id_full}:")
    print(f"  Current Inventory: {args.current_inventory}")
    print(f"  Model Suffix: {args.model_suffix}")
    if recommendation == ERROR_CRITICAL_FAILURE:
        print("  Recommendation: FAILED (Critical Error / Model Load Issue) [-1]")
    elif recommendation == ERROR_INVALID_ACTION_FROM_MODEL:
        print("  Recommendation: FAILED (Model predicted invalid action) [-2]")
    else:
        print(f"  Recommended Order Quantity: {recommendation}")