# main.py
import os
import sys # Keep sys for potential debug prints
from mcp.server.fastmcp import FastMCP
import gymnasium as gym
import numpy as np
import pandas as pd # Need pandas if you plan to load data later
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm # For type hinting
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import warnings
# REMOVED: import packaging # Removed this line

# Create an MCP server
# REMOVED: packaging from dependencies list
mcp = FastMCP("Supply Chain Predictions", dependencies=["pandas", "numpy"])


class InventoryEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating a single-product inventory system.

    Observation Space:
        Box(low=0, high=max_inventory, shape=(1,), dtype=np.float32)
        Represents the current on-hand inventory level.

    Action Space:
        Discrete(n)
        Where n corresponds to the number of possible order quantities
        (e.g., 0, 10, 20, ..., max_order_quantity).

    Reward:
        Default: Negative total cost (holding + stockout + order).
        Can be configured to include revenue.

    Episode Termination:
        The episode ends when the demand data runs out.
    """
    metadata = {"render_modes": [], "render_fps": 1} # No rendering implemented

    def __init__(self,
                 demand_data: np.ndarray,
                 lead_time: int = 1,
                 holding_cost: float = 0.1,
                 stockout_cost: float = 1.0,
                 order_cost: float = 0.0, # Optional fixed cost per order placed
                 unit_price: float = 2.0, # Optional revenue per unit sold
                 max_inventory: int = 1000,
                 max_order_quantity: int = 100,
                 order_step_size: int = 10, # Orders must be multiples of this
                 initial_inventory: int = 0,
                 include_revenue_in_reward: bool = False # If True, reward = profit, else reward = -cost
                ):
        """
         Initializes the inventory environment.

        Args:
            demand_data: NumPy array of daily demand values.
            lead_time: Number of days for an order to arrive. Min 0 (arrives start of next day).
            holding_cost: Cost per unit held in inventory per day (charged on ending inventory).
            stockout_cost: Cost per unit of unmet demand per day.
            order_cost: Fixed cost incurred whenever an order > 0 is placed.
            unit_price: Revenue per unit sold.
            max_inventory: Maximum allowable inventory level (capacity).
            max_order_quantity: The largest single order quantity allowed.
            order_step_size: Order quantities must be multiples of this step size.
            initial_inventory: Inventory level at the start of an episode.
            include_revenue_in_reward: Whether to maximize profit or minimize cost.
        """
        super().__init__()

        if not isinstance(demand_data, np.ndarray) or demand_data.ndim != 1:
            raise ValueError("demand_data must be a 1D NumPy array.")
        if lead_time < 0:
            raise ValueError("lead_time cannot be negative.")
        if max_inventory <= 0:
            raise ValueError("max_inventory must be positive.")
        if max_order_quantity < 0 or order_step_size <= 0 or max_order_quantity % order_step_size != 0:
            raise ValueError("Invalid order quantity parameters.")

        self.demand_data = demand_data.astype(np.int32) # Ensure integer demand
        self.episode_length = len(demand_data)
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.unit_price = unit_price
        self.max_inventory = max_inventory
        self.max_order_quantity = max_order_quantity
        self.order_step_size = order_step_size
        self.initial_inventory = min(initial_inventory, max_inventory) # Cap initial inventory
        self.include_revenue_in_reward = include_revenue_in_reward

        # --- Define Action Space ---
        # Possible order quantities: 0, step, 2*step, ..., max_order_quantity
        self.possible_orders = np.arange(0, self.max_order_quantity + self.order_step_size, self.order_step_size, dtype=np.int32)
        self.action_space = spaces.Discrete(len(self.possible_orders))
        self._action_to_quantity = {i: q for i, q in enumerate(self.possible_orders)}
        print(f"Action space: {self.action_space}")
        print(f"Action -> Quantity mapping: {self._action_to_quantity}")


        # --- Define Observation Space ---
        # Simple: Just current on-hand inventory
        self.observation_space = spaces.Box(
            low=0, high=self.max_inventory, shape=(1,), dtype=np.float32
        )
        print(f"Observation space: {self.observation_space}")

        # --- Initialize State Variables (will be reset) ---
        self.current_day_index = 0
        self.current_inventory = 0 # Will be set in reset()
        # Dictionary to track orders: {arrival_day: quantity}
        self.on_order_inventory = {}

        # --- Check for potential issues ---
        if self.episode_length == 0:
            warnings.warn("Demand data is empty. Environment cannot run.")


    def _get_obs(self) -> np.ndarray:
        """Returns the current observation."""
        return np.array([self.current_inventory], dtype=np.float32)

    def _get_info(self) -> dict:
        """Returns auxiliary information about the current step."""
        # You can customize this dict to include helpful info for debugging or analysis
        return {
            "day": self.current_day_index,
            "inventory_level": self.current_inventory,
            "on_order": sum(self.on_order_inventory.values()),
            # Add info from the *previous* step if needed (demand, sales etc.)
            # This might require storing them temporarily in self during step()
        }

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state for a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional dictionary with environment-specific options.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed) # Important for seeding internal RNG if needed

        self.current_day_index = 0
        self.current_inventory = self.initial_inventory
        self.on_order_inventory = {}

        observation = self._get_obs()
        info = self._get_info()

        # print(f"Environment Reset: Day={self.current_day_index}, Inv={self.current_inventory}") # Debug print
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step within the environment.

        Args:
            action: An integer representing the chosen action (index in possible_orders).

        Returns:
            A tuple containing:
                observation (np.ndarray): The observation after the step.
                reward (float): The reward obtained during the step.
                terminated (bool): Whether the episode has ended naturally (end of data).
                truncated (bool): Whether the episode was ended prematurely (not used here).
                info (dict): Auxiliary information.
        """
        if self.current_day_index >= self.episode_length:
            warnings.warn("Step called after environment termination.", UserWarning)
            # Still return valid types, potentially with zero reward and terminal state
            obs = self._get_obs()
            info = self._get_info()
            return obs, 0.0, True, False, info

        # --- 1. Get Actual Order Quantity from Action ---
        actual_order_quantity = self._action_to_quantity[action]
        order_placed_cost = self.order_cost if actual_order_quantity > 0 else 0.0

        # --- 2. Check for Arriving Orders ---
        arriving_orders = self.on_order_inventory.pop(self.current_day_index, 0) # Get and remove orders arriving today
        # print(f"Day {self.current_day_index}: Arriving={arriving_orders}") # Debug print

        # --- 3. Update Inventory with Arrivals (Cap at Max) ---
        self.current_inventory += arriving_orders
        # Calculate overflow (optional - could be penalized)
        overflow = max(0, self.current_inventory - self.max_inventory)
        self.current_inventory = min(self.current_inventory, self.max_inventory)
        # print(f"Day {self.current_day_index}: Inv after arrival={self.current_inventory}") # Debug print


        # --- 4. Get Demand for the Current Day ---
        demand = self.demand_data[self.current_day_index]
        # print(f"Day {self.current_day_index}: Demand={demand}") # Debug print


        # --- 5. Calculate Sales and Unmet Demand ---
        sales = min(demand, self.current_inventory)
        unmet_demand = demand - sales
        # print(f"Day {self.current_day_index}: Sales={sales}, Unmet={unmet_demand}") # Debug print


        # --- 6. Update Inventory After Sales ---
        self.current_inventory -= sales
        # print(f"Day {self.current_day_index}: Inv after sales={self.current_inventory}") # Debug print


        # --- 7. Calculate Costs for the Day ---
        holding_cost_today = self.current_inventory * self.holding_cost
        stockout_cost_today = unmet_demand * self.stockout_cost

        # --- 8. Calculate Reward ---
        if self.include_revenue_in_reward:
            revenue_today = sales * self.unit_price
            reward = revenue_today - holding_cost_today - stockout_cost_today - order_placed_cost
        else:
            # Minimize cost (negative reward)
            reward = - (holding_cost_today + stockout_cost_today + order_placed_cost)
            # Optional: Add penalty for overflow? e.g., - overflow * some_penalty

        # print(f"Day {self.current_day_index}: Costs(H={holding_cost_today:.2f}, S={stockout_cost_today:.2f}, O={order_placed_cost:.2f}), Reward={reward:.2f}") # Debug print


        # --- 9. Place the New Order (if any) ---
        if actual_order_quantity > 0:
            arrival_day = self.current_day_index + self.lead_time
            self.on_order_inventory[arrival_day] = self.on_order_inventory.get(arrival_day, 0) + actual_order_quantity
            # print(f"Day {self.current_day_index}: Placed order={actual_order_quantity}, arrives day {arrival_day}") # Debug print


        # --- 10. Advance Time ---
        self.current_day_index += 1

        # --- 11. Check for Termination ---
        terminated = self.current_day_index >= self.episode_length
        truncated = False # Not using truncation based on step limits here

        # --- 12. Get Next State Observation and Info ---
        observation = self._get_obs()
        info = { # Update info with step results
            "day": self.current_day_index -1, # Reflects the day that just finished
            "demand": demand,
            "sales": sales,
            "unmet_demand": unmet_demand,
            "holding_cost": holding_cost_today,
            "stockout_cost": stockout_cost_today,
            "order_cost": order_placed_cost,
            "order_placed": actual_order_quantity,
            "inventory_start_day": self.current_inventory + sales - arriving_orders, # Inv before arrivals
            "inventory_after_arrival": self.current_inventory + sales, # Inv after arrivals, before demand
            "inventory_end_day": self.current_inventory,
            "arriving_orders": arriving_orders,
            "on_order_total": sum(self.on_order_inventory.values()),
            "reward": reward,
            "overflow": overflow
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Perform any necessary cleanup."""
        # print("Closing InventoryEnv.")
        pass # Nothing specific needed for this simple environment


# --- Configuration Paths and Constants ---
# NOTE: Ensure these paths are correct for your environment
PROJECT_DIR = 'E://Projects//Blue_Yonder//output'
# Assuming DRIVE_PROJECT_PATH is for a Colab/Drive setup and not needed locally if PROJECT_DIR exists
# DRIVE_PROJECT_PATH = '/content/drive/MyDrive/Blue_Yonder/' # Commented out as it conflicts with local path

if os.path.exists('E://Projects//Blue_Yonder'):
    # Use local path if it exists
    INPUT_DATA_PATH = 'E://Projects//Blue_Yonder//Dataset/'
    # Ensure the output directory for model/logs exists if using local path
    os.makedirs(PROJECT_DIR, exist_ok=True)
else:
     # Fallback or alternative path if local one isn't found
     # You might need to adjust this if not running locally
     # For example, set it up for a specific deployment environment
     print("Warning: Local project directory not found. Using default/alternative paths.", file=sys.stderr)
     # Define a default path if needed, or raise an error
     # Example: PROJECT_DIR = '/app/output' # For a Docker container
     # Example: INPUT_DATA_PATH = '/app/dataset' # For a Docker container
     pass # Or handle as needed for your specific environment

# Construct full paths using the determined PROJECT_DIR and INPUT_DATA_PATH
INPUT_TRAIN_PATH = os.path.join(PROJECT_DIR, 'demand_train_single_product.npy')
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'dqn_inventory_model')
TENSORBOARD_LOG_DIR_RAW = os.path.join(PROJECT_DIR, 'dqn_inventory_tensorboard/')
TENSORBOARD_LOG_DIR = os.path.normpath(TENSORBOARD_LOG_DIR_RAW)
# INPUT_DATA is likely the raw CSV name, path should be INPUT_DATA_PATH + INPUT_DATA
RAW_INPUT_DATA_FILE = 'sales_train_validation.csv'
FULL_INPUT_DATA_PATH = os.path.join(INPUT_DATA_PATH, RAW_INPUT_DATA_FILE)

# Path to the final trained model file
MODEL_LOAD_PATH = os.path.join(PROJECT_DIR, 'dqn_inventory_model_final.zip') # Ensure this file exists


INTERNAL_ENV_PARAMS = {
    "lead_time": 2, "holding_cost": 0.2, "stockout_cost": 2.0,
    "order_cost": 5.0, "max_inventory": 100, "max_order_quantity": 50,
    "order_step_size": 10, "initial_inventory": 20,
    "include_revenue_in_reward": False,
    # Demand data needs to be loaded from file before creating the actual env for prediction
    "demand_data": np.array([0]) # Placeholder, will be replaced with loaded data
}

_LOADED_SINGLE_RL_MODEL: BaseAlgorithm | None = None # Add type hint
_SINGLE_ENV_ACTION_MAP: dict[int, int] | None = None # Add type hint
_SINGLE_MODEL_LOAD_ATTEMPTED = False

# --- Load Data Helper Function ---
def _load_demand_data(file_path: str) -> np.ndarray:
    """Loads demand data from a .npy file."""
    print(f"Attempting to load demand data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: Demand data file not found at {file_path}", file=sys.stderr)
        # In a real application, you'd handle this more robustly
        # For now, return empty array or raise error
        return np.array([], dtype=np.int32)
    try:
        data = np.load(file_path)
        print(f"Successfully loaded demand data with shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading demand data from {file_path}: {e}", file=sys.stderr)
        return np.array([], dtype=np.int32)


def _load_single_rl_model_and_map():
    """Loads the single pre-trained RL model and its action map."""
    global _LOADED_SINGLE_RL_MODEL, _SINGLE_ENV_ACTION_MAP, _SINGLE_MODEL_LOAD_ATTEMPTED

    if _LOADED_SINGLE_RL_MODEL is not None and _SINGLE_ENV_ACTION_MAP is not None:
        print("Model already loaded.") # Added log for clarity
        return True # Already loaded

    if _SINGLE_MODEL_LOAD_ATTEMPTED:
        print("Error: Previous attempt to load the RL model failed.", file=sys.stderr)
        return False # Prevent repeated failed attempts

    _SINGLE_MODEL_LOAD_ATTEMPTED = True
    print("--- Initializing RL Model Environment (first call) ---")

    # Load the actual demand data needed for the environment parameters
    # NOTE: The RL model was trained on specific demand data. Using different
    # demand data here for the environment context might be necessary for
    # Gymnasium setup but doesn't change the fact the *model's policy* is
    # based on the data it was trained on. If your model training used
    # demand_train_single_product.npy, load that here.
    model_env_demand_data = _load_demand_data(INPUT_TRAIN_PATH) # Load the training data
    if model_env_demand_data.size == 0:
         print("Failed to load demand data required for environment setup.", file=sys.stderr)
         return False

    # Create dummy env for action map and loading context
    print("Creating environment context for model loading...")
    # Update env params with the loaded demand data
    env_params_for_model_load = INTERNAL_ENV_PARAMS.copy()
    env_params_for_model_load["demand_data"] = model_env_demand_data

    try:
        temp_env_for_load = InventoryEnv(**env_params_for_model_load)
        _SINGLE_ENV_ACTION_MAP = temp_env_for_load._action_to_quantity # Store the map
        # IMPORTANT: The environment created here is ONLY for getting the action map
        # and providing context for Stable-Baselines3 loading. It's not used
        # for actual step simulations during prediction.
        temp_env_for_load.close() # Clean up the temporary env

        print(f"Action Map Loaded: {_SINGLE_ENV_ACTION_MAP}")

        # Load the model
        print(f"Loading trained DQN model from {MODEL_LOAD_PATH}...")
        if not os.path.exists(MODEL_LOAD_PATH):
             print(f"Error: Model file not found at {MODEL_LOAD_PATH}", file=sys.stderr)
             return False

        # Create a new environment instance to pass to .load() if stable_baselines3 requires it
        # Stable-Baselines3 v2+ often requires an env to be passed to load.
        # This env should ideally match the env used during training in terms of specs.
        # Using the same env_params_for_model_load should be appropriate.
        env_for_sb3_load = InventoryEnv(**env_params_for_model_load)
        _LOADED_SINGLE_RL_MODEL = DQN.load(MODEL_LOAD_PATH, env=env_for_sb3_load)
        # Close the env created *specifically* for stable_baselines3 load
        env_for_sb3_load.close()


        print("Single RL Model loaded successfully.")
        return True

    except Exception as e:
        print(f"Error during RL Model initialization or loading: {e}", file=sys.stderr)
        # Include traceback for more detailed debugging
        import traceback
        traceback.print_exc(file=sys.stderr)
        _LOADED_SINGLE_RL_MODEL = None # Ensure state reflects failure
        _SINGLE_ENV_ACTION_MAP = None
        return False


# --- Internal Function: Predict RL Action (Uses the single loaded model) ---
def _predict_single_rl_action(current_inventory: float) -> tuple[int, int]:
    """Internal function using the globally loaded single model."""
    model = _LOADED_SINGLE_RL_MODEL
    action_map = _SINGLE_ENV_ACTION_MAP

    if model is None or action_map is None:
        print("Prediction Failed: RL model or action map not loaded.", file=sys.stderr)
        return -1, -1
    if current_inventory < 0:
        print(f"Warning: Received negative inventory ({current_inventory}). Using 0 for prediction.", file=sys.stderr)
        current_inventory = 0

    try:
        # Stable-Baselines3 expects a NumPy array as input observation
        observation = np.array([current_inventory], dtype=np.float32)
        # Predict the action index
        # NOTE: deterministic=True chooses the action with the highest Q-value
        action_array, _ = model.predict(observation, deterministic=True)
        action_index = int(action_array[0]) # Ensure it's a plain int, handle potential array return

        # Map the action index to the actual quantity
        actual_order_quantity = action_map.get(action_index, -1)

        if actual_order_quantity == -1:
            print(f"Error: Predicted action index {action_index} is not in the action map {_SINGLE_ENV_ACTION_MAP}. This indicates a mismatch.", file=sys.stderr)
            return action_index, -1

        return action_index, actual_order_quantity

    except Exception as e:
        print(f"Error during internal RL prediction: {e}", file=sys.stderr)
        # Include traceback for more detailed debugging
        import traceback
        traceback.print_exc(file=sys.stderr)
        return -1, -1


# --- create_product_id function (remains unchanged) ---
def create_product_id(state_input: str, item_description: str) -> str:
    """
    Generates a standardized M5 product ID string based on limited inputs.

    Args:
        state_input: The US state, either full name or 2-letter abbreviation.
                     Must be one of: California (CA), Wisconsin (WI), Texas (TX).
                     Case-insensitive.
        item_description: A string describing the item (e.g., "soap", "batteries").

    Returns:
        A formatted product ID string (e.g., "HOUSEHOLD_1_001_CA_1").

    Raises:
        ValueError: If the provided `state_input` is invalid.
    """
    print(f"\nAttempting to create Product ID for State: '{state_input}', Desc: '{item_description}'")

    category = "HOUSEHOLD"
    department = "1"
    item_number = "001"
    store_number = "1"

    state_map = {
        "california": "CA", "ca": "CA",
        "wisconsin": "WI", "wi": "WI",
        "texas": "TX", "tx": "TX",
    }
    valid_states_msg = "California (CA), Wisconsin (WI), or Texas (TX)"

    state_abbr = state_map.get(state_input.lower())

    if state_abbr is None:
        error_message = (
            f"Invalid state input '{state_input}'. "
            f"State must be one of: {valid_states_msg}."
        )
        print(f"Error: {error_message}", file=sys.stderr)
        raise ValueError(error_message)

    product_id_generated = f"{category}_{department}_{item_number}_{state_abbr}_{store_number}"
    print(f"Generated Product ID: {product_id_generated}")
    return product_id_generated


# --- get_order_recommendation function (Uses internal ID generation) ---
@mcp.tool()
def get_order_recommendation(
    state_input: str,
    item_description: str,
    current_inventory: float
    ) -> int:
    """
    Generates the product ID from given state name and item name, then provides an order quantity
    recommendation for warehouse using a pre-trained RL model based on that product ID.

    Accepts state and item description to create the product ID first.

    Args:
        state_input: The US state needed to create the product ID.
        item_description: The item description needed to create the product ID.
        current_inventory: The current on-hand inventory level for this product.

    Returns:
        The recommended order quantity (int).
        Returns -1 if product ID generation fails OR prediction fails (e.g., model not loaded, error).
        Returns 0 if the prediction is invalid but inventory is high (safety) or a non-critical error occurs.
    """
    print(f"\n--- Starting Order Recommendation Process ---")
    print(f"Input State: '{state_input}', Item Desc: '{item_description}', Current Inventory: {current_inventory}")

    # 1. Generate the product_id using the provided inputs
    try:
        # Call create_product_id internally
        product_id = create_product_id(state_input, item_description)
        # print(f"Successfully generated Product ID: {product_id}") # create_product_id already prints this

    except ValueError as e:
        # Handle the error from create_product_id
        print(f"Recommendation Failed: Could not generate product ID. Error: {e}", file=sys.stderr)
        return -1 # Indicate failure early if product ID cannot be created

    # Log the product ID being used conceptually
    print(f"Proceeding with recommendation for conceptual product ID: {product_id}")


    # 2. Ensure the single model and action map are loaded
    #    This step conceptually uses the 'product_id' to *know which model to load*,
    #    but in THIS code, due to the caveat, it always loads the same single model.
    if not _load_single_rl_model_and_map():
        print("Recommendation Failed: Could not load RL model.", file=sys.stderr)
        return -1 # Indicate critical failure

    # 3. Get the prediction using the internal function
    # print(f"Proceeding to get prediction for {product_id}...") # Already logged above
    action_index, recommended_quantity = _predict_single_rl_action(
        current_inventory
    )

    # 4. Return the result
    if recommended_quantity != -1: # Assuming _predict returns -1 on error
        print(f"Recommendation Result for {product_id}: Order Quantity = {recommended_quantity}")
        return recommended_quantity
    else:
        # Handle prediction error - maybe return 0 as a safe default?
        print(f"Recommendation Failed for {product_id}: Error during prediction. Returning safe default (0).", file=sys.stderr)
        return 0
