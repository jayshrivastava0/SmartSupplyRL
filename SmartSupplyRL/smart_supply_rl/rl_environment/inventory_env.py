import gymnasium as gym
import numpy as np
from gymnasium import spaces
import warnings
from ..utils.logger import setup_logger # Assuming logger.py is in ../utils

class InventoryEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating a single-product inventory system.
    Adapted from notebook cell e2cad843.
    """
    metadata = {"render_modes": [], "render_fps": 1} # No rendering implemented

    def __init__(self,
                 demand_data: np.ndarray,
                 env_params: dict # Dictionary containing all other parameters
                ):
        super().__init__()
        self.logger = setup_logger(f"{self.__class__.__name__}") # Instance-specific logger

        # Unpack env_params or use defaults
        self.lead_time = env_params.get('lead_time', 1)
        self.holding_cost = env_params.get('holding_cost', 0.1)
        self.stockout_cost = env_params.get('stockout_cost', 1.0)
        self.order_cost = env_params.get('order_cost', 0.0)
        self.unit_price = env_params.get('unit_price', 2.0) # Used if include_revenue_in_reward
        self.max_inventory = env_params.get('max_inventory', 1000)
        self.max_order_quantity = env_params.get('max_order_quantity', 100)
        self.order_step_size = env_params.get('order_step_size', 10)
        initial_inventory_raw = env_params.get('initial_inventory', 0)
        self.include_revenue_in_reward = env_params.get('include_revenue_in_reward', False)

        if not isinstance(demand_data, np.ndarray) or demand_data.ndim != 1:
            self.logger.error("demand_data must be a 1D NumPy array.")
            raise ValueError("demand_data must be a 1D NumPy array.")
        if self.lead_time < 0:
            self.logger.error("lead_time cannot be negative.")
            raise ValueError("lead_time cannot be negative.")
        if self.max_inventory <= 0:
             self.logger.error("max_inventory must be positive.")
             raise ValueError("max_inventory must be positive.")
        if not (self.max_order_quantity >= 0 and self.order_step_size > 0 and \
                (self.max_order_quantity == 0 or self.max_order_quantity % self.order_step_size == 0)):
            self.logger.error(f"Invalid order quantity parameters: max_order_quantity={self.max_order_quantity}, order_step_size={self.order_step_size}")
            raise ValueError("Invalid order quantity parameters.")

        self.demand_data = demand_data.astype(np.int32)
        self.episode_length = len(demand_data)
        
        self.initial_inventory = min(initial_inventory_raw, self.max_inventory)

        # --- Define Action Space ---
        self.possible_orders = np.arange(0, self.max_order_quantity + self.order_step_size, self.order_step_size, dtype=np.int32)
        if not np.isin(0, self.possible_orders) and self.max_order_quantity > 0 : # Ensure 0 is an option if max_order_quantity > 0
            # This case should not happen if max_order_quantity % order_step_size == 0 and max_order_q >=0
             self.logger.warning("Order step size does not allow ordering 0. This might be unintended unless max_order_quantity is 0.")
        
        self.action_space = spaces.Discrete(len(self.possible_orders))
        self._action_to_quantity = {i: q for i, q in enumerate(self.possible_orders)}
        self.logger.info(f"Action space: {self.action_space}, Num actions: {len(self.possible_orders)}")
        self.logger.debug(f"Action -> Quantity mapping: {self._action_to_quantity}")

        # --- Define Observation Space ---
        # Current on-hand inventory
        self.observation_space = spaces.Box(
            low=0, high=self.max_inventory, shape=(1,), dtype=np.float32
        )
        self.logger.info(f"Observation space: {self.observation_space}")

        # State Variables
        self.current_day_index = 0
        self.current_inventory = 0.0 # Will be float due to observation space, set in reset()
        self.on_order_inventory = {} # {arrival_day: quantity}

        if self.episode_length == 0:
            self.logger.warning("Demand data is empty. Environment cannot effectively run episodes.")

    def _get_obs(self) -> np.ndarray:
        return np.array([self.current_inventory], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "day_index": self.current_day_index, # Day about to start or just finished
            "current_inventory": self.current_inventory,
            "on_order_pipeline_total": sum(self.on_order_inventory.values()),
        }

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed) # Important for seeding internal RNG
        self.current_day_index = 0
        self.current_inventory = float(self.initial_inventory) # Ensure float for consistency with obs space
        self.on_order_inventory = {}
        
        observation = self._get_obs()
        info = self._get_info() # Basic info at reset
        self.logger.debug(f"Environment Reset: Day={self.current_day_index}, Inv={self.current_inventory:.2f}")
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_day_index >= self.episode_length:
             self.logger.warning("Step called after environment termination. Returning terminal state.")
             obs = self._get_obs() # Current (likely final) observation
             info = self._get_info()
             info["warning"] = "Step called after termination"
             return obs, 0.0, True, False, info # 0 reward, terminated=True

        # 1. Get Actual Order Quantity from Action
        if action < 0 or action >= len(self._action_to_quantity):
            self.logger.error(f"Invalid action {action} received. Max action index is {len(self._action_to_quantity)-1}.")
            # Handle invalid action, e.g., treat as "do nothing" or raise error
            actual_order_quantity = 0 # Default to no order for safety
        else:
            actual_order_quantity = self._action_to_quantity[action]
        
        current_order_placed_cost = self.order_cost if actual_order_quantity > 0 else 0.0

        # 2. Check for Arriving Orders
        arriving_today = self.on_order_inventory.pop(self.current_day_index, 0)
        
        # 3. Update Inventory with Arrivals (Cap at Max)
        inventory_before_arrival = self.current_inventory
        self.current_inventory += arriving_today
        overflow = max(0, self.current_inventory - self.max_inventory)
        if overflow > 0:
            self.logger.debug(f"Day {self.current_day_index}: Overflow of {overflow} units due to arrival. Inventory capped.")
        self.current_inventory = min(self.current_inventory, self.max_inventory)
        inventory_after_arrival = self.current_inventory
        
        # 4. Get Demand for the Current Day
        demand = self.demand_data[self.current_day_index]
        
        # 5. Calculate Sales and Unmet Demand
        sales = min(demand, self.current_inventory)
        unmet_demand = demand - sales
        
        # 6. Update Inventory After Sales
        self.current_inventory -= sales
        inventory_end_of_day = self.current_inventory
        
        # 7. Calculate Costs for the Day
        holding_cost_today = inventory_end_of_day * self.holding_cost # Cost on ending inventory
        stockout_cost_today = unmet_demand * self.stockout_cost
        
        # 8. Calculate Reward
        total_cost_today = holding_cost_today + stockout_cost_today + current_order_placed_cost
        if self.include_revenue_in_reward:
            revenue_today = sales * self.unit_price
            reward = revenue_today - total_cost_today
        else:
            reward = -total_cost_today # Minimize cost (negative reward)
            if overflow > 0: # Optional: Penalize overflow
                 # reward -= overflow * self.holding_cost # e.g. treat overflow as held for one day
                 pass

        # 9. Place the New Order (if any)
        if actual_order_quantity > 0:
            arrival_day = self.current_day_index + self.lead_time
            self.on_order_inventory[arrival_day] = self.on_order_inventory.get(arrival_day, 0) + actual_order_quantity
        
        # Log step details (example)
        self.logger.debug(
            f"Day {self.current_day_index}: Action={action}({actual_order_quantity}), "
            f"InvStart={inventory_before_arrival:.1f}, Arrived={arriving_today}, InvAfterArrival={inventory_after_arrival:.1f}, "
            f"Demand={demand}, Sales={sales}, Unmet={unmet_demand}, InvEnd={inventory_end_of_day:.1f}, "
            f"Costs(H={holding_cost_today:.2f}, S={stockout_cost_today:.2f}, O={current_order_placed_cost:.2f}), Reward={reward:.2f}"
        )

        # 10. Advance Time
        self.current_day_index += 1
        
        # 11. Check for Termination
        terminated = self.current_day_index >= self.episode_length
        truncated = False # Not using truncation based on step limits here
        
        # 12. Get Next State Observation and Info
        observation = self._get_obs()
        info = {
            "day_just_finished": self.current_day_index -1,
            "demand_faced": demand,
            "sales_made": sales,
            "unmet_demand": unmet_demand,
            "ordered_quantity": actual_order_quantity,
            "arrived_quantity": arriving_today,
            "inventory_eod": inventory_end_of_day,
            "holding_cost_incurred": holding_cost_today,
            "stockout_cost_incurred": stockout_cost_today,
            "order_placement_cost_incurred": current_order_placed_cost,
            "reward_received": reward,
            "overflow_units": overflow,
            "on_order_pipeline_total": sum(self.on_order_inventory.values()),
        }
        
        return observation, reward, terminated, truncated, info

    def close(self):
        self.logger.debug("Closing InventoryEnv.")
        pass # Nothing specific needed for this simple environment

    def render(self):
        # This environment does not support rendering.
        pass