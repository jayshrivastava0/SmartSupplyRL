import numpy as np
from ..rl_environment.inventory_env import InventoryEnv # For type hinting
from ..utils.helpers import find_closest_action_index
from ..utils.logger import setup_logger

# logger = setup_logger(__name__) # Module-level logger

def run_baseline_policy(env: InventoryEnv,
                        policy_type: str = 'order_up_to',
                        policy_params: dict = None,
                        verbose: bool = False) -> dict:
    """
    Runs a simple baseline heuristic policy on the environment for one episode.
    Adapted from notebook cell 37424dfb and 97463152 (for trajectory returns).

    Args:
        env: The instantiated InventoryEnv environment.
        policy_type: The type of baseline policy ('order_up_to', 'fixed_order').
        policy_params: Dictionary of parameters for the chosen policy.
            - For 'order_up_to': {'level': int}
            - For 'fixed_order': {'order_interval': int, 'fixed_quantity': int}
        verbose: If True, log step-by-step details at INFO level.

    Returns:
        A dictionary containing performance metrics and trajectories for the episode.
    """
    logger = env.logger # Use the logger from the env instance for context if available
    if policy_params is None: policy_params = {}

    obs, info_reset = env.reset() # info_reset might be basic
    terminated, truncated = False, False

    # Metrics Tracking
    total_reward = 0.0
    rewards_list = []
    inventory_trajectory = [obs[0]] # Start with initial inventory
    action_trajectory = [] # Stores actual quantities ordered
    
    stockout_events = 0
    order_events = 0
    total_ordered_quantity = 0
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_order_placement_cost = 0.0
    total_sales = 0
    total_unmet_demand = 0
    step_count = 0

    # Use info from previous step for on_order_inventory decision making
    # For the first step, on_order_inventory is effectively 0 from env reset.
    # `info` will be populated by `env.step`
    current_step_info = info_reset 

    while not terminated and not truncated:
        current_inventory = obs[0]
        # Get on-order inventory. The env.step() info dict should have this.
        # For the first step, info_reset may not have 'on_order_pipeline_total'.
        on_order_inventory = current_step_info.get('on_order_pipeline_total', 0) if step_count > 0 else 0


        target_order_quantity = 0
        if policy_type == 'order_up_to':
            level = policy_params.get('level', 50)
            if level < 0:
                 logger.warning("'order_up_to' level cannot be negative. Using 0.")
                 level = 0
            # Inventory position = current_inventory + on_order_inventory
            inventory_position = current_inventory + on_order_inventory
            target_order_quantity = max(0, level - inventory_position)

        elif policy_type == 'fixed_order':
            order_interval = policy_params.get('order_interval', 7)
            fixed_quantity = policy_params.get('fixed_quantity', env.max_order_quantity // 2)
            if order_interval <= 0:
                logger.warning("'fixed_order' interval must be positive. Using 1.")
                order_interval = 1
            if fixed_quantity < 0:
                logger.warning("'fixed_order' quantity cannot be negative. Using 0.")
                fixed_quantity = 0
            
            # Check if fixed_quantity is a valid order size, adjust if not
            if fixed_quantity > 0 and fixed_quantity not in env.possible_orders:
                logger.warning(f"Fixed quantity {fixed_quantity} is not an allowed order size {env.possible_orders}.")
                closest_action_idx = find_closest_action_index(env, fixed_quantity)
                fixed_quantity = env._action_to_quantity[closest_action_idx]
                logger.info(f"Adjusting fixed quantity to {fixed_quantity}.")

            if step_count % order_interval == 0:
                target_order_quantity = fixed_quantity
        else:
            logger.error(f"Unknown policy_type: {policy_type}")
            raise ValueError(f"Unknown policy_type: {policy_type}")

        action_index = find_closest_action_index(env, int(round(target_order_quantity)))
        actual_order_quantity_this_step = env._action_to_quantity[action_index]
        action_trajectory.append(actual_order_quantity_this_step)

        if verbose:
            logger.info(
                f"Baseline Step {step_count + 1} (Day {current_step_info.get('day_just_finished', -1)+1 if step_count > 0 else 0 }): "
                f"Inv={current_inventory:.1f}, OnOrder={on_order_inventory}, TargetQ={target_order_quantity:.1f} "
                f"-> Action={action_index}(Order {actual_order_quantity_this_step})"
            )

        obs, reward, terminated, truncated, current_step_info = env.step(action_index)
        
        # Record Metrics from current_step_info (which reflects results of the action taken)
        total_reward += reward
        rewards_list.append(reward)
        inventory_trajectory.append(current_step_info['inventory_eod'])
        
        if current_step_info['unmet_demand'] > 0: stockout_events += 1
        if current_step_info['ordered_quantity'] > 0:
            order_events += 1
            total_ordered_quantity += current_step_info['ordered_quantity']
        
        total_holding_cost += current_step_info['holding_cost_incurred']
        total_stockout_cost += current_step_info['stockout_cost_incurred']
        total_order_placement_cost += current_step_info['order_placement_cost_incurred']
        total_sales += current_step_info['sales_made']
        total_unmet_demand += current_step_info['unmet_demand']


        if verbose:
            logger.info(
                f"  -> Demand={current_step_info['demand_faced']}, Sales={current_step_info['sales_made']}, "
                f"Unmet={current_step_info['unmet_demand']}, Reward={reward:.2f}, NextInv={obs[0]:.1f}"
            )
        step_count += 1
        if step_count > env.episode_length * 1.1: # Safety break
            logger.warning("Exceeded expected episode length in baseline. Breaking loop.")
            break
    
    avg_inventory = np.mean(inventory_trajectory[:-1]) if inventory_trajectory else 0 # Avg over EOD inv levels
    stockout_percentage = (stockout_events / step_count * 100) if step_count > 0 else 0
    order_frequency_percentage = (order_events / step_count * 100) if step_count > 0 else 0
    avg_order_quantity = (total_ordered_quantity / order_events) if order_events > 0 else 0

    results = {
        "policy_type": f"Baseline ({policy_type})",
        "policy_params": policy_params,
        "total_reward": total_reward,
        "average_eod_inventory": avg_inventory,
        "total_steps": step_count,
        "stockout_days": stockout_events,
        "stockout_percentage": stockout_percentage,
        "order_days": order_events,
        "order_frequency_percentage": order_frequency_percentage,
        "average_order_quantity": avg_order_quantity,
        "final_inventory": obs[0] if step_count > 0 else env.initial_inventory,
        "total_holding_cost": total_holding_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_order_placement_cost": total_order_placement_cost,
        "total_sales": total_sales,
        "total_unmet_demand": total_unmet_demand,
        "inventory_trajectory": inventory_trajectory,
        "reward_trajectory": rewards_list,
        "action_trajectory": action_trajectory 
    }
    logger.info(f"Baseline Policy ({policy_type}) Results: Total Reward={total_reward:.2f}, AvgInv={avg_inventory:.2f}, Stockout%={stockout_percentage:.2f}%")
    return results