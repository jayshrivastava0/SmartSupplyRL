import numpy as np
import pandas as pd # For creating results DataFrame
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.utils.env_checker import check_env # For checking custom env
from .base_rl_agent import BaseRLAgent
from ..rl_environment.inventory_env import InventoryEnv # For type hinting and action map access

class DQNAgent(BaseRLAgent):
    def train(self, train_demand_data: np.ndarray):
        self.logger.info(f"Starting DQN training for {self.product_id_full}...")
        
        # Create vectorized training environment
        train_env_vectorized = self.create_env(train_demand_data, vectorized=True)
        
        # Optional: Check underlying environment (good practice for custom envs)
        try:
            # SB3 VecEnv wraps envs in Monitor by default if not already wrapped.
            # Accessing .envs[0] might give Monitor, then .env for original.
            original_env_instance = train_env_vectorized.envs[0]
            if isinstance(original_env_instance, Monitor):
                original_env_instance = original_env_instance.env
            check_env(original_env_instance)
            self.logger.info("Underlying InventoryEnv check passed!")
        except Exception as e:
            self.logger.error(f"InventoryEnv check failed: {e}", exc_info=True)
            train_env_vectorized.close()
            return # Stop training if env is problematic
            
        # Checkpoint Callback
        checkpoint_save_path = self.model_dir / f"{self.model_save_name_base}_ckpt" # SB3 adds _<steps>.zip
        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.agent_params.get('save_freq_checkpoints', 10000) // train_env_vectorized.num_envs, 1),
            save_path=str(self.model_dir), # Directory to save checkpoints
            name_prefix=f"{self.model_save_name_base}_ckpt", # Prefix for checkpoint files
            save_replay_buffer=False, # Usually not needed for checkpoints
            save_vecnormalize=False # Not using VecNormalize here
        )
        
        # Optional: Evaluation Callback for early stopping or best model saving
        # eval_env = self.create_env(train_demand_data[-200:], vectorized=False) # Use a small validation set from end of train
        # eval_callback = EvalCallback(eval_env, best_model_save_path=str(self.model_dir),
        #                              log_path=str(self.tensorboard_log_dir / 'eval_logs'),
        #                              eval_freq=max(5000 // train_env_vectorized.num_envs, 1),
        #                              deterministic=True, render=False,
        #                              n_eval_episodes=3,
        #                              name_prefix=f"{self.model_save_name_base}_best")


        self.model = DQN(
            policy=self.agent_params.get("policy_network", "MlpPolicy"),
            env=train_env_vectorized,
            verbose=self.agent_params.get("verbose", 1),
            learning_rate=self.agent_params.get("learning_rate", 1e-4),
            buffer_size=self.agent_params.get("buffer_size", 50000),
            learning_starts=self.agent_params.get("learning_starts", 1000),
            batch_size=self.agent_params.get("batch_size", 64),
            tau=self.agent_params.get("tau", 1.0),
            gamma=self.agent_params.get("gamma", 0.99),
            train_freq=self.agent_params.get("train_freq", 4),
            gradient_steps=self.agent_params.get("gradient_steps", 1),
            exploration_fraction=self.agent_params.get("exploration_fraction", 0.15),
            exploration_initial_eps=self.agent_params.get("exploration_initial_eps", 1.0),
            exploration_final_eps=self.agent_params.get("exploration_final_eps", 0.05),
            target_update_interval=self.agent_params.get("target_update_interval", 1000),
            tensorboard_log=str(self.tensorboard_log_dir),
            seed=self.agent_params.get("seed", None),
            policy_kwargs=self.agent_params.get("policy_kwargs", None)
        )
            
        self.logger.info(f"DQN Model Defined. Policy Architecture:\n{self.model.policy}")
        self.logger.info(f"Logging TensorBoard data to: {self.tensorboard_log_dir}")

        training_successful = False
        try:
            self.model.learn(
                total_timesteps=self.agent_params.get("total_timesteps", 100000),
                log_interval=self.agent_params.get("log_interval", 10), # Log summary stats this many Dones
                callback=[checkpoint_callback], # Add eval_callback here if using
                tb_log_name="DQN_run", # Name for this run in TensorBoard
                reset_num_timesteps=False # If resuming training, set to False
            )
            training_successful = True
            self.logger.info("DQN Training Completed.")
        except Exception as e:
            self.logger.error(f"An error occurred during DQN training: {e}", exc_info=True)
        finally:
            if training_successful:
                self.save_model(suffix="final_trained")
            else:
                self.logger.warning("DQN training did not complete successfully or was interrupted.")
                # Optionally save interrupted model if self.model exists
                if self.model:
                    self.save_model(suffix="interrupted")
            train_env_vectorized.close()
            # if eval_env: eval_env.close()

    def evaluate_agent_performance(self, test_demand_data: np.ndarray, verbose: bool = False) -> dict:
        """ Evaluates the trained DQN agent. Adapted from notebook cell 97463152. """
        self.logger.info(f"Evaluating DQN agent for {self.product_id_full}...")
        if self.model is None:
            self.logger.error("DQN model not trained or loaded. Cannot evaluate.")
            return {}
            
        eval_env = self.create_env(test_demand_data, vectorized=False)
        
        obs, info_reset = eval_env.reset()
        terminated, truncated = False, False

        # Metrics Tracking
        total_reward = 0.0
        rewards_list = []
        inventory_trajectory = [obs[0]] # Initial inventory
        action_idx_trajectory = []
        action_qty_trajectory = []
        
        stockout_events = 0
        order_events = 0
        total_ordered_quantity = 0
        total_holding_cost = 0.0
        total_stockout_cost = 0.0
        total_order_placement_cost = 0.0
        total_sales = 0
        total_unmet_demand = 0
        step_count = 0

        while not terminated and not truncated:
            action_index, action_quantity = self.predict_action(obs, env_instance=eval_env, deterministic=True)
            action_idx_trajectory.append(action_index)
            action_qty_trajectory.append(action_quantity)

            if verbose:
                 self.logger.info(
                    f"Eval Step {step_count + 1}: Inv={obs[0]:.1f} "
                    f"-> Agent Action: Idx={action_index} (Order {action_quantity})"
                )
            
            obs, reward, terminated, truncated, step_info = eval_env.step(action_index)

            total_reward += reward
            rewards_list.append(reward)
            inventory_trajectory.append(step_info['inventory_eod'])
            
            if step_info['unmet_demand'] > 0: stockout_events += 1
            if step_info['ordered_quantity'] > 0:
                order_events += 1
                total_ordered_quantity += step_info['ordered_quantity']

            total_holding_cost += step_info['holding_cost_incurred']
            total_stockout_cost += step_info['stockout_cost_incurred']
            total_order_placement_cost += step_info['order_placement_cost_incurred']
            total_sales += step_info['sales_made']
            total_unmet_demand += step_info['unmet_demand']

            if verbose:
                self.logger.info(
                    f"  -> Demand={step_info['demand_faced']}, Sales={step_info['sales_made']}, Unmet={step_info['unmet_demand']}, "
                    f"Reward={reward:.2f}, NextInv={obs[0]:.1f}"
                )
            step_count += 1
            if step_count > eval_env.episode_length * 1.1: # Safety break
                self.logger.warning("Exceeded expected episode length during evaluation. Breaking loop.")
                break
        
        eval_env.close()

        avg_inventory = np.mean(inventory_trajectory[:-1]) if inventory_trajectory else 0
        stockout_percentage = (stockout_events / step_count * 100) if step_count > 0 else 0
        order_frequency_percentage = (order_events / step_count * 100) if step_count > 0 else 0
        avg_order_quantity = (total_ordered_quantity / order_events) if order_events > 0 else 0
        
        results = {
            "policy_type": "DQN Agent",
            "total_reward": total_reward,
            "average_eod_inventory": avg_inventory,
            "total_steps": step_count,
            "stockout_days": stockout_events,
            "stockout_percentage": stockout_percentage,
            "order_days": order_events,
            "order_frequency_percentage": order_frequency_percentage,
            "average_order_quantity": avg_order_quantity,
            "final_inventory": obs[0] if step_count > 0 else eval_env.initial_inventory, # obs is last obs
            "total_holding_cost": total_holding_cost,
            "total_stockout_cost": total_stockout_cost,
            "total_order_placement_cost": total_order_placement_cost,
            "total_sales": total_sales,
            "total_unmet_demand": total_unmet_demand,
            "inventory_trajectory": inventory_trajectory,
            "reward_trajectory": rewards_list,
            "action_idx_trajectory": action_idx_trajectory,
            "action_qty_trajectory": action_qty_trajectory
        }
        self.logger.info(f"DQN Agent Evaluation Results: Total Reward={total_reward:.2f}, AvgInv={avg_inventory:.2f}, Stockout%={stockout_percentage:.2f}%")
        return results


    def predict_action(self, observation: np.ndarray, env_instance: InventoryEnv, deterministic: bool = True) -> tuple[int, int]:
        if self.model is None:
            self.logger.error("DQN model not loaded. Cannot predict action.")
            return -1, -1 

        action_array, _ = self.model.predict(observation, deterministic=deterministic)
        action_index = int(action_array.item()) # .item() if it's a 0-dim array from SB3

        if not (0 <= action_index < len(env_instance._action_to_quantity)):
            self.logger.error(f"Predicted action index {action_index} is out of bounds for env's action map.")
            return action_index, -1 # Return invalid index and -1 quantity

        actual_order_quantity = env_instance._action_to_quantity.get(action_index)
        return action_index, actual_order_quantity


    def save_model(self, suffix: str = "final_trained"):
        if self.model is None:
            self.logger.error("No DQN model instance to save.")
            return
        
        full_save_path = self._get_full_model_path(suffix)
        try:
            self.model.save(str(full_save_path))
            self.logger.info(f"DQN model saved to: {full_save_path}")
        except Exception as e:
            self.logger.error(f"Error saving DQN model: {e}", exc_info=True)

    def load_model(self, suffix: str = "final_trained") -> bool:
        load_path = self._get_full_model_path(suffix)
        if not load_path.exists():
            self.logger.error(f"DQN model file not found: {load_path}")
            self.model = None
            return False
        try:
            # When loading, SB3 needs an environment to set up spaces, even if it's a dummy one.
            # We use the env_params stored with the agent to create this dummy context.
            # No actual demand data is needed for the dummy env context at load time.
            dummy_env_for_load = self.create_env(demand_data=np.array([0]), vectorized=False)
            self.model = DQN.load(str(load_path), env=dummy_env_for_load)
            dummy_env_for_load.close() # Close the temporary environment
            self.logger.info(f"DQN model loaded from: {load_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading DQN model: {e}", exc_info=True)
            self.model = None
            return False