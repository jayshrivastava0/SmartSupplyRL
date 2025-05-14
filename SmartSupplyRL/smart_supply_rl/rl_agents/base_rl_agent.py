from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from ..utils.config_loader import Config
from ..utils.logger import setup_logger
from ..rl_environment.inventory_env import InventoryEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

class BaseRLAgent(ABC):
    def __init__(self, product_id_full: str, config: Config, 
                 agent_specific_params: dict, env_params_for_agent: dict):
        """
        Abstract base class for RL agents.

        Args:
            product_id_full: Full product ID (e.g., FOODS_3_090_CA_3_evaluation).
            config: Global configuration object.
            agent_specific_params: Hyperparameters specific to the RL algorithm (e.g., DQN learning rate).
            env_params_for_agent: Parameters to instantiate InventoryEnv for this agent.
        """
        self.product_id_full = product_id_full
        self.config = config
        self.agent_params = agent_specific_params
        self.env_params = env_params_for_agent # Store env params used with this agent
        self.model: BaseAlgorithm | None = None # Will hold the SB3 model instance
        self.logger = setup_logger(f"{self.__class__.__name__}-{self.product_id_full.replace('_evaluation', '')}")
        
        # Model save path base name (without .zip or specific suffixes)
        self.model_save_name_base = f"{self.__class__.__name__.lower()}_{self.product_id_full}"
        self.model_dir = self.config.get_path('output_dir_abs', 'models', 'rl')
        # self.model_dir.mkdir(parents=True, exist_ok=True) # Handled by Config

        self.tensorboard_log_dir = self.config.get_path(
            'output_dir_abs', 'logs', 'tensorboard', 
            f"{self.__class__.__name__.lower()}_{self.product_id_full.replace('_evaluation','')}"
        )
        # self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True) # Handled by Config

    def _get_full_model_path(self, suffix: str) -> Path:
        """Constructs the full path for a model file with a given suffix."""
        return self.model_dir / f"{self.model_save_name_base}_{suffix}.zip"

    def _make_env_fn(self, demand_data: np.ndarray):
        """Returns a function that creates an InventoryEnv instance."""
        def _init():
            # Pass the agent-specific env_params
            return InventoryEnv(demand_data=demand_data, env_params=self.env_params)
        return _init

    def create_env(self, demand_data: np.ndarray, vectorized: bool = True) -> InventoryEnv | VecEnv :
        """Creates an InventoryEnv instance, optionally vectorized."""
        if vectorized:
            return DummyVecEnv([self._make_env_fn(demand_data)])
        else:
            # Pass the agent-specific env_params
            return InventoryEnv(demand_data=demand_data, env_params=self.env_params)

    @abstractmethod
    def train(self, train_demand_data: np.ndarray):
        """Trains the RL agent."""
        pass

    @abstractmethod
    def evaluate_agent_performance(self, test_demand_data: np.ndarray, verbose: bool = False) -> dict:
        """Evaluates the trained agent on test data and returns performance metrics."""
        pass

    @abstractmethod
    def predict_action(self, observation: np.ndarray, env_instance: InventoryEnv, deterministic: bool = True) -> tuple[int, int]:
        """
        Predicts an action (index and quantity) given an observation.
        Args:
            observation: Current environment observation.
            env_instance: An instance of InventoryEnv to map action index to quantity.
                          This is needed because the mapping depends on env's `possible_orders`.
            deterministic: Whether to use deterministic actions (True for eval/prod).
        Returns:
            Tuple (action_index, action_quantity). Returns (-1, -1) on error.
        """
        pass
            
    @abstractmethod
    def save_model(self, suffix: str = "final_trained"):
        """Saves the trained agent's model."""
        pass

    @abstractmethod
    def load_model(self, suffix: str = "final_trained") -> bool:
        """
        Loads a pre-trained agent's model.
        Returns True if successful, False otherwise.
        """
        pass