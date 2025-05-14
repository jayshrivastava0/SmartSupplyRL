import yaml
from pathlib import Path
import os

class Config:
    _instance = None

    def __new__(cls, config_dir_name="config"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            
            # Determine project root dynamically
            # Assumes this file is at smart_supply_rl/utils/config_loader.py
            # So, project_root is three levels up from this file's directory
            cls._instance.project_root = Path(__file__).resolve().parents[2]
            
            cls._instance.config_path_base = cls._instance.project_root / config_dir_name
            
            cls._instance.main_config = cls._load_yaml(cls._instance.config_path_base / "main_config.yaml")
            
            # Construct absolute paths from main_config relative paths
            cls._instance.main_config['data_dir_abs'] = cls._instance.project_root / cls._instance.main_config['data_dir']
            cls._instance.main_config['output_dir_abs'] = cls._instance.project_root / cls._instance.main_config['output_dir']

            cls._instance.prophet_params_all = cls._load_yaml(cls._instance.config_path_base / "prophet_params.yaml")
            cls._instance.dqn_params_all = cls._load_yaml(cls._instance.config_path_base / "dqn_params.yaml")
            
            # Ensure output directories exist
            cls._instance.get_path('output_dir_abs', 'processed_data').mkdir(parents=True, exist_ok=True)
            cls._instance.get_path('output_dir_abs', 'models', 'forecasting').mkdir(parents=True, exist_ok=True)
            cls._instance.get_path('output_dir_abs', 'models', 'rl').mkdir(parents=True, exist_ok=True)
            cls._instance.get_path('output_dir_abs', 'results').mkdir(parents=True, exist_ok=True)
            cls._instance.get_path('output_dir_abs', 'logs').mkdir(parents=True, exist_ok=True) # For app.log
            cls._instance.get_path('output_dir_abs', 'logs', 'tensorboard').mkdir(parents=True, exist_ok=True)


        return cls._instance

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_path(self, base_key: str, *subpaths: str) -> Path:
        """
        Constructs a path relative to a base path defined in main_config.
        Example: config.get_path('output_dir_abs', 'models', 'rl')
        """
        base_path = Path(self.main_config[base_key])
        return base_path.joinpath(*subpaths)
        
    def get_m5_file_path(self, file_key: str) -> Path:
        """
        Gets the full path to an M5 dataset file.
        Example: config.get_m5_file_path('sales')
        """
        return self.get_path('data_dir_abs', self.main_config['m5_files'][file_key])

    def get_prophet_params(self, product_id_full: str = None) -> dict:
        """Gets Prophet parameters, specific to product_id_full or default."""
        if product_id_full and product_id_full in self.prophet_params_all:
            return self.prophet_params_all[product_id_full]
        return self.prophet_params_all["default"]

    def get_dqn_params(self, product_id_full: str = None) -> dict:
        """Gets DQN parameters, specific to product_id_full or default."""
        if product_id_full and product_id_full in self.dqn_params_all:
            return self.dqn_params_all[product_id_full]
        return self.dqn_params_all["default"]
        
    def get_env_params(self, product_id_full: str = None) -> dict:
        """Helper to get 'env_config' part of DQN parameters."""
        dqn_config = self.get_dqn_params(product_id_full)
        return dqn_config.get("env_config", {})

    def get_product_specific_config(self, product_id_full: str) -> dict:
        """Gets general product-specific configurations from main_config."""
        return self.main_config.get("product_specific_configs", {}).get(product_id_full, {})

    def get_train_days_for_product(self, product_id_full: str) -> int:
        """Gets the number of training days for a specific product, falling back to default."""
        product_conf = self.get_product_specific_config(product_id_full)
        return product_conf.get("train_days_split", self.main_config.get("default_train_days_split"))

# Example Usage (typically done once and passed around or re-instantiated):
# config = Config()
# sales_file = config.get_m5_file_path('sales')
# dqn_lr = config.get_dqn_params()['learning_rate']
# env_lead_time = config.get_env_params()['lead_time']
# output_models_path = config.get_path('output_dir_abs', 'models')