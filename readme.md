# Smart Supply RL: Inventory Optimization Project

This project implements a Reinforcement Learning (RL) based system for optimizing inventory decisions for products from the M5 forecasting competition dataset. It also includes traditional forecasting models (Prophet, ARIMA) for comparison and potential use in feature engineering.

## Project Structure

```
smart_supply_rl/
├── config/                 # Configuration files (YAML)
│   ├── main_config.yaml
│   ├── prophet_params.yaml
│   └── dqn_params.yaml
├── data/
│   └── M5_dataset/         # Raw M5 data (sales, calendar, prices) - Gitignored
├── output/                 # Generated files (processed data, models, results, logs) - Gitignored
│   ├── processed_data/
│   ├── models/
│   ├── results/
│   └── logs/
├── smart_supply_rl/        # Main Python package
│   ├── __init__.py
│   ├── api/                # User-facing services (e.g., recommendation)
│   ├── baselines/          # Heuristic inventory policies
│   ├── data_processing/    # Data loading, parsing, feature engineering
│   ├── forecasting/        # Forecasting models (Prophet, ARIMA)
│   ├── rl_agents/          # RL agent implementations (DQN, PPO)
│   ├── rl_environment/     # Custom inventory Gym environment
│   └── utils/              # Utility functions (config, logger, helpers, plotting)
├── main.py                 # Command-Line Interface (CLI) for running pipeline steps
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd smart_supply_rl
    ```

2.  **Create a Python virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download M5 Data:**
    Place the M5 competition CSV files (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`) into the `data/M5_dataset/` directory.

5.  **Configure Paths (if needed):**
    Review `config/main_config.yaml` to ensure `data_dir` and `output_dir` are set correctly relative to your project root if you deviate from the default structure.

## Usage (CLI - `main.py`)

The `main.py` script provides a command-line interface to run various parts of the pipeline.

**General Usage:**
```bash
python main.py <command> [options]
```

**Available Commands:**

*   `preprocess_data`: Loads and preprocesses M5 sales data for a specific product.
    ```bash
    python main.py preprocess_data --product_id_full FOODS_3_090_CA_3_evaluation [--train_days 1500]
    ```

*   `analyze_timeseries`: Performs ADF tests and plots ACF/PACF for a product's training data.
    ```bash
    python main.py analyze_timeseries --product_id_full FOODS_3_090_CA_3_evaluation
    ```

*   `train_forecaster`: Trains a forecasting model (Prophet or ARIMA).
    ```bash
    python main.py train_forecaster --product_id_full FOODS_3_090_CA_3_evaluation --model_type prophet
    python main.py train_forecaster --product_id_full FOODS_3_090_CA_3_evaluation --model_type arima
    ```

*   `train_rl`: Trains an RL agent (e.g., DQN).
    ```bash
    python main.py train_rl --product_id_full FOODS_3_090_CA_3_evaluation --agent_type dqn
    ```

*   `evaluate_rl`: Evaluates a trained RL agent against a baseline heuristic policy.
    ```bash
    python main.py evaluate_rl --product_id_full FOODS_3_090_CA_3_evaluation --agent_type dqn [--model_suffix final_trained] [--baseline_level 60] [--verbose]
    ```

*   `get_recommendation`: Gets an order quantity recommendation from a trained RL model.
    ```bash
    python main.py get_recommendation --product_id_full FOODS_3_090_CA_3_evaluation --inventory 25 [--model_suffix final_trained]
    ```


![image](https://github.com/user-attachments/assets/0c12cee3-4a81-4d1d-899d-a60f0189f4f0)

For more details on options for each command:
```bash
python main.py <command> --help
```

## TensorBoard Monitoring

During RL agent training (`train_rl`), logs are saved to `output/logs/tensorboard/`. You can monitor training progress using TensorBoard:
```bash
tensorboard --logdir output/logs/tensorboard/
```
Then navigate to `http://localhost:6006` in your browser.
