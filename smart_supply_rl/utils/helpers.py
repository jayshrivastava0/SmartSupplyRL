import numpy as np
# Conditional import for type hinting to avoid circular dependency if InventoryEnv is not yet defined
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from ..rl_environment.inventory_env import InventoryEnv

def create_product_id(state_input: str, item_description: str) -> str:
    """
    Generates a standardized M5 product ID string based on limited inputs.

    This function is currently configured to generate IDs only for a fixed
    item ('HOUSEHOLD_1_001') within a fixed store ('1') in one of the
    three valid M5 competition states.
    This is from notebook cell fcd06b26.

    Args:
        state_input: The US state, either full name or 2-letter abbreviation.
                     Must be one of: California (CA), Wisconsin (WI), Texas (TX).
                     Case-insensitive.
        item_description: A string describing the item (e.g., "soap", "batteries").
                          Currently *not* used to determine category/department,
                          but included for potential future expansion.

    Returns:
        A formatted product ID string (e.g., "HOUSEHOLD_1_001_CA_1_evaluation").
        Note: The original notebook did not add "_evaluation" here, but it's common
        in M5 data. For this helper, we'll stick to the notebook's original output.
        The M5Parser class will handle variants like "_evaluation".

    Raises:
        ValueError: If the provided `state_input` does not correspond to
                    California, Wisconsin, or Texas.
    """
    # print(f"Attempting to create Product ID for State: '{state_input}', Desc: '{item_description}'") # Use logger

    # --- Fixed components based on current requirements ---
    category = "HOUSEHOLD"
    department = "1" # Note: In M5, dept_id is like HOUSEHOLD_1, not just 1.
                     # The example in notebook generated HOUSEHOLD_1_001...
                     # Let's assume the notebook's intention for this specific function.
    item_number = "001" # Needs to be a 3-digit string
    store_number = "1"  # Fixed store number

    # --- State Abbreviation Mapping ---
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
        # print(f"Error: {error_message}") # Use logger
        raise ValueError(error_message)

    # Format: CATEGORY_DEPT_ITEMNUM_STATE_STORENUM
    # The notebook example implies something like: ITEMPART_STATE_STOREPART
    # ITEMPART = CATEGORY_DEPT_ITEMNUM -> HOUSEHOLD_1_001
    # STATE -> CA
    # STOREPART -> 1 (for store CA_1)
    # product_id_generated = f"{category}_{department}_{item_number}_{state_abbr}_{store_number}"
    # The notebook `create_product_id` structure matches `TARGET_ITEM_ID` + `_` + `TARGET_STORE_ID` (without _validation or _evaluation)
    # e.g. FOODS_3_090_CA_3
    # The function seems to be constructing the "item_id" part combined with state and a store number.
    # If item_description was used, it'd be more complex.
    # Given the fixed components, it produces: HOUSEHOLD_1_001_CA_1, HOUSEHOLD_1_001_WI_1, etc.
    
    # Sticking to the structure of the notebook's example:
    # This function creates an ID like "HOBBIES_1_001_CA_1", not "HOBBIES_1_001_CA_1_evaluation"
    # The "evaluation" part is typically on the full ID in sales_train_evaluation.csv
    product_id_generated = f"{category}_{department}_{item_number}_{state_abbr}_{store_number}"


    # print(f"Generated Product ID: {product_id_generated}") # Use logger
    return product_id_generated


def find_closest_action_index(env, target_quantity: int) -> int: # env: InventoryEnv
    """
    Finds the action index corresponding to the largest possible order quantity
    that is less than or equal to the target quantity.
    From notebook cell 37424dfb (part of run_baseline_policy).

    Args:
        env: The InventoryEnv instance (must have `possible_orders` attribute).
        target_quantity: The desired quantity to order.

    Returns:
        The discrete action index.
    """
    if not hasattr(env, 'possible_orders'):
        raise AttributeError("Environment instance must have a 'possible_orders' attribute.")

    if target_quantity <= 0:
        return 0 # Action index 0 typically corresponds to ordering 0 units

    possible_orders = env.possible_orders # This is a np.array from InventoryEnv
    
    # Find indices where possible order is <= target quantity
    valid_indices = np.where(possible_orders <= target_quantity)[0]

    if len(valid_indices) == 0:
        # This can happen if target_quantity is positive but smaller
        # than the smallest non-zero order step. Order 0 in this case.
        return 0
    else:
        # Return the index corresponding to the largest valid possible order
        best_action_index = valid_indices.max()
        return int(best_action_index)