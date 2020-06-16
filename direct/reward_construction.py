import pandas as pd
import numpy as np

import sys
sys.path.append("../")

from utils.rewards import get_reward_for_example

def get_reward_mapping(outcomes_df, reward_params, include_defer=False):
    '''
        Construct DataFrame containing rewards for provided dataset at the specified reward parameter values.

        Inputs:
            - outcomes_df (Pandas DataFrame): Fully observed action outcomes for dataset of interest.
                Each column should contain the outcomes for a single action.
            - reward_params (dict): Parameter setting to be used for constructing reward values. Keys are parameter 
            names, values are the corresponding values

        Outputs:
            - Pandas DataFrame containing reward for each action for all examples in provided data
    '''

    reward_df = outcomes_df.apply(lambda x: get_reward_for_example(x, reward_params, 
                                                                include_defer=include_defer), axis=1)
    reward_df.insert(loc=0, column='example_id', value=outcomes_df['example_id'].copy())
    return reward_df


