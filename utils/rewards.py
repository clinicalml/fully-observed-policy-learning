import pandas as pd

def get_reward_for_example(outcomes, reward_params, include_defer=False):

    '''
        Construct reward mapping for a single example in dataset.

        Inputs:
            - outcomes (Pandas Series): Contains fully observed action outcomes for example of interest.
            - reward_params (dict): Parameter setting to be used for constructing reward values. Keys are parameter 
            names, values are the corresponding values

        Outputs:
            - Pandas Series containing reward for each action available in provided outcomes data
    '''

    # Mapping of action to corresponding reward for specified example
    actions = [action for action in list(outcomes.index) if action != 'example_id' and action != 'prescription']
    
    # Example code for reward construction in antibiotic prescription setting:
    reward_dict = {}
    omega = reward_params['omega']

    for action in actions:
        reward_dict[action] = omega * (1-outcomes[action]) + (1-omega) * int(action in ['NIT', 'SXT'])

    if include_defer:
        actions.append('defer')
        assert 'r_defer' in reward_params

        r_defer = reward_params['r_defer'] 
        reward_dict['defer'] = reward_dict[outcomes['prescription']] + r_defer

    # Return Series object with reward for each action
    return pd.Series([reward_dict[action] for action in actions], index=actions)
