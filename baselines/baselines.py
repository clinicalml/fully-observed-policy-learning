import numpy as np
import pandas as pd

def get_policy_constrained(preds_df, outcomes_df, params):
    
    # Extract necessary parameters for alg
    n, d = len(preds_df), len(outcomes_df) - 1
    outcomes = [outcome for outcome in outcomes_df.columns if outcome != 'example_id']
    
    tol = params['tol'] / n
    relax = params['relax']
    max_iter = params['max_iter']

    # Initialize step size for updating costs
    dcost = params['init_step_size'] * np.ones(d)

    # Initialize costs
    cost = np.zeros(d)
    
    # Frequencies of treatments to match clinicians
    target_freq = outcomes_df['prescription'].value_counts(normalize=True)
    target_freq = np.array([target_freq.get(outcome) for outcome in outcomes])
    current_freq = target_freq.copy()
    
    preds = preds_df[[f'predicted_prob_{outcome}' for outcome in outcomes]].values
    
    for i in range(max_iter):

        # Updates for each cost
        cost += dcost * np.sign(current_freq - target_freq)

        # Ensure costs have zero mean
        cost -= np.mean(cost)
        
        # Reduce step size
        dcost *= relax
        
        # Get updated treatment frequencies
        adjusted_preds = preds + cost
        curr_policy = np.argmin(adjusted_preds, axis=1)
        min_preds = np.min(adjusted_preds, axis=1)
        current_freq = np.bincount(curr_policy, minlength=d) / n
        
        error = np.max(np.abs(current_freq - target_freq)) * n
        
        if i % 10 == 0:
            print(current_freq)
            print(f"Error at iteration {i}: {error}")
        
        if error < tol:
            break
    
    # Get final policy with adjusted costs
    final_policy_df = preds_df[['example_id']].copy()
    final_policy_idx = np.argmin(preds + cost, axis=1)
    
    final_policy_df['policy'] = [outcomes[action_idx] for action_idx in final_policy_idx]
    
    policy_outcomes_df = final_policy_df.merge(outcomes_df, on='example_id')
    stats = get_iat_broad_bootstrapped(policy_outcomes_df, col_name='policy')
    
    return final_policy_df, cost, stats


def get_policy_unconstrained(preds_df, outcomes_df):

    outcomes = [outcome for outcome in outcomes_df.columns if outcome != 'example_id']

    def get_policy_for_row(row):
        row_preds = row[[f'predicted_prob_{abx}' for outcome in outcomes]]
        return row_preds.idxmin()[-3:]

    preds_df['policy'] = preds_df.apply(get_policy_for_row, axis=1)
    policy_df = preds_df[['example_id', 'policy']].copy()
    policy_outcomes_df  = policy_df.merge(outcomes_df, on='example_id')

    stats = get_iat_broad_bootstrapped(policy_outcomes_df, col_name='policy')

    return policy_df, stats


        
    
