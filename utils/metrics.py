import pandas as pd

def get_metrics(policy_outcomes_df, policy_colname='action', include_defer=False):

    iat = policy_outcomes_df.apply(
        lambda x: x[f'{x[policy_colname]}']==1, axis=1).mean()

    broad = policy_outcomes_df.apply(
        lambda x: x[policy_colname] in ['CIP', 'LVX'], axis=1).mean() 
    
    return {
        'iat': iat, 'broad': broad
    }

def get_metrics_with_deferral(policy_outcomes_df, policy_colname='action'):

    assert 'prescription' in policy_outcomes_df.columns
    policy_outcomes_df['action_final'] =  policy_outcomes_df.apply(
            lambda x: x[policy_colname] if x[policy_colname] != 'defer' else x['prescription'],
            axis=1
        )
    
    iat = policy_outcomes_df.apply(
        lambda x: x[f"{x['action_final']}"]==1, axis=1).mean()

    broad = policy_outcomes_df.apply(
        lambda x: x['action_final'] in ['CIP', 'LVX'], axis=1).mean() 

    decision_cohort = policy_outcomes_df[policy_outcomes_df[policy_colname] != 'defer']

    iat_alg = decision_cohort.apply(
        lambda x: x[f"{x['action_final']}"]==1, axis=1).mean()

    broad_alg = decision_cohort.apply(
        lambda x: x['action_final'] in ['CIP', 'LVX'], axis=1).mean() 

    iat_doc = decision_cohort.apply(
        lambda x: x[f"{x['prescription']}"]==1, axis=1).mean()

    broad_doc = decision_cohort.apply(
        lambda x: x['prescription'] in ['CIP', 'LVX'], axis=1).mean() 
    
    return {
        'iat': iat, 'broad': broad,
        'iat_alg': iat_alg, 'broad_alg': broad_alg,
        'iat_doc': iat_doc, 'broad_doc': broad_doc,
        'defer_rate': 1 - len(decision_cohort)/len(policy_outcomes_df)
    }
