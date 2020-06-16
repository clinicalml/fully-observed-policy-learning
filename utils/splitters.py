import logging
from sklearn.utils import shuffle


def split_pids(cohort_info_df,
                seed, train_size=.7): 

    pids = sorted(cohort_info_df['person_id'].unique())
    shuffled_pids = shuffle(pids, random_state=seed)
    cutoff = int(len(shuffled_pids)*train_size)
    train_pids, val_pids = shuffled_pids[:cutoff], shuffled_pids[cutoff:]
    
    return train_pids, val_pids



def split_cohort_abx(cohort_df, outcomes_df, 
                     cohort_info_df,
                     seed, train_size=.7):

    '''
        Given a DataFrame containing cohort features and a DataFrame containing outcome
        labels, splits the features / labels data into train / validation sets on the basis
        of person ID. This ensures that there are no individuals with data in both the training
        and validation sets.
    '''

    train_pids, val_pids = split_pids(cohort_info_df, seed, train_size)

    train_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(train_pids))]['example_id'].values
    val_eids = cohort_info_df[cohort_info_df['person_id'].isin(set(val_pids))]['example_id'].values

    # Extract features for train / val example IDs
    train_cohort_df = cohort_df[cohort_df['example_id'].isin(set(train_eids))]
    val_cohort_df = cohort_df[cohort_df['example_id'].isin(set(val_eids))]

    logging.info(f"Train cohort size: {len(train_cohort_df)}")
    logging.info(f"Validation cohort size: {len(val_cohort_df)}")

    # Extract outcome labels for train / val cohorts - ensure same example ID order by merging
    train_outcomes_df = train_cohort_df[['example_id']].merge(outcomes_df, on='example_id', how='inner')
    val_outcomes_df =  val_cohort_df[['example_id']].merge(outcomes_df, on='example_id', how='inner')

    assert list(train_cohort_df['example_id'].values) == list(train_outcomes_df['example_id'].values)
    assert list(val_cohort_df['example_id'].values) == list(val_outcomes_df['example_id'].values)

    return train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df
