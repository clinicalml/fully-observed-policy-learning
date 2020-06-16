import logging
from collections import defaultdict

import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from direct.direct_policy_model import DirectPolicyModel
from direct.reward_construction import get_reward_mapping
from utils.utils import save_frontier_to_csv
from utils.metrics import get_metrics, get_metrics_with_deferral


def construct_policy_frontier(exp_name, num_trials, 
                             cohort_df, outcomes_df, 
                             reward_params, training_params, 
                             validate=True, split_fn=None, metadata_df=None,
                             include_defer=False,
                             test_cohort_df=None, test_outcomes_df=None):


    model_dicts, dataset_splits = train_all_models(exp_name, num_trials, 
                                                   cohort_df, outcomes_df, 
                                                   reward_params, training_params,
                                                   validate=validate,
                                                   split_fn=split_fn,
                                                   metadata_df=metadata_df,
                                                   include_defer=include_defer,
                                                   test_cohort_df=test_cohort_df, 
                                                   test_outcomes_df=test_outcomes_df)


    frontiers_dict = evaluate_model(model_dicts, 
                                    dataset_splits,
                                    outcomes_df,
                                    reward_params,
                                    test_outcomes_df=test_outcomes_df,
                                    include_defer=include_defer)

    return frontiers_dict




def train_all_models(exp_name, num_trials,
                     cohort_df, outcomes_df,
                     reward_params, training_params,
                     validate=True, split_fn=None, metadata_df=None,
                     include_defer=False,
                     test_cohort_df=None, test_outcomes_df=None):
    
    model_dicts_list, train_val_splits = [], []

    for trial in range(num_trials):
        logging.info(f"Starting trial {trial}")

        if validate:
            if split_fn is not None:
                 train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = split_fn(
                                                                                        cohort_df, outcomes_df, metadata_df,
                                                                                        seed=24+trial, train_size=0.7
                                                                                    )

            else:
                train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = train_test_split(
                                                                                        cohort_df, outcomes_df, 
                                                                                        seed=24+trial, train_size=0.7
                                                                                    )

            train_val_splits.append({'train': train_cohort_df, 'val': val_cohort_df})

        else:
            train_cohort_df, train_outcomes_df = cohort_df, outcomes_df
            val_cohort_df, val_outcomes_df = test_cohort_df, test_outcomes_df
            train_val_splits.append({'train': train_cohort_df, 'test': val_cohort_df})


        models_dict_for_trial = train_model(exp_name, trial, 
                                            train_cohort_df, val_cohort_df, 
                                            train_outcomes_df, val_outcomes_df, 
                                            reward_params, training_params,
                                            include_defer=include_defer)

        model_dicts_list.append(models_dict_for_trial)

    return model_dicts_list, train_val_splits



def train_model(exp_name, trial,
                train_cohort_df, val_cohort_df,
                train_outcomes_df, val_outcomes_df, 
                reward_params,
                training_params,
                include_defer=False):

    models_dict = {}

    for reward_param in reward_params:

        logging.info(f"Running experiment for setting={reward_param}")

        train_reward_df = get_reward_mapping(train_outcomes_df, reward_params=reward_param,
                                             include_defer=include_defer)
        val_reward_df = get_reward_mapping(val_outcomes_df, reward_params=reward_param,
                                         include_defer=include_defer)

        # Construct model description string
        model_desc = f"trial_{trial}_setting_{list(reward_param.items())}" 

        # Train model
        actions = [col for col in train_reward_df.columns if col != 'example_id']
        action_map = {i: action for i, action in enumerate(actions)}

        model = DirectPolicyModel(num_inputs=train_cohort_df.shape[1]-1,
                                num_outputs=train_outcomes_df.shape[1]-1,
                                action_map=action_map,
                                desc=model_desc, exp_name=exp_name)
        
        model.train_policy(train_cohort_df, val_cohort_df,
                           train_reward_df, val_reward_df,
                           train_outcomes_df, val_outcomes_df,
                           training_params)
        
        models_dict[tuple(reward_param.items())] = model

    return models_dict


def evaluate_model(model_dicts_list, 
                   train_val_splits,
                   outcomes_df,
                   reward_params,
                   test_outcomes_df=None,
                   include_defer=False):

    all_stats = defaultdict(list)
    reward_param_names, metric_names = sorted(reward_params[0].keys()), None

    param_settings_array = np.array([
        [param_setting[param_name] for param_name in reward_param_names]
        for param_setting in reward_params
    ])

    for train_val_split, model_dict in zip(train_val_splits, model_dicts_list):
        
        for cohort_name, cohort_df in train_val_split.items():
            stats_for_param = []

            for param_setting in reward_params:
                logging.info("Storing primary outcomes...")

                # Compute metrics of interest here
                current_model = model_dict[tuple(param_setting.items())]
                cohort_actions_df = current_model.get_actions(cohort_df)

                if cohort_name == 'test': 
                    cohort_actions_outcomes_df = cohort_actions_df.merge(test_outcomes_df, on='example_id') 
                else:
                    cohort_actions_outcomes_df = cohort_actions_df.merge(outcomes_df, on='example_id') 
                
                metrics = get_metrics(cohort_actions_outcomes_df) if not(include_defer) else get_metrics_with_deferral(cohort_actions_outcomes_df)

                if metric_names is None:
                    metric_names = list(metrics.keys())

                stats_for_param.append([metrics[name] for name in metric_names])

            all_stats[cohort_name].append(np.array(stats_for_param))

    columns = reward_param_names + metric_names + [f'{metric}_stdev' for metric in metric_names]
    stats_dict_final = {}

    for cohort_name, stats_for_cohort in all_stats.items():
        stats_means = np.array(stats_for_cohort).mean(axis=0)
        stats_stdevs = np.array(stats_for_cohort).std(axis=0)
        
        stats_final = np.hstack([np.array(param_settings_array),  stats_means, stats_stdevs])
        
        logging.info("Completed calculating means")
        stats_dict_final[cohort_name] =  pd.DataFrame(stats_final, columns=columns)

    return stats_dict_final

