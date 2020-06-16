import argparse
import logging
from datetime import datetime 

import os
import sys
import json

sys.path.append('../')

import numpy as np
import pandas as pd

from indirect.train_outcome_models import train_outcome_models_main
from direct import run_direct_policy_training
from indirect import expected_reward_maximization, thresholding
from utils.utils import get_param_combinations
from utils.splitters import split_cohort_abx 

parser = argparse.ArgumentParser(description='process parameters for experiment')

parser.add_argument('--exp_name',
                     type=str, required=True,
                     help='Name of experiment')

parser.add_argument('--mode', 
                    type=str, choices=['direct', 'thresholding', 'exp_reward_max'],
                    help='Policy learning mode')

parser.add_argument('--num_trials',
                     type=int, default=20,
                     help='Number of trials to run experiment')

parser.add_argument('--features_path',
                    type=str, required=True,
                    help='Filepath for cohort features')

parser.add_argument('--outcomes_path',
                    type=str, required=True,
                    help='Filepath for outcome data')

parser.add_argument('--metadata_path',
                    type=str, required=False,
                    help='Filepath for cohort metadata (e.g., information to be used for creating train/validation splits)')

parser.add_argument('--validate', 
                    action='store_true',
                    help='Whether to perform train / val splits')

### Parameters to be used if evaluating on test set ####

parser.add_argument('--test_features_path',
                    type=str,
                    help='Filepath for test cohort features')

parser.add_argument('--test_outcomes_path',
                    type=str,
                    help='Filepath for test outcome data')


#### Parameters for learning conditional outcome models ####

parser.add_argument('--best_models_path',
                    type=str, 
                    help='Path to JSON containing mapping of optimal model class for each outcome')

parser.add_argument('--best_hyperparams_path',
                    type=str, 
                    help='Path to JSON containing optimal hyperparameter settings for predictive models for each outcome')


#### Parameters for thresholding approach ####

parser.add_argument('--predictions_path',
                    type=str, 
                    help='Path to prdictions generated from trained outcome prediction models')

parser.add_argument('--threshold_combos_path',
                    type=str, 
                    help='Path to JSON containing list of optimal threshold combinations selected from validation data')

parser.add_argument('--best_thresholds_path',
                    type=str, 
                    help='Path to CSV containing list of optimal threshold combinations selected from validation data')

parser.add_argument('--threshold_selection_config_path',
                    type=str, 
                    help='Path to JSON file containing parameters to be used for model training (in thresholding mode)')


#### Parameters for direct learning approach ####

parser.add_argument('--model_params_config_path',
                    type=str, 
                    help='Path to JSON file containing parameters to be used for model training (in direct mode)')

parser.add_argument('--reward_params_path',
                    type=str,
                    help='Path to JSON containing range of parameters to be used in reward function (in direct mode)')

parser.add_argument('--defer',
                    action='store_true',
                    help='Flag to include deferral as action in direct policy')





if __name__ == '__main__':
    args = parser.parse_args()

    # Setting up directories for storing logs and trained models
    log_time = datetime.now().strftime("%d-%m-%Y_%H%M%S")

    log_folder_path = f"experiment_results/{args.exp_name}/experiment_logs" 
    model_folder_path = f"experiment_results/{args.exp_name}/models"
    results_folder_path = f"experiment_results/{args.exp_name}/results"

    if not os.path.exists(log_folder_path): 
        os.makedirs(log_folder_path)

    if not os.path.exists(model_folder_path): 
        os.makedirs(model_folder_path)

    if not os.path.exists(results_folder_path): 
        os.makedirs(results_folder_path)

    logging.basicConfig(filename=f"experiment_results/{args.exp_name}/experiment_logs/experiment_{log_time}.log",
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(args)

    # Read in features and outcomes used for policy training

    logging.info("Reading in data...")

    train_features_df = pd.read_csv(args.features_path)
    train_outcomes_df = pd.read_csv(args.outcomes_path)

    logging.info(f"Train cohort size: {train_features_df.shape[0]}")
    cohort_metadata = pd.read_csv(args.metadata_path) if args.metadata_path is not None else None
   
    # If in test mode, load in test features / labels, along with optimal hyperparamters / models selected from validation  
    test_features_df = pd.read_csv(args.test_features_path) if not args.validate else None
    test_outcomes_df = pd.read_csv(args.test_outcomes_path) if not args.validate else None
    best_model_classes_dict, best_hyperparams_dict = None, None
    
    if not (args.validate or args.mode == 'direct') : 
        with open(args.best_models_path) as f:
            best_model_classes_dict = json.load(f)
        with open(args.best_hyperparams_path) as f:
            best_hyperparams_dict = json.load(f)
    
    # Train outcome models if indirect learning method specified

    if args.predictions_path is None:

        if args.mode == 'thresholding' or args.mode == 'exp_reward_max':
            preds_df = train_outcome_models_main(train_features_df, train_outcomes_df,
                                                 results_path=results_folder_path,
                                                 validate=args.validate,
                                                 test_cohort_df=test_features_df,
                                                 test_outcomes_df=test_outcomes_df,
                                                 best_hyperparams_dict=best_hyperparams_dict, 
                                                 best_model_classes_dict=best_model_classes_dict,
                                                 cohort_metadata=cohort_metadata)

    else:
        preds_df = pd.read_csv(args.predictions_path)

    # Load in parameters for learning policies across multiple reward settings 
    # Used with indirect (expected reward maximization) or direct learning methods
    if args.reward_params_path:
        with open(args.reward_params_path) as f:
            reward_params = json.load(f)

        reward_params_list = get_param_combinations(reward_params)

    # Construct policy frontiers using specified method
    if args.mode == 'thresholding':

        best_settings_df = pd.read_csv(args.best_thresholds_path) if args.best_thresholds_path is not None else None

        threshold_space = None
        if args.threshold_combos_path is not None:
            with open(args.threshold_combos_path) as f:
                threshold_space = json.load(f)

        threshold_selection_config = None
        if args.threshold_selection_config_path is not None:
            with open(args.threshold_selection_config_path) as f:
                threshold_selection_config = json.load(f)

        frontiers_dict = thresholding.construct_policy_frontier(preds_df, train_outcomes_df, 
                                                  validate=args.validate,
                                                  thresholds=threshold_space,
                                                  threshold_selection_config=threshold_selection_config,
                                                  best_settings_df=best_settings_df,
                                                  test_outcomes_df=test_outcomes_df)



    elif args.mode == 'exp_reward_max':
        num_trials = 20 if args.validate else 1

        frontiers_dict = expected_reward_maximization.construct_policy_frontier(
                                preds_df, train_outcomes_df, reward_params_list,
                                validate=args.validate,
                                test_outcomes_df=test_outcomes_df,
                                num_trials=num_trials
                            )

    elif args.mode == 'direct':

        # Load in parameters used for training direct model (e.g., learning rate, optimizer)
        training_params = {}

        if args.model_params_config_path:
            with open(args.model_params_config_path) as f:
                training_params = json.load(f)
        
        frontiers_dict = run_direct_policy_training.construct_policy_frontier(
                args.exp_name, args.num_trials,
                train_features_df, train_outcomes_df,
                reward_params_list, training_params,
                validate=args.validate,
                split_fn=split_cohort_abx, 
                include_defer=args.defer,
                metadata_df=cohort_metadata,
                test_cohort_df=test_features_df, 
                test_outcomes_df=test_outcomes_df
            )

    else:
        raise ValueError("Training mode not recognized.")



    for cohort, frontier in frontiers_dict.items():
        frontier.to_csv(os.path.join(results_folder_path, f'frontier_{cohort}.csv'), index=None)




