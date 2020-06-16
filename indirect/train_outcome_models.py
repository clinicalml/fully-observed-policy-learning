import pandas as pd
import numpy as np

from collections import defaultdict
import itertools
from datetime import datetime

import argparse
import logging
import os
import sys
import json
import math

sys.path.append('../')
from indirect.hyperparameter_grid import HyperparameterGrid
from utils.utils import get_base_model, apply_variance_threshold
from utils.splitters import split_cohort_abx

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score 


def train_predictive_model_for_model_class(cohort_df, outcomes_df, outcome_col,
                                          validate=True,
                                          test_cohort_df=None, test_outcomes_df=None,
                                          model_class='lr', 
                                          num_trials=20,
                                          param_setting=None,
                                          early_stopping_rounds=None,
                                          split_fn=None, cohort_metadata=None):

                    
    if param_setting is None:
        grid = HyperparameterGrid()
        param_grid = grid.param_grids[model_class]
        parameters = list(ParameterGrid(param_grid))
    
    else:
        parameters = [param_setting]
    
    best_combo, best_val_auc = None, 0
    
    for i, param in enumerate(parameters):

        logging.info(f"Training model for combination {i+1} / {len(parameters)} combinations")

        val_aucs, val_rounds = [], []
        
        for trial in (range(num_trials)):
            clf = get_base_model(model_class)
            clf.set_params(**param)

            if validate:
                if split_fn is not None:
                     train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = split_fn(
                                                                                            cohort_df, outcomes_df, cohort_metadata,
                                                                                            seed=24+trial, train_size=0.7
                                                                                        )

                else:

                    train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = train_test_split(
                                                                                            cohort_df, outcomes_df, 
                                                                                            seed=24+trial, train_size=0.7
                                                                                        )

            else:
                train_cohort_df, train_outcomes_df = cohort_df, outcomes_df
                val_cohort_df, val_outcomes_df = test_cohort_df, test_outcomes_df

            train_x, selector = apply_variance_threshold(train_cohort_df.drop(columns=['example_id']).values)
            val_x, _ = apply_variance_threshold(val_cohort_df.drop(columns=['example_id']).values, selector=selector)

            clf.fit(train_x, train_outcomes_df[outcome_col].values)
            
            val_preds = clf.predict_proba(val_x)[:, 1]
            val_auc = roc_auc_score(val_outcomes_df[outcome_col], val_preds)
            
            val_aucs.append(val_auc)

            if model_class == 'xgb':
                clf.fit(train_x, train_outcomes_df[outcome_col].values,
                        early_stopping_rounds=early_stopping_rounds,
                        eval_metric='auc', verbose=False,
                        eval_set=[(val_x, val_outcomes_df[outcome_col].values)])

                if early_stopping_rounds is not None:
                    val_aucs.append(clf.best_score)
                    val_rounds.append(clf.best_iteration)
                else:
                    val_preds = clf.predict_proba(val_x)[:, 1]
                    val_aucs.append(roc_auc_score(val_outcomes_df[outcome_col].values,
                                                  val_preds))
                    val_rounds.append(clf.n_estimators)
            
            else:
                clf.fit(train_x, train_outcomes_df[outcome_col].values)
            
                val_preds = clf.predict_proba(val_x)[:, 1]
                val_auc = roc_auc_score(val_outcomes_df[outcome_col],
                                        val_preds)
                val_aucs.append(val_auc)
        

        if np.mean(val_aucs) > best_val_auc:
            best_combo, best_val_auc = param, np.mean(val_aucs)
            if model_class == 'xgb':
                best_combo['n_estimators']  = math.ceil(np.mean(val_rounds)/5)*5
          
    return best_combo, best_val_auc


def get_best_params_by_model_class(cohort_df, outcomes_df,
                                   outcome_col, 
                                   model_classes=['lr', 'dt', 'rf'],
                                   num_trials=5,
                                   split_fn=None, cohort_metadata=None):

    hyperparams_by_model, val_aucs_by_model = {}, {}

    for model_class in model_classes:

        logging.info(f'Training models for class {model_class}')

        early_stopping_rounds = 10 if model_class == 'xgb' else None
        
        best_hyperparams, best_val_auc = train_predictive_model_for_model_class(cohort_df, outcomes_df,
                                                                               outcome_col,
                                                                               num_trials=num_trials,
                                                                               model_class=model_class,
                                                                               early_stopping_rounds=10,
                                                                               split_fn=split_fn, 
                                                                               cohort_metadata=cohort_metadata)

        hyperparams_by_model[model_class] = best_hyperparams
        val_aucs_by_model[model_class] = best_val_auc

    return hyperparams_by_model, val_aucs_by_model


def train_models_for_best_params(cohort_df, outcomes_df,
                               outcome_col,
                               best_hyperparams,
                               model_classes=['lr', 'dt', 'rf'],
                               num_trials=20,
                               split_fn=None, cohort_metadata=None):


    val_aucs_by_model = {}

    for model_class in model_classes:

        logging.info(f'Training models for class {model_class}')

        _, best_val_auc = train_predictive_model_for_model_class(cohort_df, outcomes_df,
                                                               outcome_col,
                                                               num_trials=num_trials,
                                                               model_class=model_class,
                                                               param_setting=best_hyperparams[model_class],
                                                               split_fn=split_fn, 
                                                               cohort_metadata=cohort_metadata)

        val_aucs_by_model[model_class] = best_val_auc

    return val_aucs_by_model



def construct_train_val_preds_df(cohort_df, outcomes_df,
                                 best_hyperparams_dict,
                                 best_models_by_outcome,
                                 validate=True,
                                 test_cohort_df=None, test_outcomes_df=None,
                                 num_splits=20, split_fn=None, cohort_metadata=None):

    
    all_train_val_pred_dfs = []
    
    for split in range(num_splits):

        if validate:
            if split_fn is not None:
                train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = split_fn(
                                                                                        cohort_df, outcomes_df, cohort_metadata,
                                                                                        seed=24+split, train_size=0.7
                                                                                    )

            else:
                train_cohort_df, train_outcomes_df, val_cohort_df, val_outcomes_df = train_test_split(
                                                                                        cohort_df, outcomes_df, 
                                                                                        seed=24+split, train_size=0.7
                                                                                        )
        else:
            train_cohort_df, train_outcomes_df = cohort_df, outcomes_df
            val_cohort_df, val_outcomes_df = test_cohort_df, test_outcomes_df

        train_pred_df = train_cohort_df[['example_id']].copy()
        val_pred_df = val_cohort_df[['example_id']].copy()

        outcomes = outcomes_df.drop(columns=['example_id']).columns
        for outcome_col in outcomes:
            
            train_x, selector = apply_variance_threshold(train_cohort_df.drop(columns=['example_id']).values)
            val_x, _ = apply_variance_threshold(val_cohort_df.drop(columns=['example_id']).values, selector=selector)
            
            best_model_class = best_models_by_outcome[outcome_col]
            clf = get_base_model(best_model_class)
            clf.set_params(**best_hyperparams_dict[outcome_col][best_model_class])
            clf.fit(train_x, train_outcomes_df[outcome_col].values)
            
            train_preds = clf.predict_proba(train_x)[:, 1]
            val_preds = clf.predict_proba(val_x)[:, 1]

            train_pred_df[f'predicted_prob_{outcome_col}'] = train_preds
            val_pred_df[f'predicted_prob_{outcome_col}'] = val_preds
            
        train_pred_df['is_train'] = 1
        val_pred_df['is_train'] = 0
        
        train_val_preds_df = pd.concat([
                train_pred_df, val_pred_df
            ], axis=0)

        train_val_preds_df['split_ct'] = split
        all_train_val_pred_dfs.append(train_val_preds_df)
        
    return pd.concat(all_train_val_pred_dfs, axis=0)
    

def get_best_models(aucs_by_outcome):
    best_model_by_outcome = {}

    for outcome, aucs_for_outcome in aucs_by_outcome.items():
        best_auc, best_model_class = 0, None

        for model_class, auc in aucs_for_outcome.items():
            if auc > best_auc:
                best_auc, best_model_class = auc, model_class
        best_model_by_outcome[outcome] = best_model_class

    return best_model_by_outcome




def train_outcome_models_main(cohort_df, outcomes_df, results_path,
                              validate=True,
                              test_cohort_df=None,
                              test_outcomes_df=None,
                              best_hyperparams_dict=None, 
                              best_model_classes_dict=None,
                              cohort_metadata=None):

    outcomes = outcomes_df.drop(columns=['example_id']).columns

    if validate:

        hyperparams_by_outcome, val_aucs_by_outcome = {}, {}

        for outcome_col in outcomes:

            logging.info(f'Training models for {outcome_col}')

            best_params_for_outcome, val_aucs_for_outcome = get_best_params_by_model_class(cohort_df, outcomes_df, 
                                                                                 outcome_col, num_trials=5,
                                                                                 split_fn=split_cohort_abx,
                                                                                 cohort_metadata=cohort_metadata)

            logging.info(f'Evaluating tuned models for {outcome_col}')
            val_aucs_for_outcome_best_params = train_models_for_best_params(cohort_df, outcomes_df,
                                                                 outcome_col=outcome_col, num_trials=20,
                                                                 best_hyperparams=best_params_for_outcome,
                                                                 split_fn=split_cohort_abx,
                                                                 cohort_metadata=cohort_metadata)


            hyperparams_by_outcome[outcome_col] = best_params_for_outcome
            val_aucs_by_outcome[outcome_col] = val_aucs_for_outcome_best_params

        with open(os.path.join(results_path, 'hyperparameters.json'), 'w') as fp:
            json.dump(hyperparams_by_outcome, fp)

        with open(os.path.join(results_path, 'val_aucs.json'), 'w') as fp:
            json.dump(val_aucs_by_outcome, fp)
        
        logging.info("Finding best model based on validation AUC")
        best_models_by_outcome = get_best_models(val_aucs_by_outcome)

        with open(os.path.join(results_path, 'best_models.json'), 'w') as fp:
            json.dump(best_models_by_outcome, fp)

        logging.info("Construction of train / validation predictions to be saved")
        preds_df = construct_train_val_preds_df(cohort_df, outcomes_df,
                                               hyperparams_by_outcome,
                                               best_models_by_outcome,
                                               split_fn=split_cohort_abx, 
                                               cohort_metadata=cohort_metadata)

        preds_df.to_csv(os.path.join(results_path, 'val_predictions.csv'), index=None)

    else:
        test_aucs_by_outcome = {}

        for outcome_col in outcomes:

            logging.info(f'Training models for {outcome_col}')
            model_class = best_model_classes_dict[outcome_col]
            param_setting = best_hyperparams_dict[outcome_col][model_class]

            _, test_auc_for_outcome = train_predictive_model_for_model_class(cohort_df, outcomes_df, outcome_col,
                                                                          validate=False,
                                                                          test_cohort_df=test_cohort_df, 
                                                                          test_outcomes_df=test_outcomes_df,
                                                                          model_class=model_class,
                                                                          param_setting=param_setting,
                                                                          num_trials=1)


            test_aucs_by_outcome[outcome_col] = test_auc_for_outcome


        with open(os.path.join(results_path, 'test_aucs.json'), 'w') as fp:
            json.dump(test_aucs_by_outcome, fp)

        preds_df = construct_train_val_preds_df(cohort_df, outcomes_df,
                                               best_hyperparams_dict=best_hyperparams_dict,
                                               best_models_by_outcome=best_model_classes_dict,
                                               validate=False,
                                               test_cohort_df=test_cohort_df,
                                               test_outcomes_df=test_outcomes_df, num_splits=1)

        preds_df.to_csv(os.path.join(results_path, 'test_predictions.csv'), index=None)

    return preds_df

