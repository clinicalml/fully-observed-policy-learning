# Treatment Policy Learning with Fully Observed Outcomes

## Overview

In several medical settings, lab testing provides retrospective access to patient outcomes under a variety of different treatments, but this information is typically unavailable to clinicians when making a treatment decision. We can leverage this data to learn more effective treatment policies for use on future patients.

This library provides implementations of three approaches for learning treatment policies in the setting with fully observed outcomes and (potentially) multiple objectives:

- **Thresholding**
- **Expected Reward Maximization**
- **Direct Learning**

For more information about these approaches and their various trade-offs, check out our paper: [Treatment Policy Learning in Multiobjective Settings with Fully Observed Outcomes](https://arxiv.org/abs/2006.00927) (KDD 2020).

## Requirements

This library is built on Python 3 with pandas, sklearn, PyTorch, and numpy. If using XGBoost for training conditional outcome models in the indirect approaches, installation of the xgboost package is also required.

**TODO**: add setup script for automatically creating conda env with all necessary packages for running this code

## Usage

This library is intended to be agnostic to any particular setting, and thus requires user-specified definitions of a few functions before it can be applied to a new setting of interest. Before running an experiment, the user should define:

- **Policy Metrics** : In `utils/metrics.py`, the users should provide their desired implementation of the `get_metrics` function, which computes quantities evaluating the quality of a provided policy. <br/><br/> This function takes as input a Pandas DataFrame containing treatment outcomes for the dataset of interest, along with the chosen action for each example. It should return a dictionary mapping metric names to the computed values for the provided policy.

- **Reward Function**: This is only necessary if using the expected reward maximization or direct learning approaches, which both rely on a notion of a reward function for learning optimal policies. <br/><br/>  In `utils/rewards.py`, users should implement their desired version of the `get_reward_for_example` function. This function takes as input (1) the treatment outcomes for a given example, and (2) a dictionary containing settings of the parameters required to compute the reward. It returns a Pandas Series containing the computed rewards for each outcome.

To run an experiment, run the `policy_learning.py` script with the appropriate settings of the parameters (see descriptions in the Parameters section below). 

Example scripts can be found in `sample_validation_exp_script.sh` and `sample_test_exp_script.sh`.


## Parameters

### Core Parameters (used across all approaches)

Parameter | Description | Type | Required 
----------------|-------------|----------|----------
`--exp_name` | User-specified name for current experiment. | str | Yes
`--mode` | Policy learning method to use. Can choose from [`direct`, `thresholding`, `exp_reward_max`] | str | Yes  
`--features_path` | Path to CSV containing features to be used as model inputs for learning policies. | str | Yes 
`--outcomes_path` | Path to CSV containing treatment outcomes for dataset of interest.  | str | Yes 
`--validate` | Flag indicating if running evaluation on a validation set. If not provided, the library expects paths to data containing features / outcomes for a separate test set | | 
`--test_features_path` | Same as `features_path`, but contains data for test set. Required if `validate` flag is not provided | str | No 
`--test_outcomes_path` | Same as `outcomes_path`, but contains data for test set.  Required if `validate` flag is not provided  | str | Yes 
`--metadata_path` | Path to CSV containing any dataset metadata needed for running experiment (e.g., auxiliary information needed for generating appropriate train/validation splits)  | str | No 

### Parameters for Indirect Approaches (Thresholding / Expected Reward Maximization)

Parameter | Description | Type | Required 
----------------|-------------|----------|----------
`--best_models_path` | Path to JSON containing optimal model classes chosen during validation; map from outcome to name of optimal model class. Required if using an indirect approach in test mode. | str  | Yes (test)
`--best_hyperparams_path` | Path to JSON containing optimal hyperparameters for each outcome tuned during validation; map from outcome to hyperparameter setting. Required if using an indirect approach in test mode. | str | Yes (test)

### Parameters for Thresholding

Parameter | Description | Type | Required | Default
----------------|-------------|----------|----------|----------
`--threshold_combos_path` | Path to JSON defining threshold search space used in thresholding approach. Map from each outcome to possible thresholds for that outcome.  Required if using thresholding mode with `validate` flag set. See `sample_threshold_space.json` for example of such a file.| str | Yes | 
`--threshold_selection_config_path` | Path to JSON specifying configuration for selecting optimal thresholds. Required if using thresholding mode with `validate` flag set.  See `sample_threshold_selection_config.json` for an example.| str | Yes | 
`--best_thresholds_path` | Path to . Required if using thresholding mode for evaluation on test set (i.e., without the `validate` flag).| str | Yes | 


### Parameters for Direct Learning
Parameter | Description | Type | Required | Default
----------------|-------------|----------|----------|----------
`--reward_params_path` | Path to JSON defining reward parameter space. Map from each parameter in reward function to values at which we want to learn a policy. See `sample_reward_space.json` for an example. | str | Yes | 
`--model_params_config_path` | Path to JSON containing parameters for training the direct model (e.g., learning rate, optimizer, etc.). Default values of parameters can be seen in `direct/direct_policy_mode.py`| str | No | 


