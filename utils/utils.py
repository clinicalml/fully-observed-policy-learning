import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import VarianceThreshold

def get_param_combinations(reward_params):
    param_names = reward_params.keys()
    param_values = (reward_params[param_name] for param_name in param_names)
    param_combos = [dict(zip(param_names, combination))  for combination in itertools.product(*param_values)]

    return param_combos


def convert_fnr_to_prob(true_outcomes, preds, fnr):
    desired_tpr = 1 - fnr
    fprs, tprs, thresholds = roc_curve(true_outcomes, preds)
    
    diffs = [abs(t - desired_tpr) for t in tprs]
    i = diffs.index(min(diffs))
    return thresholds[i], fprs[i], tprs[i]

 
def save_frontier_to_csv(frontier_df, save_path):
	frontier_df.to_csv(save_path, index=None)
	return True


def apply_variance_threshold(X, selector=None):
    if selector is None:
        selector = VarianceThreshold()
        selector.fit(X)
    
    X = selector.transform(X)
    return X, selector


def get_base_model(model_class):
    if model_class =='lr':
        clf = LogisticRegression()

    elif model_class == 'dt':
        clf = DecisionTreeClassifier()

    elif model_class == 'rf':
        clf = RandomForestClassifier()

    elif model_class == 'xgb':
        clf = XGBClassifier()
        
    return clf
