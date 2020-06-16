import sys
sys.path.append('../')

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim

import numpy as np
import pandas as pd

from direct.policy_losses import policy_loss
from direct.policy_model_architectures import PolicyNetLinear

class DirectPolicyModel:
    def __init__(self, num_inputs, num_outputs, 
                 action_map,
                 exp_name=None, desc=None):

        # Mapping of action numbers to human-readable names for actions
        self.action_map = action_map

        # Underlying PyTorch model representing the learned policy
        self.model = PolicyNetLinear(num_inputs, num_outputs)

        # Name of experiment in which model was created
        self.exp_name = exp_name

        # Description string for this model
        self.desc = desc

        self.default_training_params = {
            'num_epochs': 50,
            'optimizer': 'adam',
            'lr': 1e-4,
            'reg_type': 'l2',
            'lambda_reg': 0
        }


    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        logging.info("Model loaded for evaluation.")


    def get_weights(self):
        return list(model.model.parameters())[0].detach().numpy().T


    def get_actions(self, cohort):
        '''
            Given dataset features, returns (human-readable) actions chosen by the current 
            learned policy model for provided examples.
            
            Input: 
                - cohort (Pandas DataFrame containing example_id's and features to be input to policy model).
                  Must contain `example_id' column, used to serve as unique identifier of each example.

            Output: 
                - Pandas DataFrame with two columns: example_id, action
                  Each row contains an example ID and the corresponding action chosen by the current
                  learned policy model.
        '''

        cohort_tensor = torch.from_numpy(cohort.drop(columns=['example_id']).values)
        action_probs = self.model(cohort_tensor.float())

        chosen_actions = np.argmax(action_probs.detach().numpy(), axis=1)
        actions_df = pd.DataFrame({'example_id': [eid for eid in cohort['example_id'].values],
                                   'action': [self.action_map[action] for action in chosen_actions]})

        return actions_df


    def get_action_distribution(self, cohort):
        '''
            Returns unnormalized distribution over actions for the provided dataset
            under the currently learned policy model.
        '''

        actions_df = self.get_actions(cohort)
        return actions_df["action"].value_counts()


    def get_metrics(self, cohort, outcomes):
        return None


    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                  nesterov=True, momentum=.9)
        return optimizer


    def get_regularization_loss(self, reg_type, lambda_reg):
        regularization_loss = 0

        # L1 regularization 
        if reg_type == 'l1':
            for name, param in self.model.named_parameters():
                if 'bias' not in name:
                    regularization_loss += torch.sum(torch.abs(param))

         # L2 regularization 
        if reg_type == 'l2':
            for name, param in self.model.named_parameters():
                if 'bias' not in name:
                    regularization_loss += torch.sum(torch.pow(param, 2))

        return lambda_reg*regularization_loss

    def get_data_tensors(self, cohort_df, rewards_df):
        features_tensor = torch.from_numpy(cohort_df.drop(columns=['example_id']).values)
        rewards_tensor = torch.from_numpy(rewards_df.drop(columns=['example_id']).values)

        return features_tensor, rewards_tensor


    def train_policy(self, train_cohort, val_cohort,
                     train_rewards_df, val_rewards_df,
                     train_outcomes_df, val_outcomes_df,
                     training_params={},
                     print_interval=5):

        '''
            Train direct policy model.

            Inputs:
                - 

        '''

        curr_training_params = {}

        for param_name in self.default_training_params.keys():
            curr_training_params[param_name] = training_params.get(param_name,
                                                                 self.default_training_params[param_name])

        logging.info(curr_training_params)

        # Construct training and validation datasets
        train_features_tensor, train_rewards_tensor = self.get_data_tensors(train_cohort, train_rewards_df)
        val_features_tensor, val_rewards_tensor = self.get_data_tensors(val_cohort, val_rewards_df)

        train_dataset = data_utils.TensorDataset(train_features_tensor.float(),
                                                 train_rewards_tensor.float())
        train_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Training loop
        optimizer = self.get_optimizer(optimizer_name=curr_training_params['optimizer'],
                                     lr=curr_training_params['lr'])

        num_epochs = curr_training_params['num_epochs']
        for epoch in range(num_epochs):
            for feats, rewards in train_loader:
                optimizer.zero_grad()
                output = self.model(feats)
                
                # Loss from policy distribution
                loss = policy_loss(output, rewards)
                loss += self.get_regularization_loss(reg_type=curr_training_params['reg_type'],
                                                    lambda_reg=curr_training_params['lambda_reg'])

                loss.backward()
                optimizer.step()

            # Compute reward metrics every pre-specified number of epochs
            
            if (epoch + 1) % print_interval == 0:

                logging.info(f'Finished with epoch {epoch + 1}')

                # Expected reward under currently learned policy model
                train_action_probs = self.model(train_features_tensor.float())
                val_action_probs = self.model(val_features_tensor.float())
                
                mean_train_reward = torch.mean(torch.sum(train_action_probs * train_rewards_tensor, axis=1)).item() 
                logging.info(f'Mean (expected) train reward: {mean_train_reward}')

                mean_val_reward = torch.mean(torch.sum(val_action_probs * val_rewards_tensor, axis=1)).item() 
                logging.info(f'Mean (expected) val reward: {mean_val_reward}')

                # Actual (realized) reward (i.e., after argmax) under current policy model
                mean_train_realized_reward = self.get_mean_realized_reward(train_cohort, train_rewards_df)
                logging.info(f'Mean (realized) train reward: {mean_train_realized_reward}')

                mean_val_realized_reward = self.get_mean_realized_reward(val_cohort, val_rewards_df)
                logging.info(f'Mean (realized) val reward: {mean_val_realized_reward}')

                # Compute additional metrics specified in get_metrics() function
                train_metrics = self.get_metrics(train_cohort, train_outcomes_df)
                val_metrics = self.get_metrics(val_cohort, val_outcomes_df)



        torch.save(self.model.state_dict(), 
                   f"experiment_results/{self.exp_name}/models/{self.desc}_final.pth")


    def get_mean_reward(self, cohort_df, rewards_df):
        pass


    def get_mean_realized_reward(self, cohort_df, rewards_df):
        cohort_actions = self.get_actions(cohort_df) 
        rewards_merged = rewards_df.merge(cohort_actions, on='example_id', how='inner')
        
        return rewards_merged.apply(lambda x: x[x['action']], axis=1).mean()
                
