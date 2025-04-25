import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


"""
PPO RL using Centralized Training with Decentralized Execution
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)



class ValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    




class UXSimPPO:
    def __init__(self, env, learning_rate, gamma, gae_lambda, epochs, mini_batch_size, value_network_loss_coef, entropy_coef, clip_epsilon, number_steps):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.value_network_loss_coef = value_network_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.number_steps = number_steps
        self.observation_space = 4
        self.action_space = 2

        self.max_obs = 0
        for node in self.env.W.NODES_NAME_DICT:
            length = int(len(self.env.W.NODES_NAME_DICT[node].inlinks))
            if length > self.max_obs:
                self.max_obs = length

        self.PAD_VALUE = -1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_network = PolicyNetwork(self.max_obs, self.action_space).to(self.device)
        self.value_network = ValueNetwork(self.max_obs).to(self.device)


        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.total_reward = 0
        self.rewards, self.log_probs, self.actions, self.states, self.values = [], [], [], [], []

        self.policy_network.train()
        self.value_network.train()




    def predict(self, observations):
            i = 0
            predictions = []
            with torch.no_grad():
                for obs in observations:
                    
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    if len(obs) < self.max_obs:
                        obs_tensor = F.pad(obs_tensor, (0, self.max_obs - len(obs)), value=self.PAD_VALUE)

                    state = torch.as_tensor(obs_tensor, dtype=torch.float32, device=self.device)
  
                    action_probs = self.policy_network(state).squeeze(0)
                    dist = torch.distributions.Categorical(action_probs + 1e-8)

                    action = dist.sample()
                    predictions.append(action.cpu())
                    self.actions.append(action.cpu())
                    self.states.append(state)

                    log_prob = dist.log_prob(action)
                    self.log_probs.append(log_prob.detach())
                    value = self.value_network(state)
                    self.values.append(value.detach())
                    i += 1

            return predictions
    
    


    def update(self, rewards):
        for r in rewards:
            self.rewards.append(r)




    def compute_gae(self):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        print(values.shape)

        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        advantages = torch.zeros(T, device=self.device)
        values = torch.cat([values, values[-1:]])

        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
    


    
    def train(self, t):
        
        if (t+1) % self.number_steps == 0:

            returns, advantages = self.compute_gae()

            dataset = TensorDataset(
                torch.stack(self.states),
                torch.stack(self.actions),
                torch.stack(self.log_probs),
                torch.stack(self.values),
                returns,
                advantages
            )

            data_loader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

            for i in range(self.epochs):

                for batch in data_loader:
                    state_batch, action_batch, old_log_prob_batch, value_network_batch, return_batch, advantage_batch = batch
                    state_batch = state_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    action_batch = action_batch.long()
                    old_log_prob_batch = old_log_prob_batch.to(self.device).detach()
                    return_batch = return_batch.to(self.device).detach()
                    advantage_batch = advantage_batch.to(self.device).detach()

                    action_probs = self.policy_network(state_batch)
                    action_prob_batch = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    new_log_prob_batch = torch.log(action_prob_batch + 1e-8)
                    dist = torch.distributions.Categorical(probs=action_probs)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
                    surr1 = ratio * advantage_batch
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                    policy_network_loss = -torch.min(surr1, surr2).mean()

                    #value_network_batch = self.value_network(state_batch).view(-1)
                    value_network_loss = F.mse_loss(value_network_batch.flatten(), return_batch.flatten()) * self.value_network_loss_coef

                    entropy_loss = -entropy * self.entropy_coef

                    policy_loss = policy_network_loss + entropy_loss
                    value_loss = value_network_loss
                    total_loss = policy_loss + value_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            self.scheduler.step()
            self.rewards, self.log_probs, self.actions, self.states, self.values = [], [], [], [], []
