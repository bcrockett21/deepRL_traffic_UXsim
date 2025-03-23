import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset









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
        self.fc3 = nn.Linear(256, 128)
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_network = PolicyNetwork(self.max_obs, self.action_space).to(self.device)
        #self.policy_network.apply(self.init_weights)
        self.value_network = ValueNetwork(self.max_obs).to(self.device)
        #self.value_network.apply(self.init_weights)


        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.policy_network_optimizer, step_size=50, gamma=0.5)
        self.total_reward = 0


        self.rewards, self.log_probs, self.actions, self.states, self.values = [], [], [], [], []
        self.policy_network.train()
        self.value_network.train()


    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)



    def predict(self, observations):
            predictions = []

            for obs in observations:
                if len(obs) < self.max_obs:
                    diff = self.max_obs - len(obs)
                    for i in range(diff):
                        obs.append(0)
                state = torch.tensor(obs, dtype=torch.float32).clone().to(self.device)
                action_probs = self.policy_network(state)
                dist = torch.distributions.Categorical(action_probs)

                action = dist.sample()
                predictions.append(action.clone().detach().cpu())
                self.actions.append(action.clone().detach().cpu())
                self.states.append(state.clone())

                log_prob = dist.log_prob(action)
                self.log_probs.append(log_prob.clone())
                value = self.value_network(state)
                self.values.append(value.clone().detach().cpu())
            
            
            return predictions
    
    
    def update(self, rewards):
        for r in rewards:
            self.rewards.append(r)
    

    def compute_gae(self):
        returns = []
        advantages = []
        G = 0
        advantage = 0
        for r in range(len(self.rewards) - 1, -1, -1):
            G = self.rewards[r] + self.gamma * G
            returns.insert(0, G)
            error = self.rewards[r] + self.gamma * (self.values[r + 1] if r + 1 < len(self.values) else 0) - self.values[r]
            advantage = error + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)

        returns = torch.tensor(returns).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
    


    
    def train(self, t):
        
        if (t+1) % self.number_steps == 0:
            print("Training...")

            returns, advantages = self.compute_gae()



            dataset = TensorDataset(
                torch.stack(self.states),
                torch.stack(self.actions),
                torch.stack(self.log_probs),
                returns,
                advantages
            )
            
            data_loader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)


            for _ in range(self.epochs):
                for batch in data_loader:
                    state_batch, action_batch, old_log_prob_batch, return_batch, advantage_batch = batch
                    state_batch = state_batch.to(self.device)
                    action_batch = action_batch.to(self.device)
                    old_log_prob_batch = old_log_prob_batch.to(self.device).detach()
                    return_batch = return_batch.to(self.device).detach()
                    advantage_batch = advantage_batch.to(self.device).detach()

                    action_probs = self.policy_network(state_batch)
                    action_prob_batch = action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    new_log_prob_batch = torch.log(action_prob_batch + 1e-8)
                    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()

                    ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
                    surr1 = ratio * advantage_batch
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                    policy_network_loss = -torch.min(surr1, surr2).mean()

                    value_network_batch = self.value_network(state_batch).squeeze()
                    value_network_loss = F.mse_loss(value_network_batch.view(-1), return_batch.view(-1)) * self.value_network_loss_coef

                    entropy_loss = -entropy * self.entropy_coef
                    loss = policy_network_loss + value_network_loss + entropy_loss

                    self.policy_network_optimizer.zero_grad()
                    self.value_network_optimizer.zero_grad()
                    loss.backward()
                    self.policy_network_optimizer.step()
                    self.value_network_optimizer.step()

            self.scheduler.step()
        

            self.rewards, self.log_probs, self.actions, self.states, self.values = [], [], [], [], []
            
        else:
            pass

        self.entropy_coef = max(0.01, self.entropy_coef * 0.975)
        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.95)
            

    
