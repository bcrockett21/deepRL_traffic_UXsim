import gymnasium as gym
import numpy as np
import csv
import random
import sys
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import Actor
from small_network import SmallNetwork
from medium_network import MediumNetwork
from large_network import LargeNetwork
from full_network import FullNetwork
#from eigen_ppo import EigenPPO
from plot import PlotRewards
import pickle
import torch.nn as nn
from uxsim import *
from pathlib import Path
import argparse





parser = argparse.ArgumentParser(description="Define traffic flow and model type.")
#parser.add_argument("--traffic_flow", choices=["low", "medium", "high", "all"], default= "input/low", help="Choose from the following traffic scenarios: [low, medium, high, all]")
#parser.add_argument("--model", choices=["PPO", "A2C", "SAC", "all"], default="input/DQN", help="Choose from the following deep learning models: [PPO, A2C, SAC, all]")
#parser.add_argument("--network_size", choices=["1", "2", "3", "4", "all"], default="input/small", help="Choose from the following network sizes: [1, 2, 3, 4, all]")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of 30 second intervals to train the model.")
args = parser.parse_args()

#traffic_flow = args.traffic_flow
#model = args.model
#network_size = args.network_size
timesteps = args.timesteps


traffic_flows =["low", "medium", "high"]    
models = ["TD3"]
network_sizes = ["1"]





class Simulation(gym.Env):

    def __init__(self, traffic_flow, model, network_size, time_steps):
        """""""""
        Simulated Nicholasville Rd environment.
        Nodes:
            - Start
            - 25 intersections
            - 46 Cross Streets
            - 144 links between nodes
            - End

        Action Space:
            - Discrete(25)
                - 0: Green for NS, Red for EW_create
                - 1: Green for EW_create, Red for NS

        State Definition:
            - Box(144)
                - Number of waiting vehicles at each link
        
        Reward:
            - Average speed of vehicles across all links

        """""""""



        self.traffic_flow = traffic_flow
        self.model = model
        self.network_size = network_size
        self.time_steps = time_steps


        if self.network_size == "1":
            self.network = SmallNetwork(self.traffic_flow, self.model)
            self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.load_network(show=False)

        elif self.network_size == "2":
            self.network = MediumNetwork(self.traffic_flow, self.model)
            self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.load_network(show=False)

        elif self.network_size == "3":
            self.network = LargeNetwork(self.traffic_flow, self.model)
            self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.load_network(show=False)

        else:
            self.network = FullNetwork(self.traffic_flow, self.model)
            self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.load_network(show=False)


        self.step_count = 0
        self.log_steps = []
        self.log_rewards = []
        self.log_queues = []
        self.reward_file = f"/Users/blakecrockett/Documents/ds_capstone/data/{self.model}_{self.traffic_flow}_{self.network_size}_step_rewards.csv"





    def reset(self, seed=None, options=None):

        super().reset(seed=seed)


        if seed is not None:
            np.random.seed(seed)
        
        self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.load_network(show=False)

        self.W.rng = np.random.default_rng(seed=seed)

        observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.log_state = []
        self.log_rewards = []


        return observation, None



    def get_state(self):

        link_queues = []

        for link in self.links:

            current_link = self.W.LINKS_NAME_DICT[link]
            queue_length = current_link.num_vehicles_queue
            
            link_queues.append(queue_length)
        
        return link_queues
    


    def compute_queues(self):
        links_queues = []

        for link in self.links:
            queue = self.W.LINKS_NAME_DICT[link].num_vehicles_queue
            #print(link, queue)
            links_queues.append(queue)

        return sum(links_queues)
    


    def step(self, action):
        self.step_count += 1
        self.log_steps.append(self.step_count)

        # Initialize CSV file
        with open(self.reward_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Queue Length", "Reward"])

        reward = 0
           
        action = np.random.binomial(1, action).astype(int)

        prev_sum_queues = sum(self.get_state())

        for i in range(len(action)):

            #print("\nACTION: ", rounded_actions[i])
            self.intersections[i].signal_phase = action[i]

            self.intersections[i].signal_t = 0
            #print("SIGNAL: ", self.intersections[i].signal_phase, "\n")

        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t2=30)



        observation = np.array(self.get_state(), dtype=np.float32)

        sum_queues = sum(self.get_state())
        #print("PREV: ", prev_sum_queues)
        #print("SUM: ", sum_queues)
        reward = -(sum_queues - prev_sum_queues)

        self.log_queues.append(sum_queues)


        if self.step_count == self.time_steps:
            for i in range(len(self.log_steps)):
                with open(self.reward_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.log_steps[i], self.log_queues[i], reward])



        #print(f"Reward: {reward}")

        terminate = False
        truncate = False

        if self.W.check_simulation_ongoing() == False:
            #self.W.analyzer.print_simple_stats()
            #self.W.analyzer.macroscopic_fundamental_diagram()
            #self.W.analyzer.network_anim(detailed=1, network_font_size=0, figsize=(30,30), file_name="/Users/blakecrockett/Documents/ds_capstone/charts/anim_test.gif")
            #self.W.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.1, interval=10, trace_length=5, file_name="/Users/blakecrockett/Documents/ds_capstone/charts/fancy_test.gif")


            terminate = True

        self.log_state.append(observation)
        self.log_rewards.append(reward)

        #print(self.get_state())

        return observation, reward, terminate, truncate, {}
    








class PostTrainingCallback(BaseCallback):
    def __init__(self, custom_function=None, verbose=0):
        super().__init__(verbose)
        self.custom_function = custom_function 

    def on_training_end(self) -> None:
        if self.custom_function:
            self.custom_function()

            


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = None
        self.episode_counts = None

        # Initialize CSV file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])

    def _on_training_start(self) -> None:
        
        num_envs = self.training_env.num_envs
        self.episode_rewards = [0] * num_envs
        self.episode_counts = [0] * num_envs

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        
        for i, done in enumerate(dones):
            self.episode_rewards[i] += rewards[i]
            #print("ACTION: ", self.locals['actions'])
            if done:
                self.episode_counts[i] += 1
                print(f"Env {i}, Episode: {self.episode_counts[i]}, Reward: {self.episode_rewards[i]}")
                
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_counts[i], self.episode_rewards[i]])
                self.episode_rewards[i] = 0
        return True


    

for flow in traffic_flows:

    for size in network_sizes:

            for model in models:

                print(f"Training {model} model on {flow} traffic flow with network size {size}.")


                if model == "PPO":
                    env = Simulation(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
                    # PPO model
                    PPO_model = PPO(
                        "MlpPolicy", 
                        env,
                        learning_rate=1e-4,
                        batch_size=16,
                        n_steps=32,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        n_epochs=5,
                        max_grad_norm=0.5,
                        ent_coef=0.1,
                        vf_coef=0.5,
                        sde_sample_freq=-1,
                        verbose=1,
                        device='cpu'
                    )



                    log_file_path = f"/Users/blakecrockett/Documents/ds_capstone/data/{model}_{flow}_{size}.csv"
                    reward_logger_callback = RewardLoggerCallback(log_file=log_file_path)
                    callback_list = CallbackList([reward_logger_callback])

                    PPO_model.learn(total_timesteps=timesteps, callback=callback_list)
                    print(f"{model} model on {flow} traffic flow with network size {size} training complete.")
            





                elif model == "A2C":
                    env = Simulation(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
                    # A2C model
                    A2C_model = A2C("MlpPolicy", 
                                    env, 
                                    learning_rate=1e-4, 
                                    n_steps=32, 
                                    gamma=0.99, 
                                    gae_lambda=.95,
                                    ent_coef=0.2, 
                                    vf_coef=0.5, 
                                    max_grad_norm=0.5, 
                                    rms_prop_eps=1e-05, 
                                    use_rms_prop=True,
                                    use_sde=False, 
                                    sde_sample_freq=-1, 
                                    rollout_buffer_class=None, 
                                    rollout_buffer_kwargs=None, 
                                    normalize_advantage=False, 
                                    stats_window_size=100, 
                                    tensorboard_log=None, 
                                    policy_kwargs=None, 
                                    verbose=1, 
                                    seed=None, 
                                    device='auto', 
                                    _init_setup_model=True
                                    )



                    log_file_path = f"/Users/blakecrockett/Documents/ds_capstone/data/{model}_{flow}_{size}.csv"
                    reward_logger_callback = RewardLoggerCallback(log_file=log_file_path)
                    callback_list = CallbackList([reward_logger_callback])
                    A2C_model.learn(total_timesteps=timesteps, callback=callback_list)
                    print(f"{model} model on {flow} traffic flow with network size {size} training complete.")




                elif model == "SAC":

                    if size == "1":
                        target_entropy = -4
                    elif size == "2":
                        target_entropy = -8
                    elif size == "3":
                        target_entropy = -12
                    else:
                        target_entropy = -26

                    env = Simulation(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
                    # SAC model
                    SAC_model = SAC(
                        "MlpPolicy", 
                        env,
                        learning_rate=1e-4,
                        buffer_size=8000,
                        learning_starts=1000,
                        batch_size=64,
                        train_freq=1,
                        gamma=0.99,
                        tau=0.005,
                        ent_coef='auto',
                        gradient_steps=1,
                        target_update_interval=1,
                        sde_sample_freq=-1,
                        target_entropy=target_entropy,
                        use_sde_at_warmup=False,
                        verbose=1,
                        device='cpu'
                    )

                    log_file_path = f"/Users/blakecrockett/Documents/ds_capstone/data/{model}_{flow}_{size}.csv"
                    reward_logger_callback = RewardLoggerCallback(log_file=log_file_path)
                    callback_list = CallbackList([reward_logger_callback])

                    SAC_model.learn(total_timesteps=timesteps, callback=callback_list)
                    print(f"{model} model on {flow} traffic flow with network size {size} training complete.")


                else:

                    env = Simulation(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
                    # TD3 model
                    TD3_model = TD3("MlpPolicy", 
                        env, 
                        learning_rate=0.0001, 
                        buffer_size=10000, 
                        learning_starts=100, 
                        batch_size=256, 
                        tau=0.005, 
                        gamma=0.99, 
                        train_freq=1, 
                        gradient_steps=1, 
                        policy_delay=2, 
                        target_policy_noise=0.2, 
                        target_noise_clip=0.5, 
                        stats_window_size=100, 
                        verbose=1, 
                        device='auto', 
                        _init_setup_model=True)

                    log_file_path = f"/Users/blakecrockett/Documents/ds_capstone/data/{model}_{flow}_{size}.csv"
                    reward_logger_callback = RewardLoggerCallback(log_file=log_file_path)
                    callback_list = CallbackList([reward_logger_callback])

                    TD3_model.learn(total_timesteps=timesteps, callback=callback_list)
                    print(f"{model} model on {flow} traffic flow with network size {size} training complete.")

