from uxsim import *
from uxsim.OSMImporter import OSMImporter
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
from downtown_lexington import Lexington
from plot import PlotRewards
import pickle
import torch.nn as nn
from pathlib import Path
import argparse




traffic_flows = ["low", "medium", "high"]
network_sizes = ["downtown"]
models = ["PPO", "A2C", "SAC", "TD3"]
# downtown lexington



parser = argparse.ArgumentParser(description="Define traffic flow and model type.")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of 30 second intervals to train the model.")
args = parser.parse_args()
timesteps = args.timesteps


class LexingtonDeepRL(gym.Env):
    def __init__ (self, traffic_flow, model, network_size, time_steps):
        self.traffic_flow = traffic_flow
        self.model = model
        self.network_size = network_size
        self.time_steps = time_steps
        self.step_count = 0
        self.coordinates = [38.053903704276856, 38.0329942363066, -84.48193338399388, -84.51057394078408]
        self.log_steps = []
        self.log_rewards = []
        self.log_queues = []
        self.reward_file = f"/Users/blakecrockett/Documents/ds_capstone/data/{self.model}_{self.traffic_flow}_{self.network_size}_step_rewards.csv"

        

        if self.network_size == "downtown":
            self.network = Lexington(self.coordinates, self.traffic_flow, show=True)
            self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.create()




    def reset(self, seed=None, options=None):

        super().reset(seed=seed)


        if seed is not None:
            np.random.seed(seed)
        
        self.W, self.links, self.intersections, self.action_space, self.observation_space = self.network.create()

        self.W.rng = np.random.default_rng(seed=seed)

        observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.log_state = []
        self.log_rewards = []


        return observation, None



    def get_state(self):

        link_queues = []

        for link in self.links:

            queue_length = self.W.LINKS_NAME_DICT[link].num_vehicles_queue
            
            link_queues.append(queue_length)
        
        return link_queues
    
    


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
            self.intersections[i].signal_phase = action[i]
            self.intersections[i].signal_t = 0


        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t2=30)

        observation = np.array(self.get_state(), dtype=np.float32)

        sum_queues = sum(self.get_state())
        reward = -(sum_queues - prev_sum_queues)

        self.log_queues.append(sum_queues)

        if self.step_count == self.time_steps:
            for i in range(len(self.log_steps)):
                with open(self.reward_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.log_steps[i], self.log_queues[i], reward])


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
                    env = LexingtonDeepRL(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
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
                    env = LexingtonDeepRL(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
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

                    env = LexingtonDeepRL(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
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

                    env = LexingtonDeepRL(traffic_flow=flow, model=model, network_size=size, time_steps=timesteps)
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
