import gym
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os, json
import argparse
import yaml 
import time

import utils.downsampling as downsampling
import utils.env_utils as env_utils

from policies.agents.agent_mlp import AgentPolicyMLP

# random_seed = 0

# np.random.seed(random_seed)
# torch.manual_seed(random_seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = AgentPolicyMLP(108, 128, 2, 0.001, device)

agent.load_state_dict(torch.load('logs/DAgger/DAgger_model.pkl'))

with open('map/example_map/config_example_map.yaml') as file:
    map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)

map_conf = argparse.Namespace(**map_conf_dict)
env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
env.add_render_callback(env_utils.render_callback)

start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])



obs, step_reward, done, info = env.reset(start_pose)

if env.renderer is None:
    env.render()

while not done:
    raw_lidar_scan = obs["scans"][0]
    processed_lidar_scan = downsampling.downsample(raw_lidar_scan, 108, 'simple')

    action = agent.get_action(processed_lidar_scan)
    action_expand = np.expand_dims(action, axis=0)
    obs, reward, done, _ = env.step(action_expand)

    env.render(mode='human_fast')

    if env.lap_counts[0] > 0:
        break

curr_lap_counts = env.lap_counts[0]
curr_lap_times = env.lap_times[0]

print("-"*30)
# print("Evaluated Model: ", key)
print("Total Lap Counts: ", curr_lap_counts)
print("Total Lap Times: ", curr_lap_times)
# print("Elapsed Time for 1 Round: ", curr_lap_times/curr_lap_counts)