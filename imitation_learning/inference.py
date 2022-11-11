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

def process_parsed_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--training_config', required=True, help='the yaml file containing the training configuration')
    arg_parser.add_argument('--model_path', type=str, required=True, help='path to the model for inference')
    return arg_parser.parse_args()

if __name__ == '__main__':
    parsed_args = process_parsed_args()

    model_path = parsed_args.model_path

    yaml_loc = parsed_args.training_config
    il_config = yaml.load(open(yaml_loc), Loader=yaml.FullLoader)

    model_type = il_config['policy_type']['agent']['model']

    seed = il_config['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'mlp':
        agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'], \
                                il_config['policy_type']['agent']['hidden_dim'], \
                                1, \
                                il_config['policy_type']['agent']['learning_rate'], \
                                device)
    else:
        #TODO: Implement other model (Transformer)
        pass

    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    observation_shape = il_config['policy_type']['agent']['observation_shape']
    downsampling_method = il_config['policy_type']['agent']['downsample_method']


    # Initialize the environment
    map_conf = None

    if il_config['environment']['random_generation'] == False:
        if il_config['environment']['map_config_location'] == None:
            # If no environment is specified but random generation is off, use the default gym environment
            with open('map/example_map/config_example_map.yaml') as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            # If an environment is specified and random generation is off, use the specified environment
            with open(il_config['environment']['map_config_location']) as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        map_conf = argparse.Namespace(**map_conf_dict)
        env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
        env.add_render_callback(env_utils.render_callback)
    else:
        # TODO: If random generation is on, generate random environment
        pass
    
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
    
    obs, step_reward, done, info = env.reset(start_pose)

    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        raw_lidar_scan = obs["scans"][0]
        processed_lidar_scan = downsampling.downsample(raw_lidar_scan, observation_shape, downsampling_method)

        action_raw = agent.get_action(processed_lidar_scan)
        action = np.array([[action_raw, 3.0]])
        # action_expand = np.expand_dims(action, axis=0)
        obs, reward, done, _ = env.step(action)

        print("step_reward: ", step_reward)

        laptime += step_reward

        env.render(mode='human')
    
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)