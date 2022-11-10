import torch
import gym
import numpy as np
import argparse
import yaml

from policies.agents.agent_mlp import AgentPolicyMLP
from policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import utils.env_utils as env_utils

from bc import bc
from dagger import dagger
from hg_dagger import hg_dagger


def process_parsed_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algorithm', type=str, default='dagger', help='imitation learning algorithm to use')
    arg_parser.add_argument('--training_config', type=str, required=True, help='the yaml file containing the training configuration')
    return arg_parser.parse_args()

def initialization(il_config):
    seed = il_config['random_seed']
    # np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    

    # obs, step_reward, done, info = env.reset(np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]]))


    # Initialize the agent
    if il_config['policy_type']['agent']['model'] == 'mlp':
        agent = AgentPolicyMLP(il_config['policy_type']['agent']['observation_shape'], \
                                il_config['policy_type']['agent']['hidden_dim'], \
                                2, \
                                il_config['policy_type']['agent']['learning_rate'], \
                                device)
    else:
        #TODO: Implement other model (Transformer)
        pass


    # Initialize the expert
    if il_config['policy_type']['expert']['behavior']  == 'waypoint_follower':
        expert = ExpertWaypointFollower(map_conf)
    else:
        # TODO: Implement other expert behavior (Lane switcher and hybrid)
        pass
    
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
    # observation_gap = int(1080/il_config['policy_type']['agent']['observation_shape'])
    observation_shape = il_config['policy_type']['agent']['observation_shape']
    downsampling_method = il_config['policy_type']['agent']['downsample_method']

    render = il_config['environment']['render']
    render_mode = il_config['environment']['render_mode']

    return seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode
    

def train(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    if il_algo == 'bc':
        bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='train')
    elif il_algo == 'dagger':
        dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)
    elif il_algo == 'hg-dagger':
        hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)
    else:
        # TODO: Implement other IL algorithms (BC, HG DAgger, etc.)
        pass


if __name__ == '__main__':
    # Parse the command line arguments.
    parsed_args = process_parsed_args()
    
    # Process the parsed arguments.
    il_algo = parsed_args.algorithm
    yaml_loc = parsed_args.training_config

    il_config = yaml.load(open(yaml_loc), Loader=yaml.FullLoader)

    # Initialize
    seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode = initialization(il_config)

    # Train
    train(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)

    
    
