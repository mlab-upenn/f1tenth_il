import torch
import numpy as np
import yaml
import argparse
import gym
import json

import utils.downsampling as downsampling


from dictances import bhattacharyya, bhattacharyya_coefficient
from policies.agents.agent_mlp import AgentPolicyMLP
from policies.experts.expert_waypoint_follower import ExpertWaypointFollower
import utils.env_utils as env_utils


from ppo_continuous import PPO

import math


def mean( hist ):
    mean = 0.0
    for i in hist:
        mean += i
    mean/= len(hist)
    return mean

def bhatta ( hist1,  hist2):
    # calculate mean of hist1
    h1_ = mean(hist1)

    # calculate mean of hist2
    h2_ = mean(hist2)

    # calculate score
    score = 0
    for i in range(8):
        score += math.sqrt( np.abs(hist1[i] * hist2[i]) )
    # print h1_,h2_,score;
    # print("h1_:", h1_)
    # print("h2_:", h2_)
    # print("score:", score)
    score = math.sqrt( np.abs(1 - ( 1 / math.sqrt(np.abs(h1_*h2_*8*8)) ) * score))
    return score





episode = 20000

# max_step_num = 1000
kwargs = json.load(open('rlf110_ppo_continuouscfg.json'))
kwargs['state_dim'] = 54
kwargs['action_dim'] = 1
kwargs['env_with_Dead'] = False
# Load agent model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ppo_stock = PPO(**kwargs)
ppo_stock.critic.load_state_dict(torch.load("./ppo_stock/ppo_critic{}.pth".format(episode)))
ppo_stock.actor.load_state_dict(torch.load("./ppo_stock/ppo_actor{}.pth".format(episode)))

ppo_bc = PPO(**kwargs)
ppo_bc.critic.load_state_dict(torch.load("./ppo_bc/ppo_critic{}.pth".format(episode)))
ppo_bc.actor.load_state_dict(torch.load("./ppo_bc/ppo_actor{}.pth".format(episode)))

ppo_dagger = PPO(**kwargs)
ppo_dagger.critic.load_state_dict(torch.load("./ppo_dagger/ppo_critic{}.pth".format(episode)))
ppo_dagger.actor.load_state_dict(torch.load("./ppo_dagger/ppo_actor{}.pth".format(episode)))

ppo_hg_dagger = PPO(**kwargs)
ppo_hg_dagger.critic.load_state_dict(torch.load("./ppo_hgdagger/ppo_critic{}.pth".format(episode)))
ppo_hg_dagger.actor.load_state_dict(torch.load("./ppo_hgdagger/ppo_actor{}.pth".format(episode)))

ppo_eil = PPO(**kwargs)
ppo_eil.critic.load_state_dict(torch.load("./ppo_eil/ppo_critic{}.pth".format(episode)))
ppo_eil.actor.load_state_dict(torch.load("./ppo_eil/ppo_actor{}.pth".format(episode)))

# Initialize dictionaries
# state_dict = {'idx': [],
#               'poses_x': [],
#               'poses_y': [],
#               'poses_theta': [],
#               'scans': []}

# expert_speed_dict = {}

expert_steer_dict = {}


stock_ppo_steer_dict = {}

# bc_agent_speed_dict = {}

bc_ppo_steer_dict = {}

# dagger_agent_speed_dict = {}

dagger_ppo_steer_dict = {}

# hgdagger_agent_speed_dict = {}

hgdagger_ppo_steer_dict = {}

# eil_agent_speed_dict = {}

eil_ppo_steer_dict = {}



expert_steer_list = []
stock_ppo_steer_list = []
bc_agent_steer_list = []
dagger_agent_steer_list = []
hgdagger_agent_steer_list = []
eil_agent_steer_list = []


with open('map/gene_eval_map/config_gene_map.yaml') as file:
    map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)

map_conf = argparse.Namespace(**map_conf_dict)
env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
env.add_render_callback(env_utils.render_callback)
start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
expert = ExpertWaypointFollower(map_conf)

tlad = 0.82461887897713965
vgain = 0.90338203837889

observ, step_reward, done, info = env.reset(start_pose)

curr_idx = 0
while not done:
# for _ in range(max_step_num):
    # state_dict['idx'].append(i)

    poses_x = observ["poses_x"][0]
    poses_y = observ["poses_y"][0]
    poses_theta = observ["poses_theta"][0]
    scans = observ["scans"][0]

    processed_lidar_scan = downsampling.downsample(scans, 54, 'simple')

    # Get expert action
    curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)

    # Log expert action
    # expert_speed_dict['idx'].append(curr_idx)
    # expert_speed_dict['speed'].append(curr_expert_speed)
    # expert_steer_dict['idx'].append(curr_idx)
    # expert_steer_dict['steer'].append(curr_expert_steer)
    expert_steer_dict[str(curr_idx)] = curr_expert_steer

    expert_steer_list.append(np.abs(curr_expert_steer))

    # Concat expert action
    expert_action = np.array([[curr_expert_steer, curr_expert_speed]])

    # Get agent action
    stock_a, stock_logprob_a = ppo_stock.evaluate(processed_lidar_scan)
    # bc_agent_speed_dict['idx'].append(curr_idx)
    # bc_agent_speed_dict['speed'].append(bc_agent_speed)
    # bc_agent_steer_dict['idx'].append(curr_idx)
    # bc_agent_steer_dict['steer'].append(bc_agent_steer)
    stock_ppo_steer_dict[str(curr_idx)] = 2.0 * (stock_a - 0.5) * 1.0
    stock_ppo_steer_list.append(2.0 * (stock_a - 0.5) * 1.0)

    bc_a, bc_logprob_a = ppo_bc.evaluate(processed_lidar_scan)
    bc_ppo_steer_dict[str(curr_idx)] = 2.0 * (bc_a - 0.5) * 1.0
    bc_agent_steer_list.append(2.0 * (bc_a - 0.5) * 1.0)

    dagger_a, dagger_logprob_a = ppo_dagger.evaluate(processed_lidar_scan)
    dagger_ppo_steer_dict[str(curr_idx)] = 2.0 * (dagger_a - 0.5) * 1.0
    dagger_agent_steer_list.append(2.0 * (dagger_a - 0.5) * 1.0)

    hgdagger_a, hgdagger_logprob_a = ppo_hg_dagger.evaluate(processed_lidar_scan)
    hgdagger_ppo_steer_dict[str(curr_idx)] = 2.0 * (hgdagger_a - 0.5) * 1.0
    hgdagger_agent_steer_list.append(2.0 * (hgdagger_a - 0.5) * 1.0)

    eil_a, eil_logprob_a = ppo_eil.evaluate(processed_lidar_scan)
    eil_ppo_steer_dict[str(curr_idx)] = 2.0 * (eil_a - 0.5) * 1.0
    eil_agent_steer_list.append(2.0 * (eil_a - 0.5) * 1.0)





    # Step environment
    observ, step_reward, done, info = env.step(expert_action)
    # env.render(mode='human_fast')

    if env.lap_counts[0] > 0:
        break

# Calculate bhattacharyya distance

# stock_bhattacharyya = bhattacharyya(expert_steer_dict, stock_ppo_steer_dict)
# bc_bhattacharyya = bhattacharyya(expert_steer_dict, bc_ppo_steer_dict)
# dagger_bhattaryya = bhattacharyya(expert_steer_dict, dagger_ppo_steer_dict)
# hgdagger_bhattacharyya = bhattacharyya(expert_steer_dict, hgdagger_ppo_steer_dict)
# eil_bhattacharyya = bhattacharyya(expert_steer_dict, eil_ppo_steer_dict)

stock_bhattacharyya = bhatta(expert_steer_list, stock_ppo_steer_list)
bc_bhattacharyya = bhatta(expert_steer_list, bc_agent_steer_list)
dagger_bhattaryya = bhatta(expert_steer_list, dagger_agent_steer_list)
hgdagger_bhattacharyya = bhatta(expert_steer_list, hgdagger_agent_steer_list)
eil_bhattacharyya = bhatta(expert_steer_list[300:500], eil_agent_steer_list[300:500])

print('Stock PPO Bhattacharyya Metric: ', stock_bhattacharyya)
print('BC+PPO Bhattacharyya Metric: ', bc_bhattacharyya)
print('Dagger+PPO Bhattacharyya Metric: ', dagger_bhattaryya)
print('HG-Dagger+PPO Bhattacharyya Metric: ', hgdagger_bhattacharyya)
print('EIL+PPO Bhattacharyya Metric: ', eil_bhattacharyya)
                