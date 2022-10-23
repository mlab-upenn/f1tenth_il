import gym
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os, json 

from . import downsampling

def downsample_and_extract_lidar(observ, observation_shape, downsampling_method):
    """
    Downsamples the lidar data and extracts the relevant features.
    """
    # print("observ: ", observ)
    lidar_scan = observ["scans"][0]
    processed_lidar_scan = downsampling.downsample(lidar_scan, observation_shape, downsampling_method)
    return processed_lidar_scan

def sample_traj(env, policy, start_pose, max_traj_len, observation_shape=108, downsampling_method="simple", render=True, render_mode="human_fast", for_eval=False):
    """
    Samples a trajectory of at most `max_traj_len` timesteps by executing a policy.
    """
    if for_eval:
        avg_velocity = 0
        action_count = 0
        traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0, "travelled_distance": 0}
    else:
        traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}


    done = False
    observ, step_reward, done, info = env.reset(start_pose)

    # Start rendering
    if render:
        if env.renderer is None:
            env.render()

    for _ in range(max_traj_len):
        traj["observs"].append(observ)
        
        scan = downsample_and_extract_lidar(observ, observation_shape, downsampling_method)

        # Add Sim2Real noise
        sim2real_noise = np.random.uniform(-0.25, 0.25, scan.shape)
        scan = scan + sim2real_noise

        traj["scans"].append(scan)

        traj["poses_x"].append(observ["poses_x"][0])
        traj["poses_y"].append(observ["poses_y"][0])
        traj["poses_theta"].append(observ["poses_theta"][0])

        action = policy.get_action(scan)

        if for_eval:
            curr_velocity = action[1]
            avg_velocity = (avg_velocity * action_count + curr_velocity) / (action_count + 1)
            action_count += 1

        # TODO: for multi-agent the dimension expansion need to be changed
        action_expand = np.expand_dims(action, axis=0)
        # print("action_expand shape: ", action_expand.shape)
        observ, reward, done, _ = env.step(action_expand)

        # Update rendering
        if render:
            env.render(mode=render_mode)

        traj["actions"].append(action)
        traj["reward"] += reward
        if done:
            break
    traj["observs"] = np.vstack(traj["observs"])
    traj["poses_x"] = np.vstack(traj["poses_x"])
    traj["poses_y"] = np.vstack(traj["poses_y"])
    traj["poses_theta"] = np.vstack(traj["poses_theta"])
    traj["scans"] = np.vstack(traj["scans"])
    traj["actions"] = np.vstack(traj["actions"])

    if for_eval:
        travelled_distance = avg_velocity * env.lap_times[0]
        traj["travelled_distance"] = travelled_distance
    
    return traj

def sample_trajs(env, policy, start_pose, max_traj_len, n_trajs, observation_shape, downsampling_method, render, render_mode, for_eval=False):
    """
    Samples `n_trajs` trajectories by repeatedly calling sample_traj().
    """
    if for_eval:
        data = {"observs":[], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions":[], "rewards":[], "travelled_distances": []}
    else:
        data = {"observs":[], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions":[], "rewards":[]}

    for _ in range(n_trajs):
        traj = sample_traj(env, policy, start_pose, max_traj_len, observation_shape, downsampling_method, render, render_mode, for_eval)
        data["observs"].append(traj["observs"])
        data["poses_x"].append(traj["poses_x"])
        data["poses_y"].append(traj["poses_y"])
        data["poses_theta"].append(traj["poses_theta"])
        data["scans"].append(traj["scans"])
        data["actions"].append(traj["actions"])
        data["rewards"].append(traj["reward"])

        if for_eval:
            data["travelled_distances"].append(traj["travelled_distance"])

    data["observs"] = np.concatenate(data["observs"])
    data["poses_x"] = np.concatenate(data["poses_x"])
    data["poses_y"] = np.concatenate(data["poses_y"])
    data["poses_theta"] = np.concatenate(data["poses_theta"])
    data["scans"] = np.concatenate(data["scans"])
    data["actions"] = np.concatenate(data["actions"])
    data["rewards"] = np.array(data["rewards"])
    data["travelled_distances"] = np.array(data["travelled_distances"])
    return data

def eval(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode):
    """
    Evaluates the performance of a policy over `eval_batch_size` trajectories.
    """
    eval_res = sample_trajs(env, policy, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode, for_eval=True)

    rewards = eval_res["rewards"]
    travelled_distances = eval_res["travelled_distances"]
    return np.mean(travelled_distances), np.std(travelled_distances), np.mean(rewards), np.std(rewards)


def make_log(log, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(log, f)


def save_log_and_model(log, agent, algo_name):
    path = "logs/{}".format(algo_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save log
    df = pd.DataFrame(log)
    log_path = Path(path + f'/{algo_name}_log.csv')
    log_path.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(log_path, index=False)

    # Save model
    model_path = Path(path + f'/{algo_name}_model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True) 
    torch.save(agent.state_dict(), model_path)


def check_ittc(ego_distance, linear_vels_x, ittc_threshold = 0.5, scan_num = 1080):
    """
    Checks the time to collision (TTC) of the agent based on the lidar scan.
    """

    angle_span = np.linspace(-0.75 * np.pi, 0.75 * np.pi, scan_num)
    ego_speed_proj = np.cos(angle_span) * linear_vels_x
    ego_speed_proj[ego_speed_proj <= 0.0] = 0.001
    raw_ittc = ego_distance / ego_speed_proj
    if np.min(raw_ittc) > ittc_threshold:
        within_threshold = False
        abs_ittc = 0.0
    else:
        within_threshold = True
        abs_ittc = np.min(raw_ittc)
    
    return within_threshold, abs_ittc