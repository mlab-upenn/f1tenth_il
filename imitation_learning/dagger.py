import gym
import torch
import numpy as np

import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils

from dataset import Dataset

from bc import bc

from pathlib import Path

def dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    best_model_saving_threshold = 500000

    algo_name = "DAgger"
    best_model = agent
    longest_distance_travelled = 0


    # For Sim2Real
    path = "logs/{}".format(algo_name)
    num_of_saved_models = 0


    num_of_expert_queries = 0

    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10

    init_traj_len = 50
    max_traj_len = 10000
    n_batch_updates_per_iter = 1000
    n_iter = 5000

    train_batch_size = 64

    # np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Number of Samples': [], 
           'Number of Expert Queries': [], 
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform num_iter iterations of DAgger
    for iter in range(n_iter + 1):
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))


        # Evaluate the agent's performance
        # No evaluation at the initial iteration
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)
            # log["Iteration"].append(iter)
            mean_travelled_distances, stdev_travelled_distances, mean_reward, stdev_reward = agent_utils.eval(env, agent, start_pose, max_traj_len, eval_batch_size, observation_shape, downsampling_method, render, render_mode)
            
            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)
            
            # Replace the best model if the current model is better
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled) and (log['Number of Samples'][-1] < best_model_saving_threshold):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = agent


            
            # For Sim2Real
            if (log['Mean Distance Travelled'][-1] > 100):
                curr_dist = log['Mean Distance Travelled'][-1]
                current_expsamples = log['Number of Expert Queries'][-1]
                model_path = Path(path + f'/{algo_name}_svidx_{str(num_of_saved_models)}_dist_{int(curr_dist)}_expsamp_{int(current_expsamples)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True) 
                torch.save(agent.state_dict(), model_path)
                num_of_saved_models += 1



            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)

            # DELETE IT WHEN DOING SIM2REAL
            if log['Number of Samples'][-1] > 3000:
                break


        if iter == n_iter:
            break
        
        

        # Sample a trajectory with the agent and re-lable actions with the expert
        

        # Disable render for the initial iteration as it takes too much time
        # The max trajectory length is also different in the initial iteration
        if iter == 0:
            agent, log, dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='bootstrap')
        else:
            print("Sampling trajectory...")
            data = agent_utils.sample_traj(env, agent, start_pose, max_traj_len, observation_shape, downsampling_method, render, render_mode)

            # tlad and vgain are fixed value for the vehicle dynamics model
            tlad = 0.82461887897713965
            vgain = 0.90338203837889

            # Extract necessary input information from observation in the sampled trajectory
            poses_x = data['poses_x']
            poses_y = data['poses_y']
            poses_theta = data['poses_theta']


            # Get expert speed and steer and concat into expert action
            print("Expert labeling...")
            for idx in range(data['actions'].shape[0]):
                curr_poses_x = poses_x[idx][0]
                curr_poses_y = poses_y[idx][0]
                curr_poses_theta = poses_theta[idx][0]

                curr_expert_speed, curr_expert_steer = expert.plan(curr_poses_x, curr_poses_y, curr_poses_theta, tlad, vgain)
                curr_expert_action = np.array([[curr_expert_steer, curr_expert_speed]])
                # Replace original action with expert labeled action

                processed_steer = (curr_expert_steer / 2) + 0.5
                data["actions"][idx] = processed_steer

                num_of_expert_queries += 1


            # Aggregate the datasets
            print("Aggregating dataset...")
            dataset.add(data)

            log['Number of Samples'].append(dataset.get_num_of_total_samples())
            log['Number of Expert Queries'].append(num_of_expert_queries)


            # Train the agent
            print("Training agent...")
            for _ in range(n_batch_updates_per_iter):
                train_batch = dataset.sample(train_batch_size)
                agent.train(train_batch["scans"], train_batch["actions"])

    # Save log and the best model
    agent_utils.save_log_and_model(log, best_model, algo_name)