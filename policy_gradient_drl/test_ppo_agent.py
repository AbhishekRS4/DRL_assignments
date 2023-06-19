import gym
import torch
import argparse
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def test_default_ppo(policy, total_timesteps, file_weights, learning_rate, gamma=0.99, env_name="Tutankham-v4", num_runs=10, num_envs=2):
    vec_env = make_vec_env(env_name, n_envs=num_envs)

    model = PPO(
        policy, vec_env,
        tensorboard_log=f"{file_weights}_{total_timesteps}_logs",
        learning_rate=learning_rate,
        n_steps=1024,
        verbose=1,
        device="auto",
        gamma=gamma
    )

    model = PPO.load(file_weights)

    all_run_rewards = []
    for i_run in range(1, num_runs+1):
        obs = vec_env.reset()
        terminal = False
        list_rewards = []
        while not terminal:
            action, _states = model.predict(obs)
            obs, rewards, terminal, info = vec_env.step(action)
            vec_env.render("human")
            #print(rewards, terminal)
            terminal = terminal.any()
            list_rewards.append(rewards)
        list_rewards = np.array(list_rewards)
        print(f"Run: {i_run}, total rewards obtained")
        summed_rewards = np.sum(list_rewards, axis=0)
        print(summed_rewards)
        all_run_rewards.append(list(summed_rewards))
    #print(np.sum(list_rewards, axis=1))

    all_run_rewards = np.array(all_run_rewards)
    print(all_run_rewards)
    print(all_run_rewards[:, 0])
    print(all_run_rewards[:, 1])

    fig = plt.figure()
    plt.plot(np.arange(1, len(all_run_rewards)+1), all_run_rewards[:, 0], label="env_1")
    plt.plot(np.arange(1, len(all_run_rewards)+1), all_run_rewards[:, 1], label="env_2")
    plt.title("testing sum of rewards vs test game", fontsize=20)
    plt.xlabel("test game number", fontsize=20)
    plt.ylabel("sum of testing reward", fontsize=20)
    plt.grid()
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return

def test_ppo_agent(ARGS):
    if ARGS.which_cnn == "default_cnn":
        test_default_ppo(
            ARGS.policy,
            ARGS.total_timesteps,
            ARGS.file_weights,
            ARGS.learning_rate,
            gamma=ARGS.gamma,
            env_name=ARGS.env_name,
            num_envs=ARGS.num_envs,
            num_runs=ARGS.num_runs,
        )
    elif ARGS.which_cnn == "custom_cnn":
        print("not yet implemented")
    else:
        print(f"wrong option for (which_cnn={ARGS.which_cnn}) entered")
    return

def main():
    policy = "CnnPolicy"
    which_cnn = "default_cnn"
    env_name = "Tutankham-v4"

    num_envs = 2
    total_timesteps = 300000
    learning_rate = 1e-4
    gamma = 0.99
    num_runs = 10
    file_weights = f"ppo_{which_cnn}_tutankham_{total_timesteps}"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--policy", default=policy,
        type=str, help="policy that needs to be used for training")
    parser.add_argument("--which_cnn", default=which_cnn,
        type=str, help="which CNN needs to used for training")
    parser.add_argument("--file_weights", default=file_weights,
        type=str, help="file name to be used for saving weights")
    parser.add_argument("--env_name", default=env_name,
        type=str, help="which env to be used for training")

    parser.add_argument("--num_runs", default=num_runs,
        type=int, help="number of test runs")
    parser.add_argument("--num_envs", default=num_envs,
        type=int, help="number of parallel envs to be used for training")
    parser.add_argument("--total_timesteps", default=total_timesteps,
        type=int, help="number of total timesteps for which the model needs to be trained")
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate to be used")
    parser.add_argument("--gamma", default=gamma,
        type=float, help="gamma i.e. the discount factor")

    ARGS, unparsed = parser.parse_known_args()
    test_ppo_agent(ARGS)
    return


if __name__ == "__main__":
    main()
