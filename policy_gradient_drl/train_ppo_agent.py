import gym
import torch
import argparse
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train_default_ppo(policy, total_timesteps, file_weights, learning_rate, gamma=0.99, env_name="Tutankham-v4", num_envs=4):
    # Parallel environments
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
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{file_weights}_{total_timesteps}")

    """
    del model # remove to demonstrate saving and loading
    model = PPO.load(file_weights)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
    """
    return


def train_ppo_agent(ARGS):
    if ARGS.which_cnn == "default_cnn":
        train_default_ppo(
            ARGS.policy,
            ARGS.total_timesteps,
            ARGS.file_weights,
            ARGS.learning_rate,
            gamma=ARGS.gamma,
            env_name=ARGS.env_name,
            num_envs=ARGS.num_envs,
        )
    elif ARGS.which_cnn == "custom_cnn":
        print("not yet implemented")
    else:
        print(f"wrong option for (which_cnn={ARGS.which_cnn}) entered")
    return


def main():
    policy = "CnnPolicy"
    which_cnn = "default_cnn"
    file_weights = f"ppo_{which_cnn}_tutankham"
    env_name = "Tutankham-v4"

    num_envs = 4
    total_timesteps = 300000
    learning_rate = 1e-4
    gamma = 0.99

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

    parser.add_argument("--num_envs", default=num_envs,
        type=int, help="number of parallel envs to be used for training")
    parser.add_argument("--total_timesteps", default=total_timesteps,
        type=int, help="number of total timesteps for which the model needs to be trained")
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate to be used")
    parser.add_argument("--gamma", default=gamma,
        type=float, help="gamma i.e. the discount factor")

    ARGS, unparsed = parser.parse_known_args()
    train_ppo_agent(ARGS)
    return


if __name__ == "__main__":
    main()
