import os
import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn

from imageio import imread, mimsave

from train_drl_agent_dqn import DRLAgentDQN


def test_drl_agent_dqn(ARGS):
    env = CatchEnv(low_dim=bool(ARGS.low_dim))
    print(f"Testing DQN model: {ARGS.which_model}")

    print(f"Creating a DQN agent")
    drl_agent_dqn = DRLAgentDQN(
        env,
        ARGS.gamma,
        ARGS.learning_rate,
        ARGS.exp_buffer_size,
        ARGS.batch_size,
        loss_function=ARGS.loss_function,
        which_model=ARGS.which_model,
        which_optimizer=ARGS.which_optimizer,
        low_dim=ARGS.low_dim,
    )

    print(f"Loading trained model from: {ARGS.file_model}")
    drl_agent_dqn.load_model(ARGS.file_model)

    count_test_wins = 0
    for game_count in range(ARGS.num_test_games):
        test_reward = 0
        test_reward = drl_agent_dqn.play_game(game_count, ARGS.dir_save_states, save_states=bool(ARGS.save_state))

        if test_reward == 1:
            count_test_wins += 1

        for game_count in range(ARGS.num_test_games):
            list_images = os.listdir(ARGS.dir_save_states)
            state_image =



    return

def main():
    low_dim = 0
    which_model = "dqn_residual"


    return

if __name__ == "__main__":
    main()
