import os
import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn

from copy import deepcopy
from torch.autograd import Variable
from torch.optim import RMSprop, Adam
from collections import deque, namedtuple
from random import randint, sample, random
from pyats.datastructures import NestedAttrDict

from dqn import *
from catch import CatchEnv
from logger_utils import CSVWriter, write_dict_to_json


class ExperienceReplayBuffer(object):
    def __init__(self, exp_buffer_size):
        self.replay_buffer = deque(maxlen=exp_buffer_size)
        self.TransitionTable = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))

    def store_experience(self, dict_experience):
        dict_experience = NestedAttrDict(**dict_experience)
        next_state = None
        if dict_experience.next_state is not None:
            next_state = torch.FloatTensor(dict_experience.next_state)

        transition_table = self.TransitionTable(
            state=torch.FloatTensor(dict_experience.state),
            action=torch.LongTensor([[dict_experience.action]]),
            reward=torch.FloatTensor([dict_experience.reward]),
            next_state=next_state,
            terminal=dict_experience.terminal,
        )
        self.replay_buffer.append(transition_table)
        return

    def sample_experiences(self, batch_size):
        experiences = sample(self.replay_buffer, batch_size)
        experiences = self.TransitionTable(*(zip(*experiences)))
        return experiences

    def is_exp_replay_available(self, batch_size):
        is_exp_available = None
        if len(self.replay_buffer) >= batch_size:
            is_exp_available = True
        else:
            is_exp_available = False
        return is_exp_available


class DRLAgentDQN(object):
    def __init__(self, env,
                 gamma,
                 learning_rate,
                 exp_buffer_size,
                 batch_size=32,
                 loss_function="huber",
                 which_model="dqn_simple",
                 which_optimizer="rms_prop",
        ):
        self.step = 0
        self.env = env
        self.dqn_eval = None
        self.gamma = gamma
        self.criterion = None
        self.optimizer = None
        self.dqn_target = None
        self.is_apply_clip = True
        self.batch_size = batch_size
        self.num_frames_in_state = 4
        self.output_shape = (84, 84)
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.num_actions = self.env.get_num_actions()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_replay_buffer = ExperienceReplayBuffer(exp_buffer_size)

        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1000000

        if which_model.lower() == "dqn_simple":
            self.dqn_eval = DQNSimple(self.num_actions)
            self.dqn_target = DQNSimple(self.num_actions)
        elif which_model.lower() == "dqn_simple_new":
            self.dqn_eval = DQNSimpleNew(self.num_actions)
            self.dqn_target = DQNSimpleNew(self.num_actions)
        elif which_model.lower() == "dqn_residual":
            self.dqn_eval = DQNResidual(self.num_actions)
            self.dqn_target = DQNResidual(self.num_actions)
        else:
            print(f"unidentified option for (which_model={which_model}), should be one of ['dqn_simple', 'dqn_residual']")

        self.dqn_eval.to(self.device)
        self.dqn_target.to(self.device)

        if which_optimizer.lower() == "rms_prop":
            self.optimizer = RMSprop(self.dqn_eval.parameters(), lr=self.learning_rate)
        elif which_optimizer.lower() == "adam":
            self.optimizer = Adam(self.dqn_eval.parameters(), lr=self.learning_rate)
        else:
            print(f"unidentified option for (which_optimizer={which_optimizer}), should be one of ['RMSprop', 'Adam']")

        if self.loss_function.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_function.lower() == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        elif self.loss_function.lower() == "huber":
            self.criterion = nn.HuberLoss()
        else:
            print(f"unidentified option for (loss_function={loss_function}), should be one of ['mse', 'smooth_l1', 'huber']")

    def get_action(self, state):
        sample_action = None

        if random() <= self.epsilon:
            # choose a random action from the set of action space
            sample_action = randint(0, self.num_actions - 1)
        else:
            state = np.expand_dims(state, 0)
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.reshape(
                [
                    -1,
                    self.num_frames_in_state,
                    self.output_shape[0],
                    self.output_shape[1],
                ]
            ).to(self.device)

            # choose the action for which the predicted q_value is the max
            q_values = self.dqn_eval.forward(state_tensor)
            q_values = q_values.data.cpu().numpy()
            sample_action = np.argmax(q_values, 1)[0]
        return sample_action

    def update_target_model(self,):
        self.dqn_target.load_state_dict(self.dqn_eval.state_dict())
        return

    def save_model(self, file_checkpoint):
        dict_checkpoint = {
            "dqn_eval": self.dqn_eval.state_dict(),
            "dqn_target": self.dqn_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(dict_checkpoint, file_checkpoint)
        return

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= ((self.epsilon_start - self.epsilon_end) / self.epsilon_decay)
        return

    def _compute_loss(self, q_values, target_values):
        loss = self.criterion(q_values, target_values)
        return loss

    def learn(self):
        # update epsilon
        self._update_epsilon()

        # sample experiences from the experience replay buffer
        state_transitions = self.exp_replay_buffer.sample_experiences(self.batch_size)

        batch_state = torch.cat(state_transitions.state).to(self.device)
        batch_action = torch.cat(state_transitions.action).to(self.device)
        batch_reward = torch.cat(state_transitions.reward).to(self.device)
        batch_next_state = torch.cat(state_transitions.next_state).to(self.device)
        batch_terminal = state_transitions.terminal

        # reshape state and next_state tensors
        batch_state = batch_state.view(
            [
                self.batch_size,
                self.num_frames_in_state,
                self.output_shape[0],
                self.output_shape[1],
            ]
        )
        batch_next_state = batch_next_state.view(
            [
                self.batch_size,
                self.num_frames_in_state,
                self.output_shape[0],
                self.output_shape[1],
            ]
        )

        # apply reward clipping between -1 and 1
        batch_reward.data.clamp_(-1, 1)

        # apply zero grad for the optimizer
        self.optimizer.zero_grad()

        # predict q_value for state with the DQN eval model
        q_current = self.dqn_eval.forward(batch_state)
        #print(q_pred.shape, batch_action.shape)
        q_pred_current = q_current.gather(1, batch_action)
        #print(q_pred)
        #print(batch_action)
        #print(q_values)

        # predict q_value for next_state with the DQN target model
        q_next = self.dqn_target.forward(batch_next_state)
        q_pred_next = torch.max(q_next, 1)[0]
        q_pred_next[batch_terminal] = 0.0
        #print(batch_reward.shape, target_pred.shape)
        #print(target_values.shape, non_final_mask.shape, q_values.shape)
        q_target = batch_reward + (self.gamma * q_pred_next)

        #print(target_values)
        loss = self._compute_loss(q_pred_current, q_target)
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy()

    def test(self):
        test_reward = 0
        terminal = False
        current_state = self.env.reset()

        while not terminal:
            state = np.expand_dims(current_state, 0)
            state_tensor = torch.FloatTensor(state).to(self.device)
            #print(state_tensor.shape)
            state_tensor = state_tensor.reshape(
                [
                    1,
                    self.num_frames_in_state,
                    self.output_shape[0],
                    self.output_shape[1],
                ]
            )

            q_pred = self.dqn_eval.forward(state_tensor)
            action = np.argmax(q_pred.data.cpu().numpy(), 1)
            next_state, test_reward, terminal = self.env.step(action[0])
            current_state = next_state

        return test_reward


def train_drl_agent_dqn(ARGS):
    env = CatchEnv()

    num_actions = env.get_num_actions()
    num_episodes = ARGS.num_episodes

    dir_model = os.path.join(ARGS.dir_model, f"{ARGS.which_model}_{ARGS.loss_function}_{ARGS.which_optimizer}_{ARGS.batch_size}")

    drl_agent_dqn = DRLAgentDQN(
        env,
        ARGS.gamma,
        ARGS.learning_rate,
        ARGS.exp_buffer_size,
        ARGS.batch_size,
        loss_function=ARGS.loss_function,
        which_model=ARGS.which_model,
        which_optimizer=ARGS.which_optimizer,
    )

    if not os.path.isdir(dir_model):
        os.makedirs(dir_model)
        print(f"created directory: {dir_model}")

    file_csv = os.path.join(dir_model, "train_logs.csv")
    csv_writer = CSVWriter(file_csv, ["episode", "mode", "test_success_rate"])
    is_train = True

    interval_test_success_rates = []
    count_test_wins = 0
    count_test_episodes = 0
    test_success_rate = 0.0

    for episode in range(1, num_episodes + 1):
        terminal = False
        current_state = env.reset()
        t_1 = time.time()
        if is_train:
            #-----------------------------#
            #        Train and Test       #
            #-----------------------------#

            #--------------------#
            #        Train       #
            #--------------------#
            # reset the environment
            # (1) state is initialized in current_state
            step_reward = 0

            while not terminal:
                drl_agent_dqn.step += 1
                # (2) for S_t take an action a_t with greedy policy
                action = drl_agent_dqn.get_action(current_state)
                #print(f"action: {action}")

                next_state, reward, terminal = env.step(action)

                # (3) store the states in the experience replay buffer
                dict_experience = {}
                dict_experience["state"] = current_state
                dict_experience["action"] = action
                dict_experience["next_state"] = next_state
                dict_experience["reward"] = reward
                dict_experience["terminal"] = terminal

                drl_agent_dqn.exp_replay_buffer.store_experience(dict_experience)

                # set next state to current state
                current_state = next_state

                # (4) train the model with the data in the experience replay buffer
                if drl_agent_dqn.exp_replay_buffer.is_exp_replay_available(ARGS.batch_size):
                    loss = drl_agent_dqn.learn()

                # (5) update the target model which is used to compute TD error
                if (drl_agent_dqn.step % ARGS.target_update_interval) == 0:
                    drl_agent_dqn.update_target_model()

            #---------------------#
            #         Test        #
            #---------------------#
            count_test_episodes += 1
            test_reward = drl_agent_dqn.test()
            if test_reward >= 1:
                count_test_wins += 1
            print(f"episode: {episode}, train, reward obtained by the agent: {test_reward}")
        else:
            #---------------------#
            #         Test        #
            #---------------------#
            count_test_episodes += 1
            test_reward = drl_agent_dqn.test()
            if test_reward >= 1:
                count_test_wins += 1
            print(f"episode: {episode}, test, reward obtained by the agent: {test_reward}")

        t_2 = time.time()

        if (episode % ARGS.test_interval) == 0:
            is_train = not(is_train)

        if (episode % ARGS.model_save_interval) == 0:
            drl_agent_dqn.save_model(os.path.join(dir_model, f"checkpoint_{episode}.pth"))

        if count_test_episodes == ARGS.test_interval:
            test_success_rate = count_test_wins / ARGS.test_interval
            interval_test_success_rates.append(test_success_rate)

        print("="*50)
        print(f"end of episode: {episode}, time: {(t_2 - t_1):.4f} sec., test success rate: {test_success_rate:.4f}")
        print("="*50)

        if is_train:
            mode = "train"
        else:
            mode = "test"
        csv_writer.write_row([episode, mode, test_success_rate])

        if count_test_episodes == ARGS.test_interval:
            count_test_wins = 0
            count_test_episodes = 0
            test_success_rate = 0.0

    interval_test_success_rates = np.array(interval_test_success_rates)
    np.save(os.path.join(dir_model, "test_success_rates.npy"), interval_test_success_rates)
    csv_writer.close()
    return


def main():
    gamma = 0.99
    batch_size = 32
    num_episodes = 2000
    learning_rate = 1e-4
    exp_buffer_size = 500000
    target_update_interval = 10
    test_interval = 10
    model_save_interval = 50

    loss_function = "mse"
    which_model = "dqn_simple_new"
    which_optimizer = "rms_prop"
    dir_model = "rl_models"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--gamma", default=gamma,
        type=float, help="gamma i.e. the discount factor")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="number of samples in a batch")
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate to be used")
    parser.add_argument("--num_episodes", default=num_episodes,
        type=int, help="number of episodes to train the DRL agent")
    parser.add_argument("--exp_buffer_size", default=exp_buffer_size,
        type=int, help="experience replay buffer size")
    parser.add_argument("--target_update_interval", default=target_update_interval,
        type=int, help="target model update interval")
    parser.add_argument("--test_interval", default=test_interval,
        type=int, help="model test interval")
    parser.add_argument("--model_save_interval", default=model_save_interval,
        type=int, help="model save interval")

    parser.add_argument("--dir_model", default=dir_model,
        type=str, help="directory where checkpoint needs to be saved")
    parser.add_argument("--loss_function", default=loss_function,
        type=str, choices=["huber", "mse", "smooth_l1"], help="loss function to be used for training")
    parser.add_argument("--which_model", default=which_model,
        type=str, choices=["dqn_simple", "dqn_simple_new", "dqn_residual"], help="which model to train")
    parser.add_argument("--which_optimizer", default=which_optimizer,
        type=str, choices=["rms_prop", "adam"], help="optimizer to be used for learning")

    ARGS, unparsed = parser.parse_known_args()
    train_drl_agent_dqn(ARGS)

    return


if __name__ == "__main__":
    main()
