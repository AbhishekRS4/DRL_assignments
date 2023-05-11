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


class RLAgent(object):
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
        self.model = None
        self.gamma = gamma
        self.criterion = None
        self.optimizer = None
        self.target_model = None
        self.is_apply_clip = True
        self.batch_size = batch_size
        self.num_frames_in_state = 4
        self.output_shape = (84, 84)
        self.num_actions = self.env.get_num_actions()
        self.learning_rate = learning_rate
        self.exp_replay_buffer = deque(maxlen=exp_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1000000

        self.Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

        if which_model.lower() == "dqn_simple":
            self.model = DQNSimple(self.num_actions)
            self.target_model = DQNSimple(self.num_actions)
        elif which_model.lower() == "dqn_simple_new":
            self.model = DQNSimpleNew(self.num_actions)
            self.target_model = DQNSimpleNew(self.num_actions)
        elif which_model.lower() == "dqn_residual":
            self.model = DQNResidual(self.num_actions)
            self.target_model = DQNResidual(self.num_actions)
        else:
            print(f"unidentified option for (which_model={which_model}), should be one of ['dqn_simple', 'dqn_residual']")

        self.model.to(self.device)
        self.target_model.to(self.device)

        if which_optimizer.lower() == "rms_prop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-2)
        elif which_optimizer.lower() == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            print(f"unidentified option for (which_optimizer={which_optimizer}), should be one of ['RMSprop', 'Adam']")

        if loss_function.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif loss_function.lower() == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        elif loss_function.lower() == "huber":
            self.criterion = nn.HuberLoss()
        else:
            print(f"unidentified option for (loss_function={loss_function}), should be one of ['mse', 'smooth_l1', 'huber']")

    def is_exp_replay_available(self,):
        flag = None
        if len(self.exp_replay_buffer) >= self.batch_size:
            flag = True
        else:
            flag = False
        return flag

    def store_experience(self, dict_experience):
        dict_experience = NestedAttrDict(**dict_experience)
        next_state = None
        if dict_experience.next_state is not None:
            next_state = torch.FloatTensor(dict_experience.next_state)

        transition = self.Transition(
            state=torch.FloatTensor(dict_experience.state),
            action=torch.LongTensor([[dict_experience.action]]),
            reward=torch.FloatTensor([dict_experience.reward]),
            next_state=next_state,
        )
        self.exp_replay_buffer.append(transition)
        return

    def sample_experiences(self,):
        experiences = sample(self.exp_replay_buffer, self.batch_size)
        experiences = self.Transition(*(zip(*experiences)))
        return experiences

    def get_action(self, state):
        sample_action = None
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                     math.exp(-1. * self.step / self.epsilon_decay)

        if random() <= self.epsilon:
            sample_action = randint(0, self.num_actions - 1)
        else:
            state = np.expand_dims(state, 0)
            state_var = Variable(torch.FloatTensor(state).cuda())
            state_var = state_var.reshape(
                [
                    -1,
                    self.num_frames_in_state,
                    self.output_shape[0],
                    self.output_shape[1],
                ]
            )
            #print(state_var.shape)
            state_var.volatile = True
            q_values = self.model(state_var)
            q_values = q_values.data.cpu().numpy()
            sample_action = np.argmax(q_values, 1)[0]
            #print(f"sample action : {sample_action}")
        return sample_action

    def update_target_model(self,):
        self.target_model = deepcopy(self.model)
        return

    def save_model(self, file_checkpoint):
        dict_checkpoint = {
            "dqn": self.model.state_dict(),
            "target": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(dict_checkpoint, file_checkpoint)
        return

    def compute_loss(self, q_values, target_values):
        loss = self.criterion(q_values, target_values)
        return loss

    def optimize(self):
        # sample experiences from the experience replay buffer
        state_transitions = self.sample_experiences()

        # create mask tensors
        non_final_mask = torch.ByteTensor(list(map(lambda ns: ns is not None, state_transitions.next_state))).cuda()
        final_mask = 1 - non_final_mask

        batch_state = Variable(torch.cat(state_transitions.state).cuda())
        batch_action = Variable(torch.cat(state_transitions.action).cuda())
        batch_reward = Variable(torch.cat(state_transitions.reward).cuda())
        #batch_reward = torch.unsqueeze(batch_reward, 1)
        batch_non_final_next_state = Variable(torch.cat([ns for ns in state_transitions.next_state if ns is not None]).cuda())
        batch_non_final_next_state.volatile = True

        # reshape state and next_state
        batch_state = batch_state.view(
            [
                self.batch_size,
                self.num_frames_in_state,
                self.output_shape[0],
                self.output_shape[1],
            ]
        )
        batch_non_final_next_state = batch_non_final_next_state.view(
            [
                -1,
                self.num_frames_in_state,
                self.output_shape[0],
                self.output_shape[1],
            ]
        )
        batch_non_final_next_state.volatile = True

        # apply reward clipping between -1 and 1
        batch_reward.data.clamp_(-1, 1)

        # predict q_value with the DQN Model
        q_pred = self.model(batch_state)
        #print(q_pred.shape, batch_action.shape)
        q_values = q_pred.gather(1, batch_action)

        # predict with the target model
        target_values = Variable(torch.zeros(self.batch_size).cuda())
        target_pred = self.target_model(batch_non_final_next_state)
        #print(batch_reward.shape, target_pred.shape)
        #print(target_values.shape, non_final_mask.shape, q_values.shape)
        target_values[non_final_mask] = batch_reward[non_final_mask] + (torch.max(target_pred, 1)[0] * self.gamma)
        target_values[final_mask] = batch_reward[final_mask].detach()

        loss = self.compute_loss(q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()

        if self.is_apply_clip:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        reward_score = int(torch.sum(batch_reward).data.cpu().numpy())
        q_mean = torch.sum(q_pred, 0).data.cpu().numpy()[0]
        target_mean = torch.sum(target_pred, 0).data.cpu().numpy()[0]

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean

    def test(self):
        self.env.reset()
        current_state, reward, terminal = self.env.step(1)

        reward = 0

        while not terminal:
            state = np.expand_dims(current_state, 0)
            state_var = Variable(torch.FloatTensor(state).cuda())
            #print(state_var.shape)
            state_var = state_var.reshape(
                [
                    1,
                    self.num_frames_in_state,
                    self.output_shape[0],
                    self.output_shape[1],
                ]
            )

            q_pred = self.model(state_var)
            action = np.argmax(q_pred.data.cpu().numpy(), 1)
            next_state, reward, terminal = self.env.step(action[0])
            current_state = next_state

        return reward


def train_drl_agent(ARGS):
    env = CatchEnv()

    num_actions = env.get_num_actions()
    num_episodes = ARGS.num_episodes

    drl_agent = RLAgent(
        env,
        ARGS.gamma,
        ARGS.learning_rate,
        ARGS.exp_buffer_size,
        ARGS.batch_size,
        loss_function=ARGS.loss_function,
        which_model=ARGS.which_model,
        which_optimizer=ARGS.which_optimizer,
    )

    if not os.path.isdir(ARGS.dir_model+"_"+ARGS.loss_function):
        os.makedirs(ARGS.dir_model+"_"+ARGS.loss_function)
        print(f"created directory: {ARGS.dir_model+'_'+ARGS.loss_function}")

    file_csv = os.path.join(ARGS.dir_model+"_"+ARGS.loss_function, "train_logs.csv")
    csv_writer = CSVWriter(file_csv, ["episode", "mode", "test_success_rate"])
    is_train = True

    interval_test_success_rates = []
    count_test_wins = 0
    count_test_episodes = 0
    test_success_rate = 0.0

    for episode in range(1, num_episodes + 1):
        t_1 = time.time()

        if is_train:
            #-----------------------------#
            #        Train and Test       #
            #-----------------------------#

            #--------------------#
            #        Train       #
            #--------------------#
            # reset the environment
            env.reset()

            # (1) state is initialized in current_state
            current_state, reward, terminal = env.step(1)
            step_reward = 0

            while not terminal:
                drl_agent.step += 1
                # (2) for S_t take an action a_t with greedy policy
                action = drl_agent.get_action(current_state)
                #print(f"action: {action}")

                next_state, reward, terminal = env.step(action)

                # (3) store the states in the experience replay buffer
                dict_experience = {}
                dict_experience["state"] = current_state
                dict_experience["action"] = action
                dict_experience["reward"] = reward

                if not terminal:
                    dict_experience["next_state"] = next_state
                else:
                    dict_experience["next_state"] = None

                drl_agent.store_experience(dict_experience)

                # (4) train the model with the data in the experience replay buffer
                if drl_agent.is_exp_replay_available():
                    loss, step_reward, _, _ = drl_agent.optimize()

                # (5) update the target model which is used to compute TD error
                if (drl_agent.step % ARGS.target_update_interval) == 0:
                    drl_agent.update_target_model()

            #---------------------#
            #         Test        #
            #---------------------#
            count_test_episodes += 1
            test_reward = drl_agent.test()
            if test_reward >= 1:
                count_test_wins += 1
            print(f"episode: {episode}, train, reward obtained by the agent: {step_reward}")
        else:
            #---------------------#
            #         Test        #
            #---------------------#
            count_test_episodes += 1
            test_reward = drl_agent.test()
            if test_reward >= 1:
                count_test_wins += 1
            print(f"episode: {episode}, test, reward obtained by the agent: {test_reward}")

        t_2 = time.time()

        if (episode % ARGS.test_interval) == 0:
            is_train = not(is_train)

        if (episode % ARGS.model_save_interval) == 0:
            drl_agent.save_model(os.path.join(ARGS.dir_model+"_"+ARGS.loss_function, f"checkpoint_{episode}.pth"))

        if count_test_episodes == ARGS.test_interval:
            test_success_rate = count_test_wins / ARGS.test_interval
            interval_test_success_rates.append(test_success_rate)

        print("="*50)
        print(f"end of episode: {episode}, time: {(t_2 - t_1):.4f} sec., test success rate: {test_success_rate:.4f}")
        print("="*50)

        if count_test_episodes == ARGS.test_interval:
            count_test_wins = 0
            count_test_episodes = 0
            test_success_rate = 0.0

        if is_train:
            mode = "train"
        else:
            mode = "test"
        csv_writer.write_row([episode, mode, test_success_rate])

    interval_test_success_rates = np.array(interval_test_success_rates)
    np.save(os.path.join(ARGS.dir_model+"_"+ARGS.loss_function, "test_success_rates.npy"), interval_test_success_rates)
    csv_writer.close()
    return


def main():
    gamma = 0.99
    batch_size = 32
    num_episodes = 2000
    learning_rate = 1e-3
    exp_buffer_size = 50000
    target_update_interval = 10
    test_interval = 10
    model_save_interval = 50

    loss_function = "huber"
    which_model = "dqn_simple_new"
    which_optimizer = "rms_prop"
    dir_model = "dqn_simple_new"

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
        type=str, choices=["dqn_simple", "dqn_residual"], help="which model to train")
    parser.add_argument("--which_optimizer", default=which_optimizer,
        type=str, choices=["rms_prop", "adam"], help="optimizer to be used for learning")

    ARGS, unparsed = parser.parse_known_args()
    train_drl_agent(ARGS)

    return


if __name__ == "__main__":
    main()
