import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn

from copy import deepcopy
from random import randint, sample
from torch.autograd import Variable
from torch.optim import RMSprop, Adam
from collections import deque, namedtuple
from pyats.datastructures import NestedAttrDict

from catch import CatchEnv
from dqn import DQNSimple, DQNResidual
from logger_utils import CSVWriter, write_dict_to_json


class RLAgent(object):
    def __init__(self, gamma,
                 num_actions,
                 learning_rate,
                 exp_buffer_size,
                 batch_size=32,
                 loss_function="huber",
                 which_model="dqn_simple",
                 which_optimizer="rms_prop",
        ):
        self.model = None
        self.gamma = gamma
        self.criterion = None
        self.optimizer = None
        self.target_model = None
        self.is_apply_clip = True
        self.min_experiences = 100
        self.batch_size = batch_size
        self.num_frames_in_state = 4
        self.output_shape = (84, 84)
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.exp_replay_buffer = deque(maxlen=exp_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

        if which_model.lower() == "dqn_simple":
            self.model = DQNSimple(self.num_actions)
            self.target_model = DQNSimple(self.num_actions)
        elif which_model.lower() == "dqn_residual":
            self.model = DQNResidual(self.num_actions)
            self.target_model = DQNResidual(self.num_actions)
        else:
            print(f"unidentified option for (which_model={which_model}), should be one of ['dqn_simple', 'dqn_residual']")

        self.model.to(self.device)
        self.target_model.to(self.device)

        if which_optimizer.lower() == "rms_prop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate, eps=1e-2)
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
        if len(self.exp_replay_buffer) >= self.min_experiences:
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
        target_values[non_final_mask] = batch_reward[non_final_mask] + target_pred.max(1)[0] * self.gamma
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


def train_drl_agent(ARGS):
    env = CatchEnv()

    num_actions = env.get_num_actions()
    num_episodes = ARGS.num_episodes

    drl_agent = RLAgent(
        ARGS.gamma,
        num_actions,
        ARGS.learning_rate,
        ARGS.exp_buffer_size,
        ARGS.batch_size,
        loss_function=ARGS.loss_function,
        which_model=ARGS.which_model,
        which_optimizer=ARGS.which_optimizer,
    )

    if not os.path.isdir(ARGS.dir_model):
        os.makedirs(ARGS.dir_model)
        print(f"created directory: {ARGS.dir_model}")

    file_csv = os.path.join(ARGS.dir_model, "train_logs.csv")
    csv_writer = CSVWriter(file_csv, ["episode", "loss", "reward"])

    sum_of_rewards = 0.0
    count_steps = 0
    for episode in range(1, num_episodes + 1):
        t_1 = time.time()
        # reset the environment
        env.reset()

        arr_losses = np.array([])

        # get the current state
        current_state, reward, terminal = env.step(1)

        while not terminal:
            random_action = randint(0, 2)
            next_state, reward, terminal = env.step(random_action)
            count_steps += 1

            dict_experience = {}
            dict_experience["state"] = current_state
            dict_experience["action"] = random_action
            dict_experience["reward"] = reward

            if not terminal:
                dict_experience["next_state"] = next_state
            else:
                dict_experience["next_state"] = None

            drl_agent.store_experience(dict_experience)

            if drl_agent.is_exp_replay_available():
                loss, sum_of_rewards, _, _ = drl_agent.optimize()
                arr_losses = np.append(arr_losses, loss)

            reward = np.clip(reward, -1., 1.)

            if (count_steps % ARGS.target_update_interval) == 0:
                drl_agent.update_target_model()

            print(f"episode: {episode}, reward obtained by the agent: {reward}")

        if (episode % 100) == 0:
            drl_agent.save_model(os.path.join(ARGS.dir_model, f"checkpoint_{episode}.pth"))

        if len(arr_losses) > 0:
            mean_loss = np.mean(arr_losses)
        else:
            mean_loss = 0.0

        t_2 = time.time()
        print("="*20)
        print(f"end of episode: {episode}, time: {t_2 - t_1} sec., mean loss: {mean_loss:.4f}, sum of rewards: {sum_of_rewards:.4f}")
        print("="*20)
        csv_writer.write_row([episode, mean_loss, sum_of_rewards])

    csv_writer.close()
    return


def main():
    gamma = 0.99
    batch_size = 32
    num_episodes = 5000
    learning_rate = 5e-4
    exp_buffer_size = 50000
    target_update_interval = 2000

    loss_function = "huber"
    which_model = "dqn_simple"
    which_optimizer = "rms_prop"
    dir_model = "dqn_simple"

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
