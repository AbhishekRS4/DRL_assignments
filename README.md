# Deep Reinforcement Learning assignments

## Info
* Code for the assignments for the [Deep Reinforcement Learning course](https://ocasys.rug.nl/current/catalog/course/WMAI024-05) offered at University of Groningen

## Instructions to run code

### Value based DRL
* Deep Q-Network (DQN) has been implemented from scratch
* Some code is inspired from [this repo](https://github.com/AndersonJo/dqn-pytorch)
* Run [value_based_drl/train_dqn_agent.py](value_based_drl/train_dqn_agent.py) for training
* Run [value_based_drl/plot_success_rates.py](value_based_drl/plot_success_rates.py) to plot the success rate graph which is nothing but the average of the rewards of `k` test episodes

### Policy based DRL
* Proximal Policy Optimization (PPO) has been implemented using `StableBaselines3`
* Run [policy_based_drl/train_ppo_agent.py](policy_based_drl/train_ppo_agent.py) for training
* Run [policy_based_drl/test_ppo_agent.py](policy_based_drl/test_ppo_agent.py) for testing
* Download csv file from the tensorboard logs and then run [policy_based_drl/plot_graph.py](policy_based_drl/plot_graph.py)

## Package dependencies
* Can be found in [requirements.txt](requirements.txt)
