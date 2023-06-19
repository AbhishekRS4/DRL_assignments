import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(ARGS):
    arr_logs = pd.read_csv(ARGS.file_csv)

    fig = plt.figure()
    plt.plot(arr_logs.Step, arr_logs.Value)
    plt.title("training reward vs step", fontsize=20)
    plt.xlabel("step", fontsize=20)
    plt.ylabel("mean episodic training reward", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return

def main():
    file_csv = "../../run-PPO_1-tag-rollout_ep_rew_mean.csv"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to numpy file with test success rates")

    ARGS, unparsed = parser.parse_known_args()
    plot_graph(ARGS)
    return

if __name__ == "__main__":
    main()
