import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_success_rates(ARGS):
    arr_success_rates = np.load(ARGS.file_npy)

    fig = plt.figure()
    plt.plot(np.arange(1, len(arr_success_rates) + 1), arr_success_rates)
    plt.title("test success rates", fontsize=20)
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("success score", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return

def main():
    file_npy = "dqn_simple/test_success_rates.npy"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_npy", default=file_npy,
        type=str, help="full path to numpy file with test success rates")

    ARGS, unparsed = parser.parse_known_args()
    plot_success_rates(ARGS)
    return

if __name__ == "__main__":
    main()
