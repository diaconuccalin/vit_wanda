import os

import matplotlib.pyplot as plt
import torch


def main():
    if not os.path.exists("plots"):
        os.mkdir("plots")

    for el in os.listdir("maps"):
        testing = torch.load("maps/" + el)

        plt.clf()
        plt.hist(testing.numpy(), log=True, bins=128)
        plt.savefig("plots/" + el + ".png")


if __name__ == "__main__":
    main()
