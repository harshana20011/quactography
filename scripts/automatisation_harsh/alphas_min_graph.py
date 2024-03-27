""" This script is used to plot the cost function in function of alpha."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def plot_alpha_cost():
    df = pd.read_csv(
        r"output\alpha_min_cost_classical_read_leftq0.txt",
        header=None,
        names=["alpha", "cost", "path"],
    )

    alpha = df["alpha"]
    cost = df["cost"]
    plt.plot(alpha, cost, ".-")
    plt.xlabel("Alpha")
    plt.ylabel("Cost")
    plt.title("Cost in function of alpha")
    plt.savefig("output/alpha_min_cost.png")

    # plt.show()
