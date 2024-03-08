"""Tracer la courbe du coût minimal trouvé en fonction de alpha. 
Les données sont lues à partir du fichier alpha_min_cost.txt."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_alpha_cost():
    df = pd.read_csv(
        r"output\alpha_min_cost.txt", header=None, names=["alpha", "cost", "path"]
    )

    alpha = df["alpha"]
    cost = df["cost"]
    plt.plot(alpha, cost, ".-")
    plt.xlabel("Alpha")
    plt.ylabel("Cost")
    plt.title("f(x) en fonction de alpha")
    plt.savefig("output/alpha_min_cost.png")

    # plt.show()
