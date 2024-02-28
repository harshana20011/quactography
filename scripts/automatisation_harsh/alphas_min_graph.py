import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"alpha_min_cost.txt", header=None, names=["alpha", "cost", "path"])

alpha = df["alpha"]
cost = df["cost"]
plt.plot(alpha, cost, ".-")
plt.xlabel("Alpha")
plt.ylabel("Cost")
plt.title("f(x) en fonction de alpha")

plt.show()
