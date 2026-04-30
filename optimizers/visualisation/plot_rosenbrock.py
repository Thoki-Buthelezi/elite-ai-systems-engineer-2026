import numpy as np
import matplotlib.pyplot as plt
import json


# create grid
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)

X, Y = np.meshgrid(x, y)
Z = Z = (1 - X)**2 + 100 * (Y - X**2)**2

plt.figure(figsize=(8,6))

# contour plot
plt.contour(X, Y, Z, levels=50, cmap="viridis")

#import data
with open("optimizers/results/rosenbrock_results.json", "r") as file:
    results = json.load(file)


# plot optimizer paths
for name, data in results.items():

    t1 = data["t1"]
    t2 = data["t2"]

    plt.plot(t1, t2, marker='o', markersize=2, label=name)

plt.scatter([1], [1], color='black', s=100, label="True Minimum")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimizer Trajectories on Rosenbrock Function")
plt.legend()

plt.savefig("optimizers/plots/optimizer_rosenbrock.png")

plt.show()