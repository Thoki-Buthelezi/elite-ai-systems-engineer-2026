import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

#create random 2 1D vectors
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)

#define rastrigin function
def rastrigin(x, y):
    return 20 + x ** 2 + y ** 2 -10 * (np.cos(2 * x) + np.cos(2 * y))

#create a 2D matrix of values
X, Y = np.meshgrid(x, y)
#apply element-wise function to the values
Z = rastrigin(X, Y)

#configuration of the figure size
plt.figure(figsize=(8, 6))

#add contour lines of the functions
plt.contour(X, Y, Z, levels=50, cmap="viridis")

#import data
import json
with open("results/rastrigin_results.json", "r") as file:
    results = json.load(file)

#plot trajectory for each optimizer
for name, data in results.items():
    theta1_history = data["theta1"]
    theta2_history = data["theta2"]

    plt.plot(theta1_history, theta2_history, marker="o",  markersize=2, label=name)


#plot the global minimum
plt.scatter([0], [0], s=100, color="black", label="Global Minimum")

#add axis to the graph
plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimizer Trajectories on Rastrigin Function")
plt.legend()

plt.savefig("plots/optimizer_rastrigin.png")

plt.show()


