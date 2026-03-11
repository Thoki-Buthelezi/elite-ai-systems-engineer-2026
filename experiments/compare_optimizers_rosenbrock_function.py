import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
import math
from src.optimizers.adagrad import AdaGrad
from src.optimizers.adam import Adam
from src.optimizers.momentum import Momentum
from src.optimizers.rmsprop import RMSProp
from src.functions.rosenbrock_function import Rosenbrock

iterations = 5000 #iterations number
eps = 1e-6 #small constant for stopping condtion

#make use of two optimizers so each parameter has its own optimizer state
def train(optimizer1, optimizer2):
    t1 = random.uniform(-3,3) #theta 1
    t2 = random.uniform(-3, 3) #theta 2
    theta1_history = [] #keep track of previous theta1
    theta2_history = [] #keep track of previious theta2
    z_hat_history = [] #keep track of the evaluated value

    function = Rosenbrock(t1, t2) #rosenbrock function to evaluate

    #start iteration
    for k in range(iterations):
        #compute gradient
        dx, dy = function.gradient()
        ##compute gradient norm
        grad_norm = math.sqrt(dx ** 2 + dy ** 2)

        #stopping condition
        if grad_norm < eps:
            break

        #compute parameter update
        t1 = optimizer1.update(t1, dx)
        t2 = optimizer2.update(t2, dy)
        #update function
        function = Rosenbrock(t1,t2)

        theta1_history.append(t1)
        theta2_history.append(t2)
        z_hat_history.append(function.evaluate())
    print(f"z={Rosenbrock(t1,t2).evaluate()} | x`={t1} | y`={t2}")
    return theta1_history, theta2_history, z_hat_history



print("adam: ", train(Adam(), Adam()))
print("adaGrad: ", train(AdaGrad(), AdaGrad()))
print("momentum: ", train(Momentum(), Momentum()))
print("rmsprop", train(RMSProp(), RMSProp()))





#implement results to a file
import json

results = {}

optimizers = {
    "Adam": (Adam(), Adam()),
    "AdaGrad": (AdaGrad(), AdaGrad()),
    "Momentum": (Momentum(), Momentum()),
    "RMSProp": (RMSProp(), RMSProp())
}

for name, opt in optimizers.items():
    t1_hist, t2_hist, z_hist = train(opt[0], opt[1])

    results[name] = {
        "t1": t1_hist,
        "t2": t2_hist,
        "z": z_hist
    }

with open("results/rosenbrock_results.json", "w") as file:
    json.dump(results, file)







