"""
    This represent an Optimization Algorithm for any 2D objective function
    It allows the user to provide:
    1. A function to optimise
    2. Two optimizer to use for optimization (one for each parameter in the 2D parameter space)

    In this particular example we will be optimmising the Rastrigin Function using several optimizers including:
    1.AdaGrad
    2.Momentum
    3.RMSProp
    4.Adam
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import math
import random

from src.optimizers.adagrad import AdaGrad
from src.optimizers.momentum import Momentum
from src.optimizers.rmsprop import RMSProp
from src.optimizers.adam import Adam
from src.functions.rastrigin import Rastrigin


iterations = 10000

def train(optimizer1, optimizer2):
    t1 = random.uniform(-5, 5) #theta 1
    t2 = random.uniform(-5, 5) #theta 2

    f = Rastrigin(t1, t2) #function to optimise

    t1_history = [] #history of theta 1
    t2_history = [] #history of theta 2

    for k in range(iterations):
        #compute the gradient
        dx, dy = f.gradient()

        #update parameters
        t1 = optimizer1.update(t1, dx)
        t2 = optimizer2.update(t2, dy)

        #update function to correspond to the new values of parameter
        f = Rastrigin(t1, t2)

        t1_history.append(t1)
        t2_history.append(t2)
        
        if k % 100 == 0:
            print(f"iter:{k} | t1:{t1} | t2:{t2} | value:{f.evaluate()}")

    
    print(f"final value:{f.evaluate()} | t1:{t1}  | t2:{t2}")
    return t1_history, t2_history

import json

results = {}
optimizers = {
    "Adam" : (Adam(), Adam()),
    "AdaGrad" : (AdaGrad(), AdaGrad()),
    "Momentum" : (Momentum(), Momentum()),
    "RMSProp" : (RMSProp(), RMSProp())
}

for name, opt in optimizers.items():
    t1_history, t2_history = train(opt[0], opt[1])

    results[name] = {
        "theta1" : t1_history,
        "theta2" : t2_history
    }

with open("results/rastrigin_results.json", "w") as file:
    json.dump(results, file)


    