"""
    in this program compare different learning rates to investigate what effect
    does learning rate have to the convergence of an optimizer.
    
    the simulation is done under exact model parameter
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from src.optimizers.adagrad import AdaGrad
from src.optimizers.adam import Adam
from src.optimizers.momentum import Momentum
from src.optimizers.rmsprop import RMSProp

#create syntheic data
data = []
for i in range(100):
    x = random.uniform(-10,10)
    y = 3 * x
    data.append((x, y))


#model parameters
batch_size = 100
iterations = 5000


def train(optimizer):
    #declare variables to return
    theta = random.uniform(-5,5)
    theta_history = []
    loss_history = []
        
    for k in range(iterations):
        loss_sum = 0
        grad_sum = 0
        #sample full batch
        batch = random.sample(data, batch_size)
        for x, y in batch:
            y_hat = theta * x #model

            loss = (y_hat - y) ** 2 #compute loss 
            loss_sum += loss  #sum loss

            grad = 2 * x * (y_hat - y) #compute gradient
            grad_sum += grad #sum gradient
        
        grad_avg = grad_sum / batch_size #computer avareage gradient
        theta = optimizer.update(theta, grad_avg) #computer theta update

        loss_avg = loss_sum / batch_size #computer loss average


        theta_history.append(theta)
        loss_history.append(loss_avg)

        

    return theta, theta_history, loss_history




# experiments/learning_rate_study.py

import json

results = {}
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]

for lr in learning_rates:

    optimizer = Adam(lr=lr)
    theta, theta_history, loss_history = train(optimizer)

    results[lr] = {
        "theta_history": theta_history,
        "loss_history": loss_history
    }

#store data into a json file
with open("results/learning_rate_results.json", "w") as f:
    json.dump(results, f)
