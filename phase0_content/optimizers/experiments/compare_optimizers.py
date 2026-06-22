import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from optimizers import adagrad
from optimizers import adam
from optimizers import rmsprop
from optimizers import momentum

"""
    In the following program, I compare several optimizers to see which one 
"""

# create synthetic data
data = []
for i in range(100):
    x = random.uniform(-10, 10)
    y = 3 * x
    data.append((x,y))

#define model paramters

batch_size = 20
iterations = 5000



#Compare optimizers on simple regression
#Goal: recover theta ≈ 3 where y = 3x
def train(optimizer):
    theta = random.uniform(-5,5)
    for k in range(iterations):
        batch = random.sample(data, batch_size)
        grad_sum = 0
        for x, y in batch:
            y_hat = theta * x
            grad_loss = 2 * x * (y_hat - y)
            grad_sum += grad_loss

        grad = grad_sum / batch_size
        theta = optimizer.update(theta, grad)

    return theta

#compare resutls
print("Adam:", train(adam.Adam()))
print("AdaGrad:", train(adagrad.AdaGrad()))
print("Momentum:", train(momentum.Momentum()))
print("RMSProp:", train(rmsprop.RMSProp()))

