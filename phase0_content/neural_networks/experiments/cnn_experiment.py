import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.neural_networks.autograd_engine import Value

'''
   tiny CNN from scracth using an autograd engine also built from scratch
   implementing 1 conv layer + tanh activation function
   visualize what filter learns and the behaviour of the loss

   input -> 5 * 5
   filter -> 3 * 3
   output -> 3 * 3

   no padding
   stride = 1
'''

import random

#input data
input = [
    [Value(random.uniform(-3,3)) for _ in range(5)]
    for _ in range(5)
]

#filter(kernel)
filter = [
    [Value(random.uniform(-1,1)) for _ in range(3)]
    for _ in range(3)
]

#predicted output, initialise to zero
output = [
    [Value(0) for _ in range(3)]
    for _ in range(3)
]

#actual target
target = [
    [Value(1),Value(0),Value(1)],
    [Value(0),Value(1),Value(0)],
    [Value(0),Value(0),Value(1)]
]

#training loop 
for epoch in range(2000):

    #foward convolutional pass
    for i in range(3):
        for j in range(3):
            sum = Value(0) #convolutional output sum before activation (start at 0)
            for fi in range(3):
                for fj in range(3):
                    sum += input[i + fi][j + fj] * filter[fi][fj] #dot product
            output[i][j] = sum # preactivation
            output[i][j] = output[i][j].tanh() #post activation value
    
    #compute loss using MSE
    loss = Value(0)
    for i in range(3):
        for j in range(3):
            loss += (output[i][j] - target[i][j]) ** 2
    
    #zero gradient for the filter
    for row in filter:
        for value in row:
            value.grad = 0

    #begin backpropagation
    loss.backward()

    #update filter parameters with lr=0.01
    for row in filter:
        for value in row:
            value.data -= 0.01 * value.grad
            # print(value.grad)

    #print loss
    if epoch % 100 == 0:
        print(f"loss:{loss.data}")

#see predicted values

for i in range(3):
    for j in range(3):
        print(f"y_hat={output[i][j].data} y={target[i][j].data}")