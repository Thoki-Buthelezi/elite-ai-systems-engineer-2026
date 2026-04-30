import math
class AdaGrad:

    #initialise optimizer paramters
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps #small constant for numerically stability
        self.grad_sum = 0 #accumulation of the squared historical gradients

    #compute parameter update
    def update(self, theta, grad):
        #update gradient
        self.grad_sum += math.pow(grad, 2)
        #adapt learning rate
        delta_theta = -((self.lr * grad )/ (self.eps + math.sqrt(self.grad_sum))) #adapt lr by scaling with inverse of the sqrt of the sum of the squared gradients
        #compute parameter update
        theta = theta + delta_theta
        return theta