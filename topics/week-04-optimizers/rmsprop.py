import math

class RMSProp:

    #initialise optimizer parameters
    def __init__(self, lr=0.001, psi=0.9, eps=1e-6):
        self.lr = lr
        self.psi = psi #decay rate
        self.eps = eps
        self.r = 0.0  #accumulation variable
    
    #compute parameter update
    def update(self, theta, grad):
        #accumulate squared gradient
        self.r = (self.psi * self.r) + (1 - self.psi) * (grad ** 2)
        #compute delta theta
        delta_theta = -(self.lr / math.sqrt(self.eps + self.r)) * grad
        #update parameter
        theta = theta + delta_theta
        return theta
