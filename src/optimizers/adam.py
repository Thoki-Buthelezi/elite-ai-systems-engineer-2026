import math

class Adam:

    #initialise optimizer parameters
    def __init__(self, lr=0.001, p1=0.9, p2=0.999, eps=1e-8):
        self.lr = lr
        self.p1 = p1 # decay rate for the 1st order moment
        self.p2 = p2 # decay rate for the 2nd order moment
        self.eps = eps
        self.t = 0 # time step
        self.m1 = 0 # 1st order moment variable
        self.m2 = 0 # 2nd order moment variable
    

    #compute parameter update
    def update(self, theta, grad):
        self.t += 1
        # update biased 1st order moment estimate
        self.m1 = self.p1 * self.m1 + (1 - self.p1) * grad
        # update biased 2nd order moment estimate
        self.m2 = self.p2 * self.m2 + (1 - self.p2) * (grad ** 2)
        # correct bias in 1st order moment
        m1_hat = self.m1 / (1 -  self.p1 ** self.t)
        # correct bias in the 2nd order momemnt
        m2_hat = self.m2 / (1 - self.p2 ** self.t)
        #compute parameter update
        delta_theta = -self.lr * (m1_hat / (math.sqrt(m2_hat) + self.eps))
        #apply update
        theta = theta + delta_theta
        return theta
