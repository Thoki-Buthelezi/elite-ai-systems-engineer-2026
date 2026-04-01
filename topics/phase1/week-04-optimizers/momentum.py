class Momentum:

    #initialise optimizer parameters
    def __init__(self, lr = 0.01, alpha=0.09):
        self.lr = lr
        self.alpha = alpha 
        self.v = 0 #velocity

    
    #compute parameter update
    def update(self, theta, grad):
        #update velocity
        self.v = (self.alpha * self.v) - (self.lr * grad)
        #update parameter 
        theta = theta + self.v
        return theta
