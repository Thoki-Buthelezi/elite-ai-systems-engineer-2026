class SGD:

    #initialise the learning rate
    def __init__(self, lr):
        self.lr = lr
    
    #compute stochastic gradient descent update
    '''
       theta is the parameter we want to optimize
       grad is the unbiased estimate of the gradient
    '''
    def update(self, theta, grad):
        #apply the update on the parameter theta
        theta = theta - self.lr * grad
        return theta
