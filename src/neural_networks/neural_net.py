#import and activations
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    #assume x = Z (pre activation)
    x = sigmoid(x)
    return 1 * (1 - x)


#Model class
class NeuralNet:

    #constructor
    def __init__(self):
        #initialise parameters
        self.W1 = np.random.randn(2, 2) * 0.01
        self.b1 = np.zeros((2, 1))

        self.W2 = np.random.randn(1, 2) * 0.01
        self.b2 = np.zeros((1,1))

    #define foward pass
    def foward(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = sigmoid(self.Z1)

        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2
    
    #compute loss function
    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        return (1/m) * np.sum( 0.5 * (A2 - Y) **2)

    #backpropagatoin
    def backwards(self, X, Y):
        m = X.shape[1]

        delta2 = (self.A2 - Y) * sigmoid_derivative(self.Z2)
        self.dW2 = (1/m) * (delta2 @ self.A1.T)
        self.db2 = (1/m) * (np.sum(delta2, axis=1, keepdims=True))

        delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.Z1)
        self.dW1 = (1/m) * (delta1 @ X.T)
        self.db1 = (1/m) * (np.sum(delta1, axis=1, keepdims=True))

    
    #update model parameter
    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.W2 -= lr * self.dW2
        self.b1 -= lr * self.db1
        self.b2 -= lr * self.db2
    
    #training loop
    def train(self, X,  Y, epochs, lr):
        for i in range(epochs):
            A2 = self.foward(X)
            loss = self.compute_loss(A2, Y)
            self.backwards(X, Y)
            self.update(lr)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")

#training loop example
if __name__ == "__main__":
    X = np.array([
        [1,2,3],
        [4,5,6]
    ])

    Y = np.array([[1,0,1]])

    model = NeuralNet()
    model.train(X, Y, epochs=1000, lr=0.01)


        
