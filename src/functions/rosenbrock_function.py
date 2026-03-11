import math

class Rosenbrock:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def evaluate(self):
        return math.pow(1-self.x, 2) + 100 * math.pow(self.y - self.x**2, 2)
    
    def gradient(self):
        dx = -2 * (1-self.x) - 400 * self.x * (self.y - self.x ** 2)
        dy = 200 * (self.y - self.x ** 2)
        return dx, dy
    