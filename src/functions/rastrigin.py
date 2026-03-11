import math

class Rastrigin:
    def __init__(self, x, y):
        self.x =  x
        self.y = y

    
    def evaluate(self):
        return  20 + math.pow(self.x, 2) + math.pow(self.y, 2) - 10 * (math.cos(2 * self.x) + math.cos(2 * self.y))


    def gradient(self):
        dx = 2 * self.x + 20 * (math.cos(2 * self.x) * math.sin(2 * self.x))
        dy = 2 * self.y + 20 * (math.cos(2 * self.y) * math.sin(2 * self.y))
        return dx, dy