import random

from autograd_engine import Value

class Neuron:
    def __init__(self, numInputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(numInputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = Value(0)
        for wi, xi in zip(self.w, x):
            act +=  wi * xi
        act = act + self.b
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

# x1 = Value(2.0)
# x2 = Value(3.0)

# n = Neuron(2)
# out = n([x1, x2])

# loss = (out - Value(1.0)) ** 2
# loss.backward()

# print(n.w[0].grad, n.w[1].grad)

class Layer:

  def __init__(self, numInputs, numOutputs):
    self.neurons = [Neuron(numInputs) for _ in range(numOutputs)]
  
  def parameters(self):
    params = []
    for n in self.neurons:
      params.extend(n.parameters())
    return params
    

  def __call__(self, x):
    return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, nin, nouts):
        self.sizes = [nin] + nouts
        self.layers = [
            Layer(self.sizes[i], self.sizes[i+1])
            for i in range(len(nouts))
        ]
    

    def parameters(self):
      params = []
      for layer in self.layers:
        params.extend(layer.parameters())
      return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
        

    
#training example
data = [
    ([Value(0.2), Value(0.3)], Value(1.0)),
    ([Value(-0.1), Value(0.5)], Value(0.0))
]

mlp = MLP(2, [2, 1])

y = Value(1.0)

losses = []

#training loop
for epoch in range(5000):
  loss = 0
  for x, y in data:
    pred = mlp(x)[0]
    loss = loss + (pred - y)**2

  for p in mlp.parameters():
    p.grad = 0
  loss.backward()
  for p in mlp.parameters():
    p.data -= 0.01 * p.grad

  losses.append(loss.data)  
  #  print(p.grad)
  if epoch % 100 == 0:
    print(f"loss:{loss.data}")


#plot loss curve
import matplotlib.pyplot as plt

plt.plot(losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig("plots/mlp_loss_curve.png")
plt.show()
