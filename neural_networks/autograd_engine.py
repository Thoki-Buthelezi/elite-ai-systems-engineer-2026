import math

#implementation for autograd from scratch

class Value:
  def __init__(self, data, children=(), op=""):
    self.data = data
    self.prev = list(children)
    self.op = op
    self._backward = lambda: None
    self.grad = 0

  def coerce(self, other):
    return other if isinstance(other, Value) else Value(other)

  def __repr__(self):
    return f"value:{self.data}, grad:{self.grad}"

  def __add__(self, other):
    other = self.coerce(other)
    x = self.data + other.data
    out = Value(x, (self, other), "+")

    def _backward():
      self.grad += 1 * out.grad
      other.grad += 1 * out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = self.coerce(other)
    x = self.data * other.data
    out = Value(x, (self, other), "*")

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, n):
    assert isinstance(n, (int, float))
    x = self.data ** n
    out = Value(x, (self, ), f"**{n}")

    def _backward():
      self.grad += (n * self.data ** (n-1)) * out.grad
    out._backward = _backward
    return out

  #activations
  def sigmoid(self):
    x = 1 / (1 + math.exp(-self.data))
    out = Value(x, (self, ), "sigmoid")

    def _backward():
      self.grad += (x * (1-x)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    x = max(0, self.data)
    out = Value(x, (self, ), "relu")

    def _backward():
      self.grad += (x > 0) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    x = math.tanh(self.data)
    out = Value(x, (self, ), "tanh")

    def _backward():
      self.grad += (1 - x ** 2) * out.grad
    out._backward = _backward
    return out

  #operations defined from the derived operations above
  def __neg__(self): return self * -1
  def __sub__(self, other): return self + (-self.coerce(other))
  def __rsub__(self, other): return -self + (self.coerce(other))
  def __radd__(self, other): return self + other
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * self.coerce(other) ** -1
  def __rtruediv__(self, other): return self.coerce(other) * self ** -1

  #backpropagation
  def backward(self):
    topo = []
    visited = set()

    def build(v):
      if v not in visited:
        visited.add(v)
        for child in v.prev:
          build(child)
        topo.append(v)

    build(self)
    self.grad = 1 #derivative w.r.t to itself

    for node in reversed(topo):
      node._backward()


# x = Value(0.5)
# y = x ** 2
# y.backward()
# print("x.grad =", x.grad)

# a = Value(2.0)
# b = Value(3.0)
# c = a * b
# d = c + a
# e = d.tanh()
# e.backward()

# print(a.grad, b.grad)
