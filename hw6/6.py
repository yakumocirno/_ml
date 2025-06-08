import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data                  # 數值
        self.grad = 0.0                   # 對本節點的梯度
        self._backward = lambda: None    # 反向函數
        self._prev = set(_children)      # 前一層節點
        self._op = _op                   # 運算類型（+ * tanh ...）

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    # === 激活函數 ===
    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        ex = math.exp(x)
        out = Value(ex, (self,), 'exp')

        def _backward():
            self.grad += ex * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        sig = 1 / (1 + math.exp(-x))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward
        return out

    # === 反向傳遞 ===
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()