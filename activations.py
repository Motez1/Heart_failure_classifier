import numpy as np
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
    
class Relu(Activation):
    def __init__(self):
        def relu(x):
            return x * (x > 0)
        
        def relu_prime(x):
            return np.where(x > 0 , 1.0 , 0.0)

        super().__init__(relu,relu_prime)
        
class LeakyRelu(Activation):
    def __init__(self):
        def leaky(x):
            return x*(x>0) + 0.01*x*(x<=0)
        def leaky_deriv(x):
            return 1.0*(x>0) + 0.1 * x*(x<=0)
        super().__init__(leaky,leaky_deriv)