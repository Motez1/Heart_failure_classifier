import numpy as np
import matplotlib.pyplot as plt
from losses import *
from activations import *
from dense import *


class NeuralNetwork:

    def __init__(self,layers=[],loss_function=mse , loss_deriv = mse_prime):
        self.layers = layers
        self.loss = loss_function
        self.loss_deriv = loss_deriv
        self.errors = []
    
    def train(self,X,Y,epochs=10,learning_rate=0.1):
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                # forward
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                # error
                error += self.loss(y, output)

                # backward
                grad = self.loss_deriv(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(X)
            if (e % 1 == 0):
                self.errors.append(error)
            print(f"{e + 1}/{epochs}, error={error}")

    def predict(self,x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def save(self,path):
        pass 

    def load(self,path):
        pass
