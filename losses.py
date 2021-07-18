import numpy as np

def mse(y,prediction):
    return np.mean(np.power((prediction - y),2))

def mse_prime(y,prediction):
    return 2*(prediction - y) / np.size(y)