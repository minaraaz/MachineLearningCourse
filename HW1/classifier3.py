import numpy as np

class softmax_classifier:
    
    def __init__(self, batch_size, dimension, m, learning_rate):
        self.weight = np.zeros((dimension, m))
        self.bias = np.zeros((m, 1))
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.m = m
    
    def softmax(self, z):
        s = np.exp(z)/np.sum(np.exp(z))
        return s
    
    def backward(self, X, Y):
        # nabla_b = [np.zeros(b.shape) for b in self.bias]
        # nabla_w = [np.zeros(w.shape) for w in self.weight]

        z = np.dot(self.weight.T,X) + self.bias
        A = self.softmax(z)

        gradient_weight = 1.0/self.batch_size * np.dot(X, ((A-Y)*A*(1-A)).T)
        gradient_bias = 1.0/self.batch_size * np.sum(((A-Y)*A*(1-A)))

        return gradient_weight, gradient_bias
    
    def update(self, X, Y):
        X = X.transpose()
        gradient_weight, gradient_bias = self.backward(X, Y)

        self.weight = self.weight - self.learning_rate * gradient_weight
        self.bias = self.bias - self.learning_rate * gradient_bias


    def predict (self, X):
        Y_prediction = np.zeros((1,X.shape[0]))
        X = X.transpose()

        A = self.softmax(np.dot(self.weight.T, X) + self.bias)
        
        return A



def OneHotEncode(y):
    Onehot_encoded = []
    for i, val in enumerate(y):
        temp = [0] * 10
        temp[val] = 1
        Onehot_encoded.append(temp)
    
    return Onehot_encoded
