import numpy as np

class softmax_classifier:
    
    def __init__(self, batch_size, dimension, m, learning_rate):
        self.weight = np.zeros((dimension, m))
        self.bias = np.zeros((m, 1))
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_class = m
    
    def softmax(self, z):
        z = z - np.max(z, axis=0)
        s = np.exp(z)/np.sum(np.exp(z), axis=0)
        return s

    def backward(self, X, Y):
        z = np.dot(self.weight.T,X) + self.bias
        A = self.softmax(z)
        A = np.repeat(A[np.newaxis], self.num_class, axis = 0) - np.repeat(np.identity(self.num_class)[:,:, np.newaxis], self.batch_size, axis=2)
        YA = np.repeat(Y.T[:, np.newaxis], self.num_class, axis = 1) * A
        YA = np.sum(YA, axis=0)

        gradient_weight = 1.0/self.batch_size * np.dot(X, YA.T)
        gradient_bias = 1.0/self.batch_size * np.sum(YA, axis=1,keepdims=True)

        return gradient_weight, gradient_bias
    
    def update(self, X, Y):
        X = X.transpose()
        gradient_weight, gradient_bias = self.backward(X, Y)

        self.weight = self.weight - self.learning_rate * gradient_weight
        self.bias = self.bias - self.learning_rate * gradient_bias


    def predict (self, X):
        X = X.transpose()
        A = self.softmax(np.dot(self.weight.T, X) + self.bias)
        
        return A

