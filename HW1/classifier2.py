import numpy as np

class classifier2:
    
    def __init__(self, number, batch_size, dimension, learning_rate):
        self.number=number
        self.weight = np.zeros((dimension, 1))
        self.bias = 0
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def sigmoid(self, z):
        s = 1.0/(1.0 + np.exp(-z))
        return s
    
    def backward(self, X, Y):
        z = np.dot(self.weight.T,X) + self.bias
        A = self.sigmoid(z)

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

        A = self.sigmoid(np.dot(self.weight.T, X) + self.bias)
        
        for i in range(A.shape[1]):
            if (A[:,i] > 0.5): 
                Y_prediction[:, i] = 1
            elif (A[:,i] <= 0.5):
                Y_prediction[:, i] = 0
        print "classifer 2 bud"
        return Y_prediction





