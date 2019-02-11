import numpy as np

class classifier:
    
    def __init__(self, number, batch_size, dimension, learning_rate,train_labels_original,test_labels_original):
        self.number=number
        self.weight = np.zeros((dimension, 1))
        self.bias = 0
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Label_train_binary = [1 if a == number else 0 for a in train_labels_original]
        self.Label_test_binary = [1 if a == number else 0 for a in test_labels_original]
        self.Label_train_binary = np.asarray(self.Label_train_binary)
        self.Label_test_binary = np.asarray(self.Label_test_binary)
        self.train_labels_shuffled = np.zeros((1,train_labels_original.shape[0]))
        # self.Y_prediction = np.zeros((1,train_labels_original.shape[0]))
    
    def sigmoid(self, z):
        s = 1.0/(1.0 + np.exp(-z))
        return s
    
    def backward(self, X, Y):
        z = np.dot(self.weight.T,X) + self.bias
        A = self.sigmoid(z)

        gradient_weight = 1.0/self.batch_size * np.dot(X, (A-Y).T)
        gradient_bias = 1.0/self.batch_size * np.sum(A-Y)

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
        
        return A

    def Label_shuffle(self, shuffled_indices):
        self.train_labels_shuffled = self.Label_train_binary[shuffled_indices]

    def train(self, i, xi):
        yi = self.train_labels_shuffled[i : i + self.batch_size]
        self.update(xi,yi)



