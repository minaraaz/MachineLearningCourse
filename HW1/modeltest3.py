import numpy as np
from classifier3 import *
from keras.datasets import mnist

# data load
(train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

# data reshape and black and white
train_images = train_images_original.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255.0

test_images = test_images_original.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255.0

train_labels = (np.arange(np.max(train_labels_original) + 1) == train_labels_original[:, None]).astype(float)
test_labels = (np.arange(np.max(test_labels_original) + 1) == test_labels_original[:, None]).astype(float)

# initial variables
batch_size = 512
epochs = 100
learning_rate = 1
m = 10
data_size = train_images.shape[0]
dimension = train_images.shape[1]
digit_classifier = []
models_train_accuracy = []
models_test_accuracy = []
predicted_labels_train = np.zeros((1,train_labels_original.shape[0]))
predicted_labels_test = np.zeros((1,test_labels_original.shape[0]))

# creating 10 classifiers
digit_classifier = softmax_classifier(batch_size, dimension, m, learning_rate)

# training classifiers
for epoch in range(epochs):
        Y_train_dic = []
        Y_test_dic = []
        shuffled_indices = np.random.permutation(data_size)
        train_images_shuffled = train_images[shuffled_indices]
        train_labels_shuffled = train_labels[shuffled_indices]

        for i in range(0, data_size, batch_size):
                xi = train_images_shuffled[i : i + batch_size]
                yi = train_labels_shuffled[i : i + batch_size]
                digit_classifier.update(xi,yi)
        
        
        Y_train_dic.append(digit_classifier.predict(train_images))
        Y_test_dic.append(digit_classifier.predict(test_images))

        predicted_labels_train = np.squeeze(np.argmax(Y_train_dic, axis = 0))
        predicted_labels_test = np.squeeze(np.argmax(Y_test_dic, axis = 0))

        train_accuracy = np.mean([int(i==j) for i, j in zip(predicted_labels_train, train_labels_original)]) * 100.0
        models_train_accuracy.append(train_accuracy)
        
        test_accuracy = np.mean([int(i==j) for i, j in zip(predicted_labels_test, test_labels_original)]) * 100.0
        models_test_accuracy.append(test_accuracy)

        print "epoch " + str(epoch) + " ====> train accuracy: {0:0.4f} ====> test accuracy: {1:0.4f}".format(train_accuracy, test_accuracy)