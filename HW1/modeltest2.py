import numpy as np
from classifier2 import classifier2
from keras.datasets import mnist


def model(number,batch_size, epochs, learning_rate, train_images, train_labels_original, test_images, test_labels_original):
        data_size = train_images.shape[0]
        dimension = train_images.shape[1]
        number_classifier = classifier2(number, batch_size, dimension, learning_rate)
        Label_train_binary = [1 if a == number_classifier.number else 0 for a in train_labels_original]
        Label_test_binary = [1 if a == number_classifier.number else 0 for a in test_labels_original]
        Label_train_binary = np.asarray(Label_train_binary)
        Label_test_binary = np.asarray(Label_test_binary)

        for epoch in range(epochs):
                shuffled_indices = np.random.permutation(data_size)
                train_images_shuffled = train_images[shuffled_indices]
                train_labels_shuffled = Label_train_binary[shuffled_indices]
                for i in range(0, data_size, batch_size):
                        xi = train_images_shuffled[i : i + batch_size]
                        yi = train_labels_shuffled[i : i + batch_size]
                        number_classifier.update(xi,yi)
        Label_prediction_train = number_classifier.predict(train_images)
        Label_prediction_test = number_classifier.predict(test_images)

        train_accuracy = 100.0 - np.mean(np.abs(Label_prediction_train - Label_train_binary) * 100.0)
        test_accuracy = 100.0 - np.mean(np.abs(Label_prediction_test - Label_test_binary) * 100.0)

        return train_accuracy, test_accuracy


(train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

train_images = train_images_original.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255.0


test_images = test_images_original.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255.0


batch_size = 128
epochs = 100
learning_rate = 0.1


train_accuracy, test_accuracy = model(0,batch_size, epochs, learning_rate, train_images, train_labels_original, test_images, test_labels_original)


print "train accuracy: "+ str(train_accuracy)
print "test accuracy: "+ str(test_accuracy)
    