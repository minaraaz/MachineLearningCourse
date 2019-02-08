import numpy as np
from classifier import classifier
from keras.datasets import mnist


(train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

train_images = train_images_original.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images_original.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

batch_size = 128
epochs = 10
data_size = train_images.shape[0]
dimension = train_images.shape[1]
zero_classifier = classifier(0, batch_size, dimension, 0.1)
Y_zero = [1 if a == 0 else 0 for a in train_labels_original]
Y_zero = np.asarray(Y_zero)
Y_zero_prediction = [0] * data_size


for epoch in range(epochs):
    shuffled_indices = np.random.permutation(data_size)
    print len(shuffled_indices)
    train_images_shuffled = train_images[shuffled_indices]
    train_labels_shuffled = Y_zero[shuffled_indices]
    for i in range(0, data_size, batch_size):
        xi = train_images_shuffled[i : i + batch_size]
        yi = train_labels_shuffled[i : i + batch_size]
        print xi.shape
        zero_classifier.update(xi,yi)

Y_zero_prediction = zero_classifier.predict(train_images)

    