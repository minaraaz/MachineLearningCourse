import numpy as np
import keras
from keras import models
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical  
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
import matplotlib.pyplot as plt

# load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# initial parameters
class_names = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_size = train_images.shape[0]
valid_size = data_size * 20 / 100
train_size = data_size * 80 / 100
batch_size = 32
epochs = 15


# preparing dataset
train_labels = to_categorical(train_labels, num_classes = len(class_names))
test_labels = to_categorical(test_labels, num_classes = len(class_names))
test_images = test_images.astype('float32') / 255.0
train_images = train_images.astype('float32') / 255.0

# creating validation set
shuffled_indices = np.random.permutation(data_size)
train_set_x = train_images[shuffled_indices]
train_set_y = train_labels[shuffled_indices]

validation_set_x = train_set_x[-valid_size:]
validation_set_y = train_set_y[-valid_size:]

train_set_x = train_set_x[:train_size]
train_set_y = train_set_y[:train_size]

print validation_set_x.shape
print validation_set_y.shape

print train_set_x.shape
print train_set_y.shape

# creating model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu', strides = 2))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu', strides=2))
model.add(Conv2D(128, (3, 3), activation = 'relu'))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'rmsprop')
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = sgd)


# training model
history = model.fit(train_set_x, train_set_y, batch_size = batch_size, epochs=15, verbose=2, validation_data = (validation_set_x,validation_set_y))

# plot loss
training_loss = history.history['acc']
test_loss = history.history['val_acc']

epoch_count = range(1, len(training_loss) + 1) 

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()