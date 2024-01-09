import random
import numpy as np
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Write code to train a CNN model which classifies the CIFAR100 dataset. Load the dataset directly from keras - https://keras.io/api/datasets/cifar100/
    # There are coarse and fine labels.
    # Load only the fine labeled dataset.
    # Images are 32x32 pixels.
    # Three color channels.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')

# make a dictionary for the following labels where the keysn are 0-99:
    # The labels are - ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# Print the shapes of the train and test data sets.
print('The shape of the train dataset is', x_train.shape)
print('The shape of the test dataset is', x_test.shape)
print('The shape of the train labels is', y_train.shape)
print('The shape of the test labels is', y_test.shape)

# Visualize the first 30 images from the train dataset
print('\nHere are the first 30 images from the train dataset: \n')
plt.figure(figsize=(10,10))
for value in range(0,30,1):
    image_number = random.randint(0, 50000)
    # print(image_number)
    test_sample = np.array(x_train[image_number])
    plt.subplot(8,8,value+1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.grid(False)
    plt.imshow(test_sample, cmap='gray')
plt.show()

# Scale the pixel values
x_train = x_train/255
x_test = x_test/255

# Build a CNN sequence of layers. Must contain the following layers. Hyper parameters are up to you. Use: At least 1 convolutional layer, At least 1 dropout layer, At least 1 maxpool layer, At least 1 flatten layer and At least 1 dense layer
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))

# Use the loss function sparse_categorical_crossentropy when compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with 25 epochs
history = model.fit(x_train, y_train, epochs=25, validation_split = 0.2)

# Plot the loss and accuracy curves for both train and validation sets.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Visualize the predicted and actual image labels for the first 30 images in the dataset.
print('\nHere are the first 30 images from the test dataset: \n')
y_pred = model.predict(x_test)
plt.figure(figsize=(10,10))
for value in range(0,30,1):
    image_number = random.randint(0, 10000)
    # print(image_number)
    test_sample = np.array(x_test[image_number])
    plt.subplot(8,8,value+1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.grid(False)
    plt.imshow(test_sample, cmap='gray')
    plt.title('Predicted: ' + str(y_pred[image_number]) + ' Actual: ' + str(y_test[image_number]))
plt.show()

# Visualize 30 random misclassified images
print('\nHere are 30 random misclassified images: \n')
plt.figure(figsize=(10,10))
for value in range(0,30,1):
    image_number = random.randint(0, 10000)
    # print(image_number)
    test_sample = np.array(x_test[image_number])
    plt.subplot(8,8,value+1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.grid(False)
    plt.imshow(test_sample, cmap='gray')
    plt.title('Predicted: ' + str(y_pred[image_number]) + ' Actual: ' + str(y_test[image_number]))
plt.show()



