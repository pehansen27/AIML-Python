# Peyton Hansen
# Classifying Apparel Fashion

import random
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#1. Set random seed to 2023, for reproducibility of results.
keras.utils.set_random_seed(2023)
tf.config.experimental.enable_op_determinism()
print('1. Using random seed 2023. \n')

#2. Load the dataset. (1)
fashion = tf.keras.datasets.fashion_mnist
print('2. The dataset has been loaded. \n')

#3. Separate the dataset into feature set and target variable. Also separate the train and test partitions. (1)
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
print('3. I separated the dataset into feature set and target variable and train and test partitions. \n')

#4. Print the shapes of the train and test sets for the features and target. (1)
print('4. train images: \n', train_images.shape, '\ntrain labels: \n',train_labels.shape, '\ntest images: \n',test_images.shape, '\ntest labels: \n', test_labels.shape, '\n')

#5. Are the target variable values clothing or numbers? (1)
print('5. The target variable values are numbers:', train_labels, '\n')

#6. If it is numbers, then how would you map numbers to clothing? Hint: Use a data dictionary (1)
value_dict = {0:'T-shirt/Top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Snearker', 8:'Bag', 9:'Ankle Boot'}
print('6. You can use a data dictionary to define the label for each numeric value:', value_dict, '\n')

#7. Show a histogram (count) of the apparel. (1)
train_df = pd.DataFrame(train_labels)
labels = list(value_dict.values())
# print(labels)
sb.countplot(x=train_labels, hue=None, palette='cool')
plt.xticks(np.arange(len(value_dict)), labels=labels)
plt.xlabel('Apparel')
plt.ylabel('Count')
plt.title('Histogram of Apparel')
print('7. Histogram of Apparel: ')
plt.show()

#8. Display 25 random apparel from the train dataset. Display their labels as shown below. (2)
print('\n8. Here are 25 random apparel from the train dataset: \n')
plt.figure(figsize=(10,10))
for value in range(0,25,1):
    image_number = random.randint(0, 60000)
    # print(image_number)
    test_sample = np.array(train_images[image_number])
    plt.subplot(8,8,value+1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.grid(False)
    plt.imshow(test_sample, cmap='gray')
    plt.title(str(value_dict[train_labels[image_number]]))
plt.show()

#9. Scale the train and test features. (1) IS THIS CORRECT?
train_images_scaled = train_images / 255
test_images_scaled = test_images / 255
print('\n9. The train and test features have been scaled. \n')

#10. Create a keras model of sequence of layers.
    #a. 1 Flatten layer and two dense layers.
    #b. Experiment with number of neurons and activation functions. (3)
model = tf.keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) #flattened rows of pixels in dataset - takes place of input layer
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
print('10. I created a keras model with 1 flatten layer and two dense layers\n')

#11. Add a dense layer as output layer. Choose the appropriate number of neurons and activation function. (1)
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print('11. I added one dense layer for an output layer, 10 neurons and used the softmax activation function.\n')

#12. Display the model summary. (1)
print('12. Model Sumary:')
model.summary()

#13. Set the model loss function as sparse_categorical_crossentropy. Set the optimizer as sgd. Set the metrics as accuracy. (1)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
print('\n13. I set the model loss function as sparse_categorical_crossentropy, the optimizer as sgd and the metrics as accuracy \n.')

#14. Fit to train the model. Use at least 100 epochs. (1)
history = model.fit(train_images_scaled, train_labels, epochs=100)
print('\n14. I fit the model using 100 epochs. \n')

#15. Plot the loss curve. (1)
print('15. Loss Curve:')
pd.DataFrame(history.history).plot()
plt.show()

#16. Display the accuracy of your model. (1)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n16. Test accuracy:', test_acc, '\n')

#17. Now, display the predicted apparel of the first row in the test dataset.
# Also display the actual apparel. Show both actual and predicted letters (as title) on the image of the apparel. (3)
predictions = model.predict(test_images)
first_image = test_images[0]
first_label = test_labels[0]
predicted_label = np.argmax(predictions[0])
plt.imshow(first_image, cmap='gray')
print('17. The first row in the test dataset:')
plt.title('\nThe predicted label is: '+ str(value_dict[predicted_label]) + ' The actual label is: ' + str(value_dict[first_label]))
plt.show()

#18. Finally, display the actual and predicted label of a misclassified apparel. (3)
print('\n18. The actual and predicted label of a misclassified apparel')
misclassified_indexes = []
for i in range(len(test_images)):
    if np.argmax(predictions[i]) != test_labels[i]:
        misclassified_indexes.append(i)
        if len(misclassified_indexes) == 1:
            break
misclassified_index = misclassified_indexes[0]
misclassified_image = test_images[misclassified_index]
misclassified_actual_label = test_labels[misclassified_index]
misclassified_predicted_label = np.argmax(predictions[misclassified_index])
plt.imshow(misclassified_image, cmap='gray')
plt.title('The misclassified predicted label is: ' + str(value_dict[misclassified_predicted_label]) + ' The actual label is: ' + str(value_dict[misclassified_actual_label]))
plt.show()