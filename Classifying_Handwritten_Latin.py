# Peyton Hansen
# Classifying Handwritten Latin

import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. Read the dataset into a dataframe. (1)
df = pd.read_csv('A_Z Handwritten Data.csv')
print('1. The dataset is now in a dataframe called df. \n')

# 2. Explore the dataset and determine what is the target variable. (1)
print(df.head())
print(df.info(), '\n')
print('2. The target variable is the \'label\' column. \n')

# 3. Separate the dataframe into feature set and target variable. (1)
X = df.iloc[:,1:] #all rows, columns 1 to the end
y = df.iloc[:,0] #labels
print('3. The dataframe has been split into feature set \'X\' and target variable \'y\':, \n', X, '\n',y, '\n')

# 4. Print the shape of feature set and target variable. (1)
print('4. Here is the shape of the feature set: \n', X.shape, '\nHere is the shape of the target variable: \n', y.shape, '\n')

# 5. Is the target variable values letters or numbers? (1)
print('5. The target variable values are numbers. \n')

# 6. If the target variable is numbers, then how would you map numbers to letters? Hint: Use a data dictionary (1)
dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
        13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
print('6. Since the target variable is composed of numbers, we can map the numbers to letters by using a data dictionary: \n', dict, '\n')

# 7. Show a histogram (count) of the letters. (1)
print('7. Here is a histogram of the letters: \n')
labels = []
for number in y:
    letter = dict[number]
    labels += letter
# print(labels)
plt.hist(labels, bins = 26, rwidth=0.8, alpha = 0.7, color='skyblue')
plt.xlabel('Letters')
plt.ylabel('Count')
plt.title('Histogram of Letters')
plt.show()

# 8. Display 64 random letters from the dataset. Display their labels as shown below.
# Hint: Plot a pyplot figure. Use plt.subplot to make the 64 subplots. Use a for loop to iterate through each one. (2)
print('8. Here are 64 random letters from the dataset: \n')
for value in range(0,64,1):
    image_number = random.randint(0, 372451)
    # print(image_number)
    test_sample = np.array(X.iloc[image_number]).reshape(28,28)
    plt.subplot(8,8,value+1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.grid(False)
    plt.imshow(test_sample, cmap='gray')
    plt.title(str(labels[image_number]))
plt.show()

# 9. Partition the data into train and test sets (70/30). Use random_state = 2023. Stratify it. (1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2023, stratify=y)
print('9. The data has been partitioned into training and testing sets. \n')

# 10. Scale the train and test features. (1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('10. The train and test features have been scaled. \n')

# 11. Create an MLPClassifier. Experiment with various parameters. random_state = 2023. (2)
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu', max_iter=25, alpha=0.001, solver='adam', random_state=2023, learning_rate_init=0.01, verbose=True)
print('11. The MLPClassifier has been created with 3 hidden layers, each with 100 neurons, the \'relu\' activation function, 25 epochs, an alpha of 0.001, a random state of 2023,'
      ' an initial learning rate of 0.01 and verbose is set to true to display the loss over each epoch. \n')

# 12. Fit to train the model. (1)
mlp.fit(X_train, y_train)
print('\n12. The model has been fit using the training data. \n')

# 13. Plot the loss curve. (1)
print('13. Here is the loss curve: \n')
plt.plot(mlp.loss_curve_)
plt.show()

# 14. Display the accuracy of your model. (1)
print('14. The accuracy is', mlp.score(X_test,y_test), '\n')

# 15. Plot the confusion matrix along with the letters. (1)
y_pred = mlp.predict(X_test)
cm = confusion_matrix(y_pred, y_test, labels=mlp.classes_)
plt.rcParams.update({'font.size': 6})
plt.rcParams['figure.figsize'] = [10, 10]
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                                            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                                            'u', 'v', 'w', 'x', 'y', 'z']).plot()
print('15. The confusion matrix: \n')
plt.show()

# 16. Now, display the predicted letter of the first row in the test dataset.
# Also display the actual letter. Show both actual and predicted letters (as title) on the image of the letter. (3)
print('16. The letter of the first row in the test dataset: \n')
sample = np.array(X_test.iloc[0]).reshape(28,28)
plt.imshow(sample, cmap='gray')
plt.title('The predicted digit is: ' + str(y_pred[0]) + ' The actual digit is: ' + str(y_test[0]))
plt.show()
print()

# 17. Finally, display the actual and predicted letter of a misclassified letter. (3)
print('17. A misclassified letter: \n')
misclassified_df = X_test[y_pred != y_test]
print(misclassified_df)
misclassified_index = misclassified_df.sample(n=1).index
print(misclassified_index)
misclassified_sample = np.array(X_test.loc[misclassified_index]).reshape(28,28)
req_id = y_test.index.get_loc(misclassified_index[0])
plt.imshow(misclassified_sample, cmap='gray')
plt.title('The misclassified predicted digit is:' + str(dict[int(y_pred[req_id])]) + ' The actual digit is:' + str(dict[int(y_test.iloc[req_id])]))
plt.show()
