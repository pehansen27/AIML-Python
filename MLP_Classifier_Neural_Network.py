# Peyton Hansen
# MLP Classifier Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

# 2.1 Generate x, y coordinates of spirally distributed blobs in two colors. See figure below. You can search for code online to do this. (2)
    # 2.1.1 Note that the spirals should complete at least one full turn.
    # 2.1.2 Add some noise
    # 2.1.3 spiral parameters using given equations
print('The x and y coordinates have been generated to create spirally distributed blobs in two colors. \n')
N = 400
noise = 0.5
theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
firstR = 2*theta + np.pi
secondR = -2*theta - np.pi
firstX = firstR * np.cos(theta) + noise * np.random.randn(N)
firstY = firstR * np.sin(theta) + noise * np.random.randn(N)
secondX = secondR * np.cos(theta) + noise * np.random.randn(N)
secondY = secondR * np.sin(theta) + noise * np.random.randn(N)
X = np.vstack([np.column_stack([firstX, firstY]), np.column_stack([secondX, secondY])])
y = np.hstack([np.zeros(N), np.ones(N)])

# 2.2 create scatter plot
print('Here is the scatter plot of the spirals: \n')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='purple', label="Spiral 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='yellow', label="Spiral 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 2.3 Create partitions with 70% train dataset. Stratify the split. Use random state of 2023.
print('The data has been partitioned into X_Train, X_test, y_train and y_test. \n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023, stratify=y)

# 2.4 Now train the network using MLP Classifier from scikit learn. The parameters are your choice.
print('The NN has been trained using MLP classifier from scikit learn. '
      'The NN has 1 hidden layer with 100 neurons, uses the \'relu\' activation function, has 1000 epochs, uses an alpha of 0.001,'
      ' uses the adam solver, uses a random state of 2023, uses an initial learning rate of 0.01 and displays the loss over '
      'iterations since verbose is set to true. \n')
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, alpha=0.001,
                     solver='adam', verbose=True, random_state=2023, learning_rate_init=0.01)
mlp.fit(X_train, y_train)
print()

# 2.5 Plot the loss curve.
print('Here is the loss curve: \n')
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# 2.6 Print the accuracy of the test partition
y_pred = mlp.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('Accuracy:', score, '\n')

# 2.7 Display the confusion matrix
print('Here is the confusion matrix: \n')
cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_).plot()
plt.show()

# 2.8 Plot the decision boundary
    # a. To plot the decision boundary, create a mesh of x and y coordinates that cover the entire field (e.g., -20 to 20 for both x and y coordinates).
    # b. You can make the mesh points 0.1 apart. So, you will have a 400x400 mesh grid.
X1 = np.arange(-20, 20, 0.1)
X2 = np.arange(-20, 20, 0.1)
X1, X2 = np.meshgrid(X1, X2)

    # c. Then reshape the meshgrid to a dataframe that has two columns and 160000 rows (each row is a mesh point).
    # d. Then classify each point using the trained model (model.predict)
X_decision = pd.DataFrame({'X0': np.reshape(X1, 160000), 'X1':np.reshape(X2, 160000)})
Z = mlp.predict(X_decision)

    # e. Then plot both the original data points (spirals) and the mesh data points.
print('Here is the plot for the decision boundary: ')
plt.scatter(X_decision['X0'], X_decision['X1'],c=Z, cmap='cool')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='purple', label="Spiral 1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='yellow', label="Spiral 2")
plt.title('Decision Boundary')
plt.show()











