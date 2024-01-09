# Peyton Hansen
# ITP 259 Fall 2023
# HW 3: Logistic Regression and KNN classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


###########
#Problem 1#
###########

print("Probelm #1 \n \n")

#1. Read the dataset into a dataframe. (1)
titanic  = pd.read_csv('titanic.csv')
print('1) \n', titanic.head(), '\n')

#2. Explore the dataset and determine what is the target variable. (1)
print('2)')
print(titanic.info(), '\nThe target variable of this dataset is \'Survived\'.\n')

#3. Drop factor(s) that are not likely to be relevant for logistic regression. (2)
titanic = titanic.drop(columns=(['Passenger']))
print('3) I dropped the \'Passenger\' column since it is not likely to be relevant for logistic regression. \n')

#4. Convert all categorical feature variables into dummy variables. (2)
titanic = pd.get_dummies(titanic, columns=['Class','Sex','Age'], drop_first=True)
print('4) I got dummy variables for columns \'Class\', \'Sex\' and \'Age\'. \n', titanic.head(), '\n')

#5. Assign X and y (1)
X = titanic.drop(columns=['Survived'])
y = titanic['Survived']
print('5)\n','X: \n', X.head(),'\n', 'y: \n', y.head(), '\n')

#6. Partition the data into train and test sets (70/30). Use random_state = 2023. Stratify the data. (2)
X_train, X_test, y_train, y_test = \
    train_test_split(X,y,stratify=y,test_size=0.30,random_state=2023)
print('6) I partitioned the data using train_test_split into the following partitions: X_train, X_test, y_train, and y_test \n')

#7. Fit the training data to a logistic regression model. (1)
logReg = LogisticRegression(max_iter=1000)
logReg.fit(X_train,y_train)
print('7) The training data has been fit to a logistic regression model. \n')

#8. Display the accuracy of your predictions. (2)
y_pred = logReg.predict(X_test)
print('8) Accuracy:', metrics.accuracy_score(y_test, y_pred), '\n')

#9. Plot the lift curve. (2)
y_probas = logReg.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, y_probas)
print('9) This is the lift curve. \n')
plt.show()

#10. Plot the confusion matrix along with the labels (Yes, No).  (2)
matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=logReg.classes_).plot()
print('10) This is the confusion matrix. \n')
plt.show()

#11. Now, display the predicted value of the survivability of a male adult passenger traveling in 3rd class. (3)
value = {'Class_2nd':[False], 'Class_3rd':[True], 'Class_Crew':[False], 'Sex_Male':[True], 'Age_Child':[False]}
value_df = pd.DataFrame.from_dict(value)
print('11) The predicted survivability of a male adult passenger traveling in 3rd class is: ', logReg.predict(value_df), '\n')

print("\n \n")

###########
#Problem 2#
###########

print("Problem #2 \n \n")

#1. Create a DataFrame “diabetes_knn” to store the diabetes data. (1)
diabetes_knn = pd.read_csv('diabetes.csv')
print('1) \n', diabetes_knn.head(), '\n')

#2. Create the Feature Matrix (X) and Target Vector (y). (1)
#print(diabetes_knn.info())
X = diabetes_knn.drop(columns='Outcome')
y= diabetes_knn['Outcome']
print('2) The Feature Matrix and Target Vector are printed below, respectively: \n', X, '\n', y, '\n')

#3. Standardize the attributes of Feature Matrix (2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('3) The Attributes of the Feature Matrix have been standardized. \n', X_scaled, '\n')

#4. Split the Feature Matrix and Target Vector into three partitions. Training A, Training B and test. They should be in the ratio 60-20-20. random_state = 2023, stratify = y.  (1)
X_trainA, X_temp, y_trainA, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=2023, stratify=y)
X_trainB, X_test, y_trainB, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023, stratify=y_temp)
print('4) The Feature Matrix and Target Vector have been split into three partitions titled X_trainA, X_trainB, X_test, y_trainA, y_trainB, and y_test. \n')

#5. Develop a KNN based model based on Training A for various ks. K should range between 1 and 30. (1)
k_values = list(range(1, 31))
train_a_accuracies = []
train_b_accuracies = []
print('5) The KNN based model has been developed. \n')

#6. Compute the KNN score (accuracy) for training A and training B data for those ks. (2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA, y_trainA)
    train_a_pred = knn.predict(X_trainA)
    train_b_pred = knn.predict(X_trainB)
    train_a_accuracy = accuracy_score(y_trainA, train_a_pred)
    train_b_accuracy = accuracy_score(y_trainB, train_b_pred)
    train_a_accuracies.append(train_a_accuracy)
    train_b_accuracies.append(train_b_accuracy)
best_k = k_values[train_b_accuracies.index(max(train_b_accuracies))]
#print(best_k)
print('6) The accuracies have been recorded and will be plotted in the next question. \n')

#7. Plot a graph of training A and training B accuracy and determine the best value of k. Label the plot. (1)
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_a_accuracies, label='Training A Accuracy')
plt.plot(k_values, train_b_accuracies, label='Training B Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.legend()
plt.grid(True)
plt.show()
print('7) The best value for K is 15.' '\n') #chose this value based on the maximum of trainB accuracies
# but also by looking at the plot to find where trainA and trainB both have high accuracy

#8. Now, using the selected value of k, score the test data set (1)
newKnn = KNeighborsClassifier(n_neighbors=15)
newKnn.fit(X_trainA, y_trainA)
y_pred = newKnn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print('8) The accuracy of the test data set using the best value for k is:', test_accuracy, '\n')

#9. Plot the confusion matrix (as a figure). (1)
matrix2 = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=matrix2, display_labels=newKnn.classes_).plot()
print('9) This is the confusion matrix. \n')
plt.show()

#10. Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age. (1)
person = {'Pregnancies':[2], 'Glucose':[150], 'BloodPressure':[85], 'SkinThickness':[22], 'Insulin':[200], 'BMI':[30], 'DiabetesPedigreeFunction':[0.3], 'Age':[55]}
person_df = pd.DataFrame.from_dict(person)
print('10) The predicted outcome of the person described is: ', newKnn.predict(person_df), '\n')