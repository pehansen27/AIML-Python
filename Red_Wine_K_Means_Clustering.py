# Peyton Hansen
# Red Wine K Means Clustering

import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset into a dataframe. Be sure to import the header. (2)
print('#1')
wine = pd.read_csv('wineQualityReds.csv')
print(wine.info())
print()

# Drop Wine from the dataframe. (1)
print('#2')
wine.drop(wine.columns[0], axis = 1, inplace = True)
print('The column \'Wine\' has been dropped from the dataframe.\n', wine.info())
print()

# Extract Quality and store it in a separate variable. (1)
print('#3')
quality = wine['quality']
print('The column \'Quality\' hase been stored as a separate variable.\n', quality)
print()

# Drop Quality from dataframe. (1) NOTE: keeping quality could add human bias to the data, so it is better to drop it
print('#4')
wine = wine.drop(columns = ['quality'])
print('The column \'Quality\' has been dropped from the dataframe.\n', wine.info())
print()

# Print the dataframe and Quality. (1)
print('#5')
print('Dataframe:\n', wine)
print('Quality:\n', quality)
print()

# Normalize all columns of the dataframe. Use the Normalizer class from sklearn.preprocessing. (2)
print('#6')
norm = Normalizer()
wine_norm = pd.DataFrame(norm.transform(wine), columns=wine.columns)
print('Dataframe has been normalized.')
print()

# Print the normalized dataframe. (1)
print('#7')
print(wine_norm)
print()

# Create a range of k values from 1:11 for KMeans clustering. Iterate on the k values and store the inertia for each clustering in a list. (2)
print('#8')
ks = range(1,11)
inertia = [] # want to minimize the inertia -> intracluster difference
for k in ks:
    model = KMeans(n_clusters=k) # instantiate the model
    model.fit(wine_norm) # train the model
    inertia.append(model.inertia_) # record the inertia
print('After iterating through values of K, the inertia for each clustering has been recorded.')
print()

# Plot the chart of inertia vs number of clusters. (2)
print('#9')
plt.plot(ks, inertia, '-o')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.grid(True)
plt.show()
print('This plot charts the inertia vs the number of k clusters.')
print()
# 4 is good for k number of clusters (elbow method looking at plot)

# What K (number of clusters) would you pick for KMeans? (1)
print('#10')
print('Using the elbow method, I would pick K=4 for KMeans in order to minimize complexity and maximize diversity.')
print()

# Now cluster the wines into K clusters. Use random_state = 2023 when you instantiate the KMeans model. Assign the respective cluster number to each wine. Print the dataframe showing the cluster number for each wine. (2)
print('#11')
model1 = KMeans(n_clusters=4, random_state=2023)
model1.fit(wine_norm)
labels = model1.predict(wine_norm)
wine_norm['Cluster Label'] = pd.Series(labels)
print(wine_norm.head())
print()

# Add the quality back to the dataframe. (1)
print('#12')
wine_norm["quality"] = quality
print('The \'Quality\' column has been added back to the dataframe. \n', wine_norm['quality'])
print()

# Now print a crosstab (from Pandas) of cluster number vs quality. Comment if the clusters represent the quality of wine. (3)
print('#13')
cross_tab = pd.crosstab(wine_norm["Cluster Label"], wine_norm["quality"])
print("Crosstab (Cluster Number vs. Quality):")
print('It does not seem that the cluster labels are very correlated to the quality. \nIf you look at a single row, the values are distributed across multiple columns - there is no single column value that correlates with that row.\n',cross_tab)