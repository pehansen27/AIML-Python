# Peyton Hansen
# ITP 259 Fall 2023
# HW 1: Red Wine Quality Data Analysis

import pandas as pd
import random


# Read the dataset into a dataframe. Be sure to import the header.
print('#1')
redWine = pd.read_csv('wineQualityReds.csv')
print(redWine.info())
print()

# Print the first 10 rows of the dataframe.
print('#2')
print('The first 10 rows of the dataframe are as follows: \n', redWine.head(10))
print()

# Print the dataframe in descending order of volatility.
print('#3')
print('The dataframe in descending order of volatility: \n', redWine.sort_values(by = ['volatile.acidity'], ascending=False))
print()

# Display those wines that have quality of 7.
print('#4')
redWine2 = redWine[redWine.quality ==7]
print('The following Wines have a quality of 7: \n', redWine2['Wine'].tolist())
print()

# What is the average pH of all wines?
print('#5')
print('The average pH of all wines is:', redWine.pH.mean())
print()

# How many wines have alcohol level more than 10?
print('#6')
level10 = redWine[redWine.alcohol > 10]
print('The amount of wines that have alcohol level more than 10 is:', level10.shape[0])
print()

# Which wine has the highest alcohol level?
print('#7')
highestAlcohol = redWine.sort_values(by='alcohol', ascending=False)
print('The wine with the highest alcohol level is: Wine', highestAlcohol.iloc[0,0])
print()

# List the residual sugar level of a random wine.
print('#8')
random = random.choice(redWine['residual.sugar'])
print('The residual sugar level of a random wine is:', random)
print()

# List a random wine that has quality of 4.
print('#9')
new = redWine[redWine.quality == 4]
choice = new.sample(n=1, random_state = 42)
wineChoice = choice['Wine'].index.values[0]
print('A random wine with a quality of 4 is: Wine', wineChoice)
print()

# Drop wines that have quality 4. How many wines are left in the dataframe?
print('#10')
filtered = redWine[redWine['quality'] != 4]
remainingWines = len(filtered)
print('There are', remainingWines, 'wines left in the dataframe after dropping wines with a quality of 4.')