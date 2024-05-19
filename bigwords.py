#Neural Network Architecture
#Neural 1
# 1. Input layer: 27 nodes (26 letters + 1 for the dot)
# 2. Hidden layer: 100 nodes
# 3. Output layer: 10 nodes
#Neural 2
# 1. Input layer: 30 nodes
# 3. Output layer: 200 nodes

#Neural 3 
# 1. Input layer: 200 nodes
# 3. Output layer: 27 nodes

# Imports
import numpy as np
import pandas as pd
import string
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#Reading the data
data = pd.read_csv('worldcities.csv')
cities = data['city'].tolist()

# Manipulating the data
for i in range(len(cities) - 1, -1, -1):  # iterate in reverse order
    if not cities[i].isalpha():
        cities.pop(i)  # remove item by index
    else:
        cities[i] = cities[i].lower()
letters = string.ascii_lowercase
for i in range(len(cities)-1,-1,-1):
    for j in cities[i]:
        if j not in letters:
            cities.pop(i)
            break
        else:
            continue
for i in range(len(cities)-1, -1 , -1):
    cities[i] = '.'+ cities[i] + '.'

#Create a mapping from letters to integers
ltoi = {}
for i in range(len(letters)):
    ltoi[letters[i]] = i
ltoi['.']=26

#Create a mapping from integers to letters
itol = {}
for i in range(len(letters)):
    itol[i] = letters[i]
itol[26] = '.'

#Neural 1

#Initializing Embeddings
embeddings = np.random.rand(27,10)

#Create a mapping from letters to integers
ltoi = {}
for i in range(len(letters)):
    ltoi[letters[i]] = i
ltoi['.']=26

#Create a mapping from integers to letters
itol = {}
for i in range(len(letters)):
    itol[i] = letters[i]
itol[26] = '.'

print(embeddings)