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
from keras.layers import Dense, Input, Flatten, Embedding, Reshape
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
#Reading the data
# data = pd.read_csv('names.csv' , header=None)
# naam = data[0].tolist()

# print('Manipulating the data')
# print('harsh modi'.split())
# for i in range(len(naam) - 1):  # iterate in reverse order
    
#     if type(naam[i]) == str and len( naam[i].split() )>1 and naam[i].split()[0].isalpha():
#         naam[i] = naam[i].split()[0]  # remove item by index
#     else:
#         continue
# for i in range(len(naam)):
#     naam[i] = naam[i].lower()

# naam = list(set(naam))

# Open the file
with open('names.txt', 'r') as file:
    # Read the contents of the file
    naam = file.read()
    naam = list(set(naam.split()))

# Now data contains the contents of the file
letters = string.ascii_lowercase
letters_list = list(letters) + ['.']

print('Create a mapping from letters to integers')
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

print('Initializing Embeddings')
embeddings = np.random.rand(27,10)


print('Preparing the data')
X = []
Y = []

for city in naam:
   
    city = '.' + city + '.'
    #write the code for taking set of all 3 characters in city
    for i in range(len(city)-3):
        X.append(np.array([embeddings[ltoi[city[i]]], embeddings[ltoi[city[i+1]]] , embeddings[ltoi[city[i+2]]]]))
        Y.append(ltoi[city[i+3]])  # Append to Y in each iteration
print(len(X[0]))
# print('Neural Network')
# model = Sequential([Dense(200, input_dim=30, activation='tanh'), Dense(27, activation='softmax')])

# print('Compiling the model')
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# print('Fitting the model')

# model.fit(X, Y, epochs=100, batch_size=10,verbose=1)
# print('Saving the model')
# model.save('femalenames.h5')


# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
# Assuming `input_shape` is the shape of your input data
input_layer = Input(shape=(3, 10))

# Flatten layer
flatten_layer = Flatten()(input_layer)

# Dense layer
dense_layer = Dense(200, activation='tanh')(flatten_layer)

# Dense layer with 27 units
output_layer = Dense(27, activation='softmax')(dense_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
from keras.utils import to_categorical

# Assuming `Y` is your target data
Y = to_categorical(Y, num_classes=27)


# Train the model
print(X.shape)
print(Y.shape)
print(X)
print(Y)
model.fit(X, Y, epochs=250, batch_size=50, verbose=2 , shuffle= True)
model.save('karpathy.h5')
# model = load_model('bigwords.h5')
# # print((model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi['.']]]).reshape(1,3,10)))[0])
# # print(np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi['.']]]).reshape(1,3,10))[0]))
# vowels = 'aeiou'
# consonants = 'bcdfghjklmnpqrstvwxyz'
# # word+= itol[np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi[word[0]]]]).reshape(1,3,10))[0])]
# # word += itol[np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi[word[0]]], embeddings[ltoi[word[1]]]]).reshape(1,3,10)))]
# outputs=[]
# for i in range(30):
#     word = '.' + random.choice(consonants) + random.choice(vowels) 

#     while len(word)< 10 and word[-1] != '.':
#         # word += itol[np.sort(model.predict(np.array([embeddings[ltoi[word[-2]]], embeddings[ltoi[word[-1]]], embeddings[ltoi[word[-1]]]]).reshape(1,3,10)))]
        
#         prediction = model.predict(np.array([embeddings[ltoi[word[-2]]], embeddings[ltoi[word[-1]]], embeddings[ltoi[word[-1]]]]).reshape(1,3,10))[0]
#         probs = []
#         for i in range(len(prediction)):
#             probs.append((prediction[i],letters_list[i]))
#             probs.sort(reverse = True , key= lambda x: x[0])
#         # biased_coin = random.random()
#         # print(biased_coin)
#         # if biased_coin <= 0.5:
#         #     word += probs[0][1]
#         # elif biased_coin > 0.5 and biased_coin <= 0.8:
#         #     word += probs[1][1]
#         # elif biased_coin > 0.8 and biased_coin <= 1:
#         #     word += probs[2][1]
#         word += probs[random.randint(0,1)][1]
        
        
        
#     outputs.append(word)
# print(outputs)
# # print(word)
# # labelletter = [itol[i] for i in range(1,26)]
# # labelletter.append('.')


# # # Assuming plot_data is a 2D array where each row is an embedding
# # x = prediction[0][:, 0]  # First dimension of each embedding
# # y = prediction[0][:, 1]  # Second dimension of each embedding
# # labelletter.remove('i')
# # x = np.delete(x, 0)
# # y = np.delete(y, 0)
# # x = np.delete(x, 7)
# # y = np.delete(y, 7)

# # plt.figure(figsize=(8,8))
# # plt.scatter(x, y, s=200)
# # # labelletter.remove('b')
# # # labelletter.remove('k')

# # # x = np.delete(x,1)
# # # x= np.delete(x,9)

# # # y =np.delete(y,1)
# # # y = np.delete(y,9)
# # print(labelletter)
# # print(x)
# # print(len(x))

# # print(labelletter)
# # print(len(labelletter))
# # print(y)
# # print(len(y))
# # # Add labels to each point
# # for i, label in enumerate(labelletter):
# #     plt.annotate(label, (x[i], y[i]))

# # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # plt.show()

