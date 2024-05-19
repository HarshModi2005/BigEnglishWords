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
import torch



model = load_model('karpathy.h5')
# print((model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi['.']]]).reshape(1,3,10)))[0])
# print(np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi['.']]]).reshape(1,3,10))[0]))
vowels = 'aeiou'
consonants = 'bcdfghjklmnpqrstvwxyz'
letters = string.ascii_lowercase
letters_list = list(letters) + ['.']

ltoi = {}
for i in range(len(letters)):
    ltoi[letters[i]] = i
ltoi['.']=26

#Create a mapping from integers to letters
itol = {}
for i in range(len(letters)):
    itol[i] = letters[i]
itol[26] = '.'
embeddings = np.random.rand(27,10)


with open('names.txt', 'r') as file:
    # Read the contents of the file
    naam = file.read()
    naam = list(set(naam.split()))
prob = np.zeros(26)

for i in range(len(naam)):
    prob[ltoi[naam[i][0]]] += 1
prob = prob/sum(prob)
print(prob)
    
# word+= itol[np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi['.']], embeddings[ltoi[word[0]]]]).reshape(1,3,10))[0])]
# word += itol[np.argmax(model.predict(np.array([embeddings[ltoi['.']], embeddings[ltoi[word[0]]], embeddings[ltoi[word[1]]]]).reshape(1,3,10)))]
outputs=[]
for i in range(30):
    word = '.' + itol[(torch.multinomial(torch.tensor(prob), 1)).item()]
    if word[-1] in vowels:
        word += random.choice(consonants)
    else:
        word += random.choice(vowels)
    while len(word)< 10 and word[-1] != '.':
        # word += itol[np.sort(model.predict(np.array([embeddings[ltoi[word[-2]]], embeddings[ltoi[word[-1]]], embeddings[ltoi[word[-1]]]]).reshape(1,3,10)))]
        
        prediction = model.predict(np.array([embeddings[ltoi[word[-2]]], embeddings[ltoi[word[-1]]], embeddings[ltoi[word[-1]]]]).reshape(1,3,10))[0]
        probs = []
        for i in range(len(prediction)):
            probs.append((prediction[i],letters_list[i]))
            probs.sort(reverse = True , key= lambda x: x[0])
        # biased_coin = random.random()
        # print(biased_coin)
        # if biased_coin <= 0.5:
        #     word += probs[0][1]
        # elif biased_coin > 0.5 and biased_coin <= 0.8:
        #     word += probs[1][1]
        # elif biased_coin > 0.8 and biased_coin <= 1:
        #     word += probs[2][1]
        word += probs[random.randint(0,3)][1]
        if word[-1] == word[-3]:
            word = word[0: len(word)-1]
        if word[-1] in vowels and word[-3] in vowels:
            word = word[0: len(word)-2]
            
        
        
        
    outputs.append(word)
print(outputs)
