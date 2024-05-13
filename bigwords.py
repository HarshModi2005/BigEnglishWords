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
data = pd.read_csv('words.csv')
cities = data['word'].tolist()
