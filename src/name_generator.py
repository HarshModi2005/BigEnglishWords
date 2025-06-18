"""
Neural Network Name Generator

This script trains a neural network to generate names based on character patterns.
The model uses a 3-layer architecture:
1. Input layer: Embeddings for 3 consecutive characters (27 possible: a-z + '.')
2. Hidden layer: 200 nodes with tanh activation
3. Output layer: 27 nodes with softmax activation (predicting next character)

Architecture:
- Input: 3 characters -> 3x10 embeddings -> flattened to 30 features
- Hidden: Dense(200, activation='tanh')
- Output: Dense(27, activation='softmax')
"""

import numpy as np
import pandas as pd
import string
import random
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten
from keras.utils import to_categorical
import os

# Constants
EMBEDDING_DIM = 10
HIDDEN_SIZE = 200
SEQUENCE_LENGTH = 3
EPOCHS = 250
BATCH_SIZE = 50
MODEL_NAME = 'name_generator.h5'

class NameGenerator:
    def __init__(self):
        self.letters = string.ascii_lowercase + '.'
        self.vocab_size = len(self.letters)
        
        # Create character mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.letters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.letters)}
        
        # Initialize random embeddings
        self.embeddings = np.random.rand(self.vocab_size, EMBEDDING_DIM)
        
        self.model = None
    
    def load_data(self, filepath='../data/names.txt'):
        """Load and preprocess name data"""
        try:
            with open(filepath, 'r') as file:
                names = file.read().split()
            # Remove duplicates and convert to lowercase
            names = list(set([name.lower() for name in names if name.isalpha()]))
            print(f"Loaded {len(names)} unique names")
            return names
        except FileNotFoundError:
            print(f"Error: Could not find {filepath}")
            return []
    
    def prepare_training_data(self, names):
        """Convert names to training sequences"""
        X, Y = [], []
        
        for name in names:
            # Add start and end tokens
            padded_name = '.' + name + '.'
            
            # Create sequences of 3 characters to predict the 4th
            for i in range(len(padded_name) - SEQUENCE_LENGTH):
                # Input: 3 character embeddings
                sequence_embeddings = []
                for j in range(SEQUENCE_LENGTH):
                    char_idx = self.char_to_idx[padded_name[i + j]]
                    sequence_embeddings.append(self.embeddings[char_idx])
                
                X.append(np.array(sequence_embeddings))
                # Target: next character index
                Y.append(self.char_to_idx[padded_name[i + SEQUENCE_LENGTH]])
        
        X = np.array(X)
        Y = to_categorical(Y, num_classes=self.vocab_size)
        
        print(f"Training data shape: X={X.shape}, Y={Y.shape}")
        return X, Y
    
    def build_model(self):
        """Build the neural network model"""
        input_layer = Input(shape=(SEQUENCE_LENGTH, EMBEDDING_DIM))
        flatten_layer = Flatten()(input_layer)
        hidden_layer = Dense(HIDDEN_SIZE, activation='tanh')(flatten_layer)
        output_layer = Dense(self.vocab_size, activation='softmax')(hidden_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    def train(self, names_file='../data/names.txt'):
        """Train the name generator model"""
        # Load and prepare data
        names = self.load_data(names_file)
        if not names:
            return
        
        X, Y = self.prepare_training_data(names)
        
        # Build and train model
        self.model = self.build_model()
        print("Model architecture:")
        self.model.summary()
        
        print("Training model...")
        history = self.model.fit(X, Y, 
                               epochs=EPOCHS, 
                               batch_size=BATCH_SIZE, 
                               verbose=2, 
                               shuffle=True,
                               validation_split=0.1)
        
        # Save model
        os.makedirs('../model', exist_ok=True)
        self.model.save(f'../model/{MODEL_NAME}')
        print(f"Model saved to ../model/{MODEL_NAME}")
        
        return history

if __name__ == "__main__":
    generator = NameGenerator()
    generator.train() 