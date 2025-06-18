"""
Name Generation Script

This script loads a trained neural network model and generates new names
based on learned patterns from the training data.
"""

import numpy as np
import string
import random
import tensorflow as tf
from keras.models import load_model
import torch
import os
import argparse

class NameGeneratorInference:
    def __init__(self, model_path='../model/name_generator.h5'):
        self.letters = string.ascii_lowercase + '.'
        self.vocab_size = len(self.letters)
        
        # Create character mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.letters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.letters)}
        
        # Initialize embeddings (same as training)
        self.embeddings = np.random.rand(self.vocab_size, 10)
        
        # Load model
        try:
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        except:
            print(f"Could not load model from {model_path}")
            self.model = None
    
    def calculate_starting_probabilities(self, names_file='../data/names.txt'):
        """Calculate probability distribution for first letters"""
        try:
            with open(names_file, 'r') as file:
                names = file.read().split()
            names = [name.lower() for name in names if name.isalpha()]
            
            # Count first letter frequencies
            first_letter_counts = np.zeros(26)  # Only a-z, not '.'
            for name in names:
                if name and name[0] in string.ascii_lowercase:
                    first_letter_counts[self.char_to_idx[name[0]]] += 1
            
            # Convert to probabilities
            probabilities = first_letter_counts / np.sum(first_letter_counts)
            return probabilities
        except:
            # Default uniform distribution if file not found
            return np.ones(26) / 26
    
    def generate_name(self, max_length=10, use_realistic_start=True):
        """Generate a single name"""
        if self.model is None:
            return "Model not loaded"
        
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        # Start the name
        if use_realistic_start:
            # Use learned probability distribution for first letter
            start_probs = self.calculate_starting_probabilities()
            first_char_idx = np.random.choice(26, p=start_probs)
            first_char = self.idx_to_char[first_char_idx]
        else:
            first_char = random.choice(consonants)
        
        # Ensure alternating vowel/consonant pattern initially
        if first_char in vowels:
            second_char = random.choice(consonants)
        else:
            second_char = random.choice(vowels)
        
        name = '.' + first_char + second_char
        
        # Generate rest of the name
        while len(name) < max_length and name[-1] != '.':
            # Get last 3 characters for prediction
            context = name[-3:]
            if len(context) < 3:
                context = '.' * (3 - len(context)) + context
            
            # Convert to embeddings
            context_embeddings = []
            for char in context:
                char_idx = self.char_to_idx[char]
                context_embeddings.append(self.embeddings[char_idx])
            
            # Predict next character
            input_array = np.array([context_embeddings])
            prediction = self.model.predict(input_array, verbose=0)[0]
            
            # Get top predictions with probabilities
            char_probs = [(prediction[i], self.letters[i]) for i in range(len(prediction))]
            char_probs.sort(reverse=True)
            
            # Sample from top predictions (add some randomness)
            next_char = char_probs[random.randint(0, min(3, len(char_probs)-1))][1]
            
            # Apply some heuristics to make names more realistic
            if len(name) > 2:
                # Avoid triple letters
                if next_char == name[-1] == name[-2]:
                    next_char = char_probs[random.randint(1, min(4, len(char_probs)-1))][1]
                
                # Avoid too many consecutive vowels
                if (next_char in vowels and name[-1] in vowels and 
                    len(name) > 3 and name[-2] in vowels):
                    # Try to pick a consonant from top predictions
                    for prob, char in char_probs[:5]:
                        if char in consonants:
                            next_char = char
                            break
            
            name += next_char
        
        # Clean up the name
        name = name.strip('.')
        return name.capitalize() if name else "Error"
    
    def generate_names(self, count=10, max_length=10):
        """Generate multiple names"""
        names = []
        for _ in range(count):
            name = self.generate_name(max_length)
            if name and name != "Error" and len(name) > 1:
                names.append(name)
        return names

def main():
    parser = argparse.ArgumentParser(description='Generate names using trained neural network')
    parser.add_argument('--count', type=int, default=10, help='Number of names to generate')
    parser.add_argument('--max-length', type=int, default=10, help='Maximum name length')
    parser.add_argument('--model', type=str, default='../model/name_generator.h5', help='Path to model file')
    
    args = parser.parse_args()
    
    generator = NameGeneratorInference(args.model)
    
    if generator.model is None:
        print("Could not load model. Please train the model first using name_generator.py")
        return
    
    print(f"Generating {args.count} names...")
    names = generator.generate_names(args.count, args.max_length)
    
    print("\nGenerated names:")
    for i, name in enumerate(names, 1):
        print(f"{i:2d}. {name}")

if __name__ == "__main__":
    main() 