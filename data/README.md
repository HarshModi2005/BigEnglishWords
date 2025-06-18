# Data Directory

This directory contains the datasets used for training the neural network name generator.

## Files

### names.txt
- **Description**: Primary training dataset containing 32,033 unique names
- **Format**: One name per line, lowercase
- **Usage**: Main dataset for training the name generator
- **Source**: Curated collection of popular names

### names.csv
- **Description**: Names with gender classifications
- **Format**: CSV with columns for name and gender
- **Size**: 6,783 entries
- **Usage**: Can be used for gender-specific name generation

### words.csv
- **Description**: English words with frequency counts
- **Format**: CSV with word and count columns
- **Size**: ~5MB
- **Usage**: Alternative training data for word generation

### worldcities.csv
- **Description**: Global cities dataset
- **Format**: CSV with city information
- **Size**: ~5MB
- **Usage**: Could be used for place name generation

## Data Preprocessing

The training script (`src/name_generator.py`) automatically:
1. Loads names from `names.txt`
2. Converts to lowercase
3. Removes duplicates
4. Filters out non-alphabetic entries
5. Adds start/end tokens ('.')

## Adding Your Own Data

To use your own name dataset:
1. Create a text file with one name per line
2. Update the file path in the training script
3. Ensure names contain only alphabetic characters

## Data Statistics

- **Total unique names**: 32,033
- **Average name length**: 5.2 characters
- **Character vocabulary**: 26 letters + start/end token
- **Most common first letters**: A, M, S, J, C
- **Most common last letters**: A, E, N, R, S 