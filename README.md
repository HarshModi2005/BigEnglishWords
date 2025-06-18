# Neural Network Name Generator

A neural network-based name generator that learns patterns from real names and generates new, plausible-sounding names. The model uses character-level sequence prediction to create names that follow similar phonetic and structural patterns as the training data.

## Features

- **Character-level neural network**: Uses embeddings and sequence prediction
- **Configurable generation**: Control name length and generation parameters
- **Realistic patterns**: Learns from real name data to generate plausible names
- **Clean architecture**: Well-structured, documented code
- **Easy to use**: Simple CLI interface for training and generation

## Project Structure

```
BigEnglishWords/
├── src/                    # Source code
│   ├── name_generator.py   # Training script
│   └── generate_names.py   # Name generation script
├── data/                   # Training data
│   ├── names.txt          # Names dataset (32K+ names)
│   ├── names.csv          # Names with gender labels
│   ├── words.csv          # English words with frequency
│   └── worldcities.csv    # World cities dataset
├── model/                  # Trained models
│   └── *.h5               # Keras model files
├── examples/              # Example outputs
│   └── output.txt         # Sample generated names
└── docs/                  # Documentation
```

## Model Architecture

The neural network uses a simple but effective architecture:

1. **Input Layer**: 3 consecutive characters → 3×10 embeddings → flattened to 30 features
2. **Hidden Layer**: Dense layer with 200 neurons and tanh activation
3. **Output Layer**: Dense layer with 27 neurons (a-z + '.') and softmax activation

The model predicts the next character based on the previous 3 characters, using '.' as start/end tokens.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd BigEnglishWords
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow numpy pandas torch matplotlib
   ```

## Usage

### Training a New Model

```bash
cd src
python name_generator.py
```

This will:
- Load the names dataset from `../data/names.txt`
- Train the neural network for 250 epochs
- Save the trained model to `../model/name_generator.h5`

### Generating Names

```bash
cd src
python generate_names.py --count 20 --max-length 8
```

**Parameters:**
- `--count`: Number of names to generate (default: 10)
- `--max-length`: Maximum name length (default: 10)
- `--model`: Path to model file (default: ../model/name_generator.h5)

### Example Output

```
Generating 10 names...

Generated names:
 1. Aria
 2. Kaden
 3. Zara
 4. Liam
 5. Maya
 6. Elena
 7. Noel
 8. Ivy
 9. Cole
10. Luna
```

## Data Sources

- **names.txt**: Collection of 32,000+ popular names
- **names.csv**: Names with gender classifications
- **words.csv**: English words with frequency counts
- **worldcities.csv**: Global cities dataset

## Customization

### Modifying Training Parameters

Edit the constants in `src/name_generator.py`:

```python
EMBEDDING_DIM = 10      # Size of character embeddings
HIDDEN_SIZE = 200       # Hidden layer neurons
SEQUENCE_LENGTH = 3     # Input sequence length
EPOCHS = 250           # Training epochs
BATCH_SIZE = 50        # Training batch size
```

### Using Different Datasets

Replace `data/names.txt` with your own name dataset (one name per line).

## Model Performance

The model typically achieves:
- **Training accuracy**: ~60-70%
- **Validation accuracy**: ~55-65%
- **Generated names**: Sound natural and follow linguistic patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Technical Details

### Character Embedding
- Each character (a-z, '.') is mapped to a 10-dimensional vector
- Embeddings are randomly initialized and learned during training
- The '.' character serves as start/end token

### Sequence Processing
- Names are padded with '.' at start and end
- Model learns to predict next character from 3-character context
- Sliding window approach creates training sequences

### Generation Strategy
- Uses learned probability distributions for realistic first letters
- Applies heuristics to avoid unrealistic patterns (triple letters, too many vowels)
- Samples from top predictions with controlled randomness

## Troubleshooting

**Model not loading?**
- Ensure the model file exists in the correct path
- Check TensorFlow/Keras compatibility
- Try retraining the model

**Poor name quality?**
- Increase training epochs
- Adjust generation parameters
- Try different sampling strategies in the generation code

**Memory issues?**
- Reduce batch size
- Use smaller datasets for training
- Consider using model checkpointing

---

*This project demonstrates character-level neural language modeling for creative text generation.*