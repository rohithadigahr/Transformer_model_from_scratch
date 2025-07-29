# Transformer_model_from_scratch

## Overview
This is a complete implementation of a Transformer model built from scratch using PyTorch. The model is designed for character-level language modeling and includes all core Transformer components without using any high-level libraries like Hugging Face Transformers.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Installation & Setup

1. **Create project directory:**
```bash
mkdir transformer_from_scratch
cd transformer_from_scratch
```

2. **Create the folder structure:**
```bash
mkdir src data outputs logs
```

3. **Set up virtual environment:**
```bash
python -m venv transformer_env
source transformer_env/bin/activate  # On Windows: transformer_env\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Run the complete training and generation:**
```bash
python main.py
```

6. **For a quick demo (faster, smaller model):**
```bash
python main.py --demo
```

## Project Structure

```
transformer_from_scratch/
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters and configuration
│   ├── data_utils.py          # Dataset and utility functions
│   ├── transformer_model.py   # Complete Transformer architecture
│   ├── train.py              # Training logic and optimization
│   └── generate.py           # Text generation and sampling
├── data/
│   └── sample_text.txt       # Training data (created automatically)
├── outputs/
│   ├── generated_samples.txt # Generated text samples
│   ├── training_curves.png   # Loss and perplexity plots
│   └── transformer_checkpoint.pth # Saved model
├── logs/
│   └── training.log          # Training metrics log
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Model Architecture

### Core Components Implemented

1. **Scaled Dot-Product Attention**
   - Implements the fundamental attention mechanism: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V`
   - Includes optional masking for causal attention
   - Dropout for regularization

2. **Multi-Head Attention**
   - Parallel attention heads that capture different types of relationships
   - Linear projections for Q, K, V and output
   - Concatenation and final linear transformation

3. **Positional Encoding**
   - Sinusoidal positional encodings to inject sequence order information
   - Uses sine and cosine functions with different frequencies
   - Added to token embeddings before processing

4. **Feed Forward Networks**
   - Position-wise fully connected layers
   - ReLU activation between two linear transformations
   - Expansion factor of 4 (d_ff = 4 * d_model)

5. **Layer Normalization**
   - Applied after each sub-layer (attention and feed-forward)
   - Helps with training stability and convergence

6. **Encoder Stack**
   - 6 identical encoder layers
   - Each layer contains multi-head attention + feed-forward
   - Residual connections around each sub-layer

### Model Hyperparameters

```python
d_model = 256        # Model dimension
n_heads = 8          # Number of attention heads
n_layers = 6         # Number of encoder layers
d_ff = 1024          # Feed-forward dimension
seq_len = 64         # Maximum sequence length
dropout = 0.1        # Dropout rate
```

### Architecture Flow

```
Input Text → Token Embedding → Positional Encoding → 
Encoder Stack (6 layers) → Output Projection → Next Token Probabilities
```

Each encoder layer:
```
Input → Multi-Head Attention → Add & Norm → 
Feed Forward → Add & Norm → Output
```

## Training Dataset

**Dataset Type**: Character-level language modeling
**Source**: Shakespeare-like text (repeated for more training data)
**Task**: Predict the next character given a sequence of previous characters

**Sample Text**:
```
"To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles..."
```

**Dataset Statistics**:
- Vocabulary size: ~50-60 unique characters (letters, spaces, punctuation)
- Sequence length: 64 characters
- Training samples: ~10,000+ character sequences
- Character-to-index mapping for tokenization

## Loss Function & Optimizer

**Loss Function**: CrossEntropyLoss
- Standard choice for classification tasks
- Applied to predict next character from vocabulary

**Optimizer**: Adam
- Learning rate: 0.0001
- Default β1=0.9, β2=0.999 parameters
- No learning rate scheduling

**Training Configuration**:
- Batch size: 32
- Number of epochs: 100 (but i have trained for only 5 epochs as the loss was already less and training time was more)
- Gradient clipping: Not applied
- Causal masking: Applied for autoregressive generation

## Key Features

1. **Causal Attention**: Uses lower triangular mask to ensure each position can only attend to previous positions
2. **Autoregressive Generation**: Model generates text one character at a time
3. **Temperature Sampling**: Uses temperature scaling during generation for diversity
4. **Modular Design**: Each component is implemented as a separate class for clarity

## Training Process

1. **Data Preparation**: Convert text to character indices
2. **Batch Creation**: Create sliding window sequences of length 64
3. **Forward Pass**: Apply causal mask and compute next-token predictions
4. **Loss Calculation**: CrossEntropy between predictions and targets
5. **Backpropagation**: Update model parameters using Adam optimizer
6. **Generation**: Sample from learned distribution to generate new text

## Expected Outputs

**Training Logs**:
```
Vocabulary size: 44
Using device: cpu
Model parameters: 4,761,132
Epoch [1/5], Loss: 2.029, Perplexity: 7.60
Epoch [2/5], Loss: 1.29, Perplexity: 3.63
Epoch [3/5], Loss: 0.64, Perplexity: 1.
...
```

**Sample Generation**:
```
Generated text:
to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them...
```

## Model Performance

- **Perplexity**: Expected to decrease from 8 to <1 during training
- **Generation Quality**: Should produce coherent Shakespeare-like text
- **Training Time**: ~10-30 minutes depending on hardware
- **Parameters**: ~4.7M parameters

## Running the Code

```bash
python transformer_model.py
```

The script will:
1. Initialize the model with specified hyperparameters
2. Create the character-level dataset
3. Train for 100 epochs with progress updates
4. Generate sample text using the trained model
5. Print training statistics and architecture details

## Technical Notes

- **Memory Usage**: ~2-4GB GPU memory for training
- **Causality**: Implemented through attention masking, not architectural changes
- **Vocabulary**: All unique characters in the input text
- **Sequence Padding**: Not required due to fixed-length sequences
- **Evaluation**: Uses generation quality as primary metric

## Future Improvements

1. Add learning rate scheduling (cosine annealing, warmup)
2. Implement beam search for better generation
3. Add positional embeddings as alternative to sinusoidal encoding
4. Implement decoder architecture for sequence-to-sequence tasks
5. Add attention visualization tools
6. Implement different attention variants (sparse, local, etc.)
