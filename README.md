# Custom GPT Implementation from Scratch

A minimal implementation of a GPT-like transformer model built from scratch using PyTorch, trained on the Cornell Movie Dialogs Corpus.

## Overview

This project implements a simplified version of GPT (Generative Pre-trained Transformer) architecture from the ground up. It includes:

- Custom multi-head attention mechanism
- Positional embeddings
- Feed-forward networks with residual connections
- BPE (Byte Pair Encoding) tokenization
- Causal language modeling training
- Text generation with temperature and top-k sampling

## Dependencies

You'll need to install the following packages:

```bash
pip install torch torchvision torchaudio
pip install requests beautifulsoup4
pip install tokenizers
pip install tqdm
```

Or install all at once:

```bash
pip install torch requests beautifulsoup4 tokenizers tqdm
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch requests beautifulsoup4 tokenizers tqdm
```

### 2. Run the Script

```bash
python main.py
```

The script will automatically:
1. Download the Cornell Movie Dialogs dataset
2. Preprocess and tokenize the text
3. Train a custom GPT model
4. Generate sample text

## Files Created

When you run the script, it will create:

```
├── cornell_movie_dialogs_corpus.zip    # Downloaded dataset
├── cornell movie-dialogs corpus/       # Extracted dataset folder
├── corpus.txt                         # Preprocessed text file
└── paste.py                          # Main script
```

## Architecture Details

### Model Components

1. **Token Embedding**: Maps vocabulary tokens to dense vectors
2. **Positional Embedding**: Adds position information to tokens
3. **Multi-Head Attention**: Implements scaled dot-product attention with causal masking
4. **Feed-Forward Network**: Two-layer MLP with ReLU activation
5. **Layer Normalization**: Stabilizes training
6. **Residual Connections**: Enables deeper networks

### Model Specifications

- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Attention Heads**: 2
- **Vocabulary Size**: 5,000 (BPE tokens)
- **Context Window**: Dynamic (limited to 128 tokens during generation)

## Training Configuration

```python
# Training Parameters
epochs = 200
learning_rate = 1e-4
optimizer = AdamW
loss_function = CrossEntropyLoss
```

## Data Processing

### Tokenization
- **Method**: Byte Pair Encoding (BPE)
- **Vocabulary Size**: 5,000 tokens
- **Special Tokens**: `[UNK]`, `[CLS]`, `[SEP]`

## Text Generation

The model supports text generation with:

- **Temperature Sampling**: Controls randomness (higher = more creative)
- **Top-k Sampling**: Limits selection to top k most likely tokens
- **Configurable Length**: Specify maximum number of tokens to generate

### Example Usage

```python
# Generate text starting with "the"
generated_text = generate(
    model=model,
    start_text="the",
    temperature=1.0,
    top_k=4,
    max_new_tokens=40
)
print(generated_text)
```

## Customization Options

### Model Architecture

```python
model = multiheadgpt(
    embed_dim=512,      # Increase embedding dimension
    hidden_dim=1024,    # Increase hidden layer size
    num_heads=8,        # More attention heads
    max_len=len(text)   # Context window size
)
```

### Adjust Training Parameters

```python
epochs = 500              # More training epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Lower learning rate
```

### Change Generation Settings

```python
generate(
    model=model,
    start_text="hello world",
    temperature=0.8,    # Less randomness
    top_k=10,          # More token choices
    max_new_tokens=100  # Longer generation
)
```

## Training Process

1. **Data Loading**: Downloads and preprocesses Cornell Movie Dialogs
2. **Tokenization**: Creates BPE tokenizer with 5K vocabulary
3. **Model Training**: 200 epochs with progress tracking
4. **Loss Monitoring**: Prints loss every 20 epochs
5. **Text Generation**: Demonstrates model capabilities

## Code Structure

### Main Components

1. **`load_cornell_movie_dialogs_words()`**: Dataset loading and preprocessing
2. **`FeedForward`**: MLP component of transformer
3. **`MultiHeadAttention`**: Self-attention mechanism with causal masking
4. **`multiheadgpt`**: Main transformer model class
5. **`generate()`**: Text generation function

### Key Features

- **Causal Masking**: Prevents the model from looking at future tokens
- **Residual Connections**: Improves gradient flow
- **Layer Normalization**: Stabilizes training
- **Positional Embeddings**: Provides sequence order information

## Learning Objectives

This implementation demonstrates:

- How transformers work at a fundamental level
- Multi-head attention mechanisms
- Causal language modeling
- Custom tokenization with BPE
- Text generation techniques
- PyTorch neural network implementation

## Limitations

- **Small Model**: Only 2 attention heads and 256 embedding dimensions
- **Limited Data**: Truncated to 100K characters for speed
- **No Validation**: Training without validation split
- **Basic Generation**: Simple top-k sampling without advanced techniques

## Troubleshooting

**CUDA Out of Memory**: Reduce model dimensions or batch size

**Download Issues**: Check internet connection for dataset download

**Import Errors**: Ensure all dependencies are installed correctly

**Slow Training**: Consider using GPU acceleration if available
