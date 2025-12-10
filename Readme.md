# From Single Neuron to Transformer

A comprehensive, hands-on journey through neural network architectures — implemented from scratch using only NumPy. No PyTorch, no TensorFlow — just pure Python and mathematics.

## Overview

This repository contains a series of Jupyter notebooks that progressively build up from a single neuron to a full Transformer architecture. Each notebook implements both **forward and backward passes manually**, giving you deep understanding of how these models actually work under the hood.

**Dataset:** Alice in Wonderland (character-level text generation)

## Notebooks

| # | Notebook | Architecture | Description |
|---|----------|--------------|-------------|
| 01a | `01a_single_neuron_classification.ipynb` | Single Neuron | Binary classification with a perceptron. Covers weights, bias, sigmoid activation, and gradient descent. |
| 01b | `01b_single_neuron_text_generation.ipynb` | Single Neuron | Character-level text generation using a single neuron. Introduction to sequence prediction. |
| 02 | `02_MLP_text_generation.ipynb` | Multi-Layer Perceptron | Adding hidden layers. Covers backpropagation through multiple layers, non-linear activations. |
| 03 | `03_RNN_text_generation.ipynb` | Vanilla RNN | Recurrent connections for sequential data. Implements Backpropagation Through Time (BPTT). |
| 04 | `04_LSTM_text_generation.ipynb` | LSTM | Long Short-Term Memory with forget, input, and output gates. Solves vanishing gradient problem. |
| 05 | `05_Attention_RNN_LSTM.ipynb` | Attention + LSTM/RNN | Attention mechanism on top of LSTM/RNN. Learn to focus on relevant parts of the sequence. |
| 06 | `06_Transformer_Text_Generation.ipynb` | Transformer (Decoder-Only) | Full transformer with multi-head self-attention, positional encoding, layer normalization, and feed-forward networks. |

## Architecture Progression

```
Single Neuron → MLP → RNN → LSTM → LSTM + Attention → Transformer
     ↓           ↓      ↓      ↓          ↓              ↓
  Basics    Depth  Sequence  Memory    Focus        Parallelism
```

## What You'll Learn

- **Single Neuron:** Weights, bias, activation functions, gradient descent
- **MLP:** Hidden layers, backpropagation, chain rule
- **RNN:** Recurrent connections, hidden states, BPTT, vanishing gradients
- **LSTM:** Gating mechanisms (forget, input, output), cell state, long-term memory
- **Attention:** Score functions, attention weights, context vectors, alignment
- **Transformer:** Self-attention, multi-head attention, positional encoding, layer normalization, residual connections

## Results Summary

| Model | Per-Character Loss | Notes |
|-------|-------------------|-------|
| Random Baseline | 3.83 | No learning |
| Single Neuron | ~3.0 | Limited capacity |
| MLP | ~2.5 | Better but no sequence modeling |
| Vanilla RNN | ~1.28 | Learns character patterns |
| LSTM | ~0.78 | Remembers longer context |
| LSTM + Attention | ~0.60 | Focuses on relevant history |
| Transformer | ~0.70 | Parallel processing, competitive with LSTM |

## Sample Generated Text

**Vanilla RNN:**
```
alice was in a little bother thing the was a little
```

**LSTM:**
```
alice was beginning to get very tired of sitting by her sister
```

**Transformer:**
```
alice had the mouse, and the queen said to the mock turtle
```

## Requirements

```
numpy
jupyter
matplotlib (optional, for visualizations)
```


## Repository Structure

```
├── data/
│   └── alice_in_wonderland.txt
├── 01a_single_neuron_classification.ipynb
├── 01b_single_neuron_text_generation.ipynb
├── 02_MLP_text_generation.ipynb
├── 03_RNN_text_generation.ipynb
├── 04_LSTM_text_generation.ipynb
├── 05_Attention_RNN_LSTM.ipynb
├── 06_Transformer_Text_Generation.ipynb
└── README.md
```


### All Models Include:
- Forward pass implementation
- Backward pass with manual gradient computation
- Training loop with loss tracking
- Text generation/sampling function

### No External Deep Learning Libraries
Everything is implemented using NumPy to ensure you understand:
- How gradients flow backward
- How weights are updated
- How each architecture component works mathematically