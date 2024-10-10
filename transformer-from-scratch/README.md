# Transformer Model and Self-Attention: Detailed Explanation

This document provides a comprehensive explanation of the **Transformer** model and the **Self-Attention** mechanism, covering all critical concepts in the architecture, including how they function and their implementation in the code provided.

**Topics covered include:**

- [Transformer Model and Self-Attention: Detailed Explanation](#transformer-model-and-self-attention-detailed-explanation)
  - [1. **Introduction to Transformers**](#1-introduction-to-transformers)
  - [2. **Self-Attention Mechanism**](#2-self-attention-mechanism)
  - [3. **Multi-Head Attention**](#3-multi-head-attention)
  - [4. **Positional Encoding**](#4-positional-encoding)
  - [5. **Encoder-Decoder Architecture**](#5-encoder-decoder-architecture)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
  - [6. **Masks in Transformers**](#6-masks-in-transformers)
    - [**Source Mask (Encoder Mask)**](#source-mask-encoder-mask)
    - [**Target Mask (Decoder Mask)**](#target-mask-decoder-mask)
  - [7. **Feed-Forward Networks**](#7-feed-forward-networks)
  - [8. **Skip Connections and Layer Normalization**](#8-skip-connections-and-layer-normalization)
  - [9. **Transformer Block Structure**](#9-transformer-block-structure)
  - [10. **Application in Sequence-to-Sequence Tasks**](#10-application-in-sequence-to-sequence-tasks)
  - [11. **Code Breakdown**](#11-code-breakdown)
  - [Conclusion](#conclusion)

## 1. **Introduction to Transformers**

The **Transformer** model is a sequence-to-sequence (Seq2Seq) deep learning architecture originally designed for tasks such as machine translation, but its design has made it applicable to a wide range of NLP and computer vision tasks. Unlike recurrent models like RNNs and LSTMs, the Transformer relies entirely on attention mechanisms to model dependencies within and between sequences.

Transformers are composed of two primary components:

- **Encoder**: Processes the input (source) sequence and generates a contextual representation.
- **Decoder**: Uses this encoded representation to generate the output (target) sequence by predicting one token at a time.

**Key Features:**

- **Self-Attention**: A mechanism to model the relationships between words in a sequence.
- **Parallelization**: Unlike RNNs, Transformers can process all tokens in a sequence simultaneously, leading to faster training.
- **Positional Encoding**: Since Transformers have no inherent understanding of sequence order, they use positional encodings to retain order information.

---

## 2. **Self-Attention Mechanism**

Self-attention is the core idea behind Transformers. It allows the model to weigh the importance of different tokens in a sequence relative to each other, enabling it to capture long-range dependencies in the sequence.

**How It Works:**

For each token in the input sequence:

- **Query $(Q)$**: A vector representation of the token in focus.
- **Key $(K)$**: A vector for each token used to calculate attention scores.
- **Value $(V)$**: The actual information of each token that contributes to the final representation.

The attention score between two tokens is computed by taking the dot product between the query of the current token and the keys of the other tokens. These scores are normalized using the **softmax** function to ensure that they sum to 1. The final output is a weighted sum of the values, where the weights are the attention scores.

**Formula:**

The output of self-attention is calculated as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

- $(Q)$: Query
- $(K)$: Key
- $(V)$: Value
- $(d_k)$: Dimension of the key vectors

---

## 3. **Multi-Head Attention**

In practice, instead of calculating a single self-attention score, the Transformer uses **multi-head attention** to capture information from multiple subspaces. Multi-head attention splits the embedding into multiple heads and applies the self-attention mechanism independently for each head. The results from all heads are then concatenated and projected back to the original embedding size.

This allows the model to learn relationships at different levels of abstraction.

**Formula:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, \dots, \text{head}\_h)W^O
$$

Where each head is computed as:

$$
\text{head}\_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

---

## 4. **Positional Encoding**

Transformers do not have a built-in way to handle the order of tokens in a sequence since self-attention treats all tokens equally regardless of their position. **Positional Encoding** is added to the input embeddings to inject information about the relative positions of the tokens.

**Formula:**

The positional encoding is calculated using sinusoidal functions:

$$
PE*{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d*{\text{model}}}}\right)
$$

$$
PE*{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d*{\text{model}}}}\right)
$$

Where `pos` is the position and `i` is the dimension.

---

## 5. **Encoder-Decoder Architecture**

The **Encoder** and **Decoder** form the backbone of the Transformer architecture. The encoder processes the input sequence, while the decoder generates the output sequence, one token at a time, using both the encoder's output and the previous tokens in the target sequence.

### Encoder

The encoder is made up of multiple layers, each containing:

- **Self-Attention**: Each token attends to all other tokens in the sequence.
- **Feed-Forward Network**: After attention, a feed-forward neural network processes the attention output.
- **Layer Normalization and Skip Connections**: Residual connections help in training deeper models by allowing gradients to flow through the network more easily.

### Decoder

The decoder has a similar structure but includes **Masked Self-Attention** to ensure that the model does not look at future tokens when generating text.

---

## 6. **Masks in Transformers**

### **Source Mask (Encoder Mask)**

- Masks are used to prevent the model from attending to padding tokens in the input sequence.
- The source mask is a binary mask where the padding positions are set to 0 and valid positions are set to 1.

### **Target Mask (Decoder Mask)**

- The target mask (also known as the **causal mask**) is used in the decoder to prevent attending to future tokens. This is essential for autoregressive generation, where the model predicts tokens one at a time.
- The target mask is a lower triangular matrix, ensuring that the model can only attend to past tokens and the current token.

---

## 7. **Feed-Forward Networks**

Each layer of the encoder and decoder contains a **Feed-Forward Network (FFN)** that processes the attention output. The FFN is a simple two-layer fully connected network applied to each position independently.

**Formula:**

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

The **ReLU** activation adds non-linearity, and the **expansion factor** (commonly 4) increases the model’s capacity by expanding the dimensionality in the hidden layer.

---

## 8. **Skip Connections and Layer Normalization**

Each layer in the Transformer includes **skip connections** (also called residual connections) which add the input of a layer directly to its output. This helps mitigate the vanishing gradient problem and allows for deeper networks to be trained.

After each sub-layer (self-attention or feed-forward), **layer normalization** is applied to stabilize and accelerate training.

**Formula:**

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

---

## 9. **Transformer Block Structure**

A **Transformer Block** is the fundamental building block of the encoder and decoder. It consists of:

1. **Multi-Head Self-Attention Layer**.
2. **Feed-Forward Neural Network**.
3. **Skip Connections and Layer Normalization**.

Each block allows the model to attend to all tokens in the sequence simultaneously and compute representations of the sequence at various levels of abstraction.

---

## 10. **Application in Sequence-to-Sequence Tasks**

The Transformer is widely used in tasks such as:

- **Machine Translation**: Translating text from one language to another.
- **Text Summarization**: Generating concise summaries of documents.
- **Question Answering**: Generating answers based on a given context.
- **Text Generation**: Generating coherent and contextually relevant text.

In these tasks, the encoder processes the input sequence and the decoder generates the output sequence, attending to both the input and the tokens it has generated so far.

---

## 11. **Code Breakdown**

**Example of a Transformer Class Implementation:**

```py
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
```

This code defines a Transformer model with two main components: an encoder and a decoder. Masks are used to handle padding tokens in the input and prevent the decoder from looking at future tokens in the output sequence.

---

## Conclusion

The Transformer model revolutionized the way we approach sequence-to-sequence tasks in NLP by replacing recurrence with attention. Self-attention enables the model to attend to all tokens in a sequence regardless of their position, making it highly effective for long-range dependencies. By applying multi-head attention, feed-forward layers, and residual connections, the Transformer achieves state-of-the-art performance in a wide range of applications.
