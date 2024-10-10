import torch
import torch.nn as nn

from TransformerBlock import TransformerBlock


class Encoder(nn.Module):
    """
    The Encoder class represents the encoder part of the Transformer model, which
    processes input sequences to generate an encoded representation. The encoder
    consists of multiple layers, each containing self-attention mechanisms and
    feed-forward networks with positional and word embeddings.

    Attributes:
        embed_size (int): The size of the input embedding vector.
        device (str): The device (CPU/GPU) on which computations are performed.
        word_embedding (nn.Embedding): The word embedding layer that maps input tokens to dense vectors.
        position_embedding (nn.Embedding): The positional embedding layer that adds positional information to tokens.
        layers (nn.ModuleList): A list of Transformer blocks, each consisting of self-attention and feed-forward layers.
        dropout (nn.Dropout): Dropout layer to prevent overfitting during training.
    """

    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        """
        Initializes the Encoder with the specified parameters, including the number
        of layers, heads for multi-head attention, and embeddings for words and positions.

        Args:
            src_vocab_size (int): The size of the source vocabulary (number of unique tokens).
            embed_size (int): The size of the embedding vector for each token.
            num_layers (int): The number of Transformer blocks in the encoder.
            heads (int): The number of attention heads for multi-head self-attention.
            device (str): The device (CPU/GPU) to use for computation.
            forward_expansion (int): The expansion factor for the hidden dimension in the feed-forward network.
            dropout (float): Dropout rate to be applied for regularization.
            max_length (int): The maximum length of the input sequences.
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Word embedding layer that converts input tokens to dense vectors
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # Positional embedding layer to add positional information to token embeddings
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the Encoder. Processes input tokens through embedding layers
        and multiple Transformer blocks, applying self-attention and feed-forward layers.

        Args:
            x (Tensor): Input sequence of token indices (N, seq_length).
            mask (Tensor): Mask to prevent attending to padding positions (N, 1, 1, seq_length).

        Returns:
            Tensor: The encoded representation of the input sequence (N, seq_length, embed_size).
        """
        N, seq_length = x.shape  # Batch size (N) and sequence length

        # Create positional indices and expand to match the batch size
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Add word and positional embeddings, then apply dropout
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # Pass the input through each Transformer block
        for layer in self.layers:
            # In the encoder, query, key, and value are the same (the input sequence itself)
            out = layer(out, out, out, mask)

        return out
