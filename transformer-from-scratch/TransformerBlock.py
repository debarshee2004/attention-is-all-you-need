import torch.nn as nn
from SelfAttention import SelfAttention

"""
TransformerBlock Class: This class represents one block of the Transformer architecture, which consists of a multi-head self-attention mechanism, followed by a feed-forward neural network. Each of these components is followed by layer normalization and skip (residual) connections.

Attributes:
attention: This is the self-attention mechanism that computes the attention scores and attends to different parts of the input sequence.
norm1 and norm2: Layer normalization is used to stabilize training and prevent overfitting. It's applied after the self-attention and feed-forward layers.
feed_forward: A feed-forward neural network consisting of two linear layers with an expansion in hidden dimensions for more complex transformations.
dropout: Dropout is applied after both normalization steps to prevent overfitting during training.

__init__ Method:
Initializes the self-attention mechanism, feed-forward network, and other components like layer normalization and dropout.
The forward_expansion argument controls the hidden layer size in the feed-forward network.
forward Method:

Attention: Performs multi-head self-attention on the input values, keys, and queries.
Residual Connection + Normalization (1st): Adds the original query to the attention output (residual connection), normalizes it using LayerNorm, and applies dropout.
Feed-Forward Network: Passes the result through a feed-forward network that expands and then reduces the dimension of the input.
Residual Connection + Normalization (2nd): Adds the original input of the feed-forward network to the output (residual connection), normalizes it, and applies dropout again.

This structure forms a core building block of Transformer models like BERT and GPT, enabling them to capture complex dependencies in the input sequence.
"""


class TransformerBlock(nn.Module):
    """
    TransformerBlock implements a single block of the Transformer model, which
    includes multi-head self-attention, layer normalization, and a feed-forward
    neural network with residual (skip) connections and dropout.

    Attributes:
        attention (SelfAttention): The self-attention mechanism that computes attention scores.
        norm1 (nn.LayerNorm): Layer normalization applied after the self-attention.
        norm2 (nn.LayerNorm): Layer normalization applied after the feed-forward network.
        feed_forward (nn.Sequential): A two-layer feed-forward network with ReLU activation.
        dropout (nn.Dropout): Dropout layer to prevent overfitting during training.
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        Initialize the Transformer block with self-attention, layer normalization,
        feed-forward layers, and dropout.

        Args:
            embed_size (int): Size of the input embedding vector.
            heads (int): Number of attention heads for the self-attention mechanism.
            dropout (float): Dropout probability to be applied to the layers.
            forward_expansion (int): Expansion factor for the hidden dimension in the feed-forward network.
        """
        super(TransformerBlock, self).__init__()

        # Multi-head self-attention layer
        self.attention = SelfAttention(embed_size, heads)

        # Layer normalization applied after self-attention and feed-forward layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward neural network with an expansion factor
        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_size, forward_expansion * embed_size
            ),  # First linear layer expands the dimension
            nn.ReLU(),  # Non-linear activation function
            nn.Linear(
                forward_expansion * embed_size, embed_size
            ),  # Second linear layer projects it back
        )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Perform the forward pass of the Transformer block.

        Args:
            value (Tensor): Input value vectors (N, value_len, embed_size).
            key (Tensor): Input key vectors (N, key_len, embed_size).
            query (Tensor): Input query vectors (N, query_len, embed_size).
            mask (Tensor): Optional mask to avoid attending to padding positions (N, 1, 1, key_len).

        Returns:
            Tensor: Output of the Transformer block (N, query_len, embed_size).
        """
        # Perform multi-head self-attention
        attention = self.attention(value, key, query, mask)

        # Add residual connection (skip connection) and apply layer normalization followed by dropout
        # Normalizing the sum of attention output and original query
        x = self.dropout(self.norm1(attention + query))

        # Apply the feed-forward network to the normalized output
        forward = self.feed_forward(x)

        # Add residual connection for the feed-forward output, followed by normalization and dropout
        out = self.dropout(self.norm2(forward + x))

        return out
