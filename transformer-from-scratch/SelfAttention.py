import torch
import torch.nn as nn

"""
What is Self-Attention?
SelfAttention Class: This class represents a multi-head self-attention mechanism that splits the input embedding into multiple heads, applies attention to each head, and concatenates the results.

__init__ Method: 
Initializes the class by setting up the necessary attributes, including the embedding size, number of heads, and the linear transformations (values, keys, queries, fc_out).

forward Method:
Values, Keys, and Queries: The input is transformed into three components (values, keys, and queries) using linear layers.
Head Splitting: The input is split into multiple heads for parallel processing, which allows the model to focus on different parts of the input.
Attention Scores: Attention is computed using a scaled dot-product between queries and keys, followed by a softmax function to obtain attention weights.
Masking: Optional masking is applied to avoid attending to padded positions or unwanted tokens.
Output Calculation: The attention weights are applied to the values, and the final result is passed through a linear layer to return to the original embedding size.

This structure helps in efficiently computing the attention mechanism and is the core component of Transformer models.
"""


class SelfAttention(nn.Module):
    """
    SelfAttention Module implements the scaled dot-product attention mechanism
    with multi-head attention. It takes in queries, keys, and values and returns
    a weighted sum of the values based on the attention scores.

    Attributes:
        embed_size (int): The size of the input embedding vectors.
        heads (int): The number of attention heads to split the embedding into.
        head_dim (int): The size of the split embedding for each attention head.
        values (nn.Linear): A linear transformation applied to the value vectors.
        keys (nn.Linear): A linear transformation applied to the key vectors.
        queries (nn.Linear): A linear transformation applied to the query vectors.
        fc_out (nn.Linear): A linear transformation applied to the concatenated output of all attention heads.
    """

    def __init__(self, embed_size, heads):
        """
        Initialize the SelfAttention module with the specified embedding size and number of heads.

        Args:
            embed_size (int): The size of the input embedding.
            heads (int): The number of attention heads to split the embedding into.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        # Split embedding size across heads

        # Ensure that the embedding size is divisible by the number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embed size must be divisible by heads"

        # Linear layers to transform the input values, keys, and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Output linear layer to project the concatenated heads back to embed_size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Perform the forward pass of the self-attention mechanism.

        Args:
            values (Tensor): Input values (N, value_len, embed_size).
            keys (Tensor): Input keys (N, key_len, embed_size).
            query (Tensor): Input queries (N, query_len, embed_size).
            mask (Tensor): Optional mask to avoid attending to padding positions (N, 1, 1, key_len).

        Returns:
            Tensor: Output of the self-attention layer (N, query_len, embed_size).
        """
        N = query.shape[0]  # Number of samples in the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Linear transformations of values, keys, and queries
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into multiple heads for parallel computation
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, value_len, self.heads, self.head_dim)

        # Compute attention scores (energy) using dot product between queries and keys
        # Shape of energy: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Apply mask to avoid attending to certain positions (e.g., padding)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax to get the attention weights, scaled by the square root of embedding size
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # Shape of attention: (N, heads, query_len, key_len)

        # Multiply attention weights with value vectors to get the final output
        # Shape of output: (N, query_len, heads, head_dim), then reshaped
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Apply the final linear transformation to concatenate all heads and project
        # the output back to the original embedding size
        out = self.fc_out(out)
        # (N, query_len, embed_size)

        return out
