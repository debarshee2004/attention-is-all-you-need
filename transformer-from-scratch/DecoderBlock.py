import torch.nn as nn

from SelfAttention import SelfAttention
from TransformerBlock import TransformerBlock


class DecoderBlock(nn.Module):
    """
    DecoderBlock is a component of the Transformer model's decoder.
    It includes masked self-attention, cross-attention with the encoder's output,
    and a feed-forward network with residual connections and normalization layers.

    Attributes:
        norm (nn.LayerNorm): Layer normalization applied after the masked self-attention.
        attention (SelfAttention): Masked self-attention mechanism for the decoder's input.
        transformer_block (TransformerBlock): A transformer block that includes cross-attention and a feed-forward network.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        """
        Initialize the DecoderBlock with masked self-attention, cross-attention,
        normalization layers, and dropout for regularization.

        Args:
            embed_size (int): The size of the input embedding vector.
            heads (int): The number of attention heads for multi-head self-attention.
            forward_expansion (int): Expansion factor for the hidden dimension in the feed-forward network.
            dropout (float): Dropout rate to apply to layers for regularization.
            device (str): Device on which the model will run (CPU or GPU).
        """
        super(DecoderBlock, self).__init__()

        # Layer normalization to stabilize training and prevent overfitting
        self.norm = nn.LayerNorm(embed_size)

        # Masked self-attention for the decoder
        self.attention = SelfAttention(embed_size, heads=heads)

        # Transformer block which contains cross-attention (using encoder output as keys/values)
        # and feed-forward layers
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        Forward pass of the DecoderBlock. It applies masked self-attention to the input, followed
        by cross-attention with the encoder output, and finally a feed-forward network.

        Args:
            x (Tensor): Target input sequence (N, trg_len, embed_size).
            value (Tensor): Encoder output values (N, src_len, embed_size).
            key (Tensor): Encoder output keys (N, src_len, embed_size).
            src_mask (Tensor): Mask for the source sequence to prevent attending to padding tokens.
            trg_mask (Tensor): Mask for the target sequence to prevent attending to future tokens (causal masking).

        Returns:
            Tensor: Output tensor after applying masked self-attention and cross-attention
                    (N, trg_len, embed_size).
        """
        # Apply masked self-attention on the target sequence
        # In masked self-attention, tokens cannot attend to future tokens in the sequence.
        attention = self.attention(x, x, x, trg_mask)

        # Add residual connection (attention output + input) and normalize
        query = self.dropout(self.norm(attention + x))

        # Apply cross-attention between the query (decoder's current state) and the encoder's output (value, key)
        # The query is now the result of masked self-attention on the target sequence
        out = self.transformer_block(value, key, query, src_mask)

        return out
