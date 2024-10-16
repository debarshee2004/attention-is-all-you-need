import torch
import torch.nn as nn

from DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    """
    Decoder is a part of the Transformer model that generates the target sequence
    based on the encoder's output and the previously generated tokens in the target sequence.

    Attributes:
        word_embedding (nn.Embedding): Embedding layer for the target vocabulary.
        position_embedding (nn.Embedding): Embedding layer for positional information.
        layers (nn.ModuleList): A list of DecoderBlock layers that apply self-attention
                                and cross-attention.
        fc_out (nn.Linear): A fully connected layer that maps the output of the last decoder block
                            to the target vocabulary size for final token prediction.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """

    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        """
        Initializes the Decoder with word embeddings, positional embeddings,
        multiple decoder blocks, and a final output layer.

        Args:
            trg_vocab_size (int): Size of the target vocabulary.
            embed_size (int): Dimensionality of the embedding vector.
            num_layers (int): Number of DecoderBlock layers in the decoder.
            heads (int): Number of attention heads for multi-head self-attention.
            forward_expansion (int): Expansion factor for the feed-forward network in each block.
            dropout (float): Dropout rate to apply to layers for regularization.
            device (str): Device on which the model will run (CPU or GPU).
            max_length (int): Maximum length of the target sequence for positional encoding.
        """
        super(Decoder, self).__init__()

        self.device = device

        # Embedding layer for target tokens
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        # Positional embedding to add sequence order information to the input embeddings
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of DecoderBlock layers, each consisting of masked self-attention,
        # cross-attention with the encoder's output, and feed-forward layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        # Fully connected output layer to predict the next token in the sequence
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        Forward pass of the Decoder. Applies word embeddings and positional embeddings
        to the target input sequence, processes it through multiple decoder layers,
        and finally generates predictions using a fully connected layer.

        Args:
            x (Tensor): Input tensor of target sequence (N, trg_len) where N is the batch size
                        and trg_len is the length of the target sequence.
            enc_out (Tensor): Output from the encoder (N, src_len, embed_size).
            src_mask (Tensor): Mask for the source sequence to prevent attending to padding tokens.
            trg_mask (Tensor): Mask for the target sequence to prevent attending to future tokens.

        Returns:
            Tensor: Output tensor with predictions for the next token (N, trg_len, trg_vocab_size).
        """
        # Get the batch size (N) and the length of the target sequence (trg_len)
        N, seq_length = x.shape

        # Create positional encodings based on the sequence length and add them to the word embeddings
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Pass through each DecoderBlock layer, where cross-attention happens with encoder output
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # Final fully connected layer to map the decoder's output to the target vocabulary size
        out = self.fc_out(x)

        return out
