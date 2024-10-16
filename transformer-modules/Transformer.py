import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    """
    The Transformer model is an attention-based sequence-to-sequence model
    consisting of an encoder and a decoder. It applies self-attention and
    cross-attention mechanisms to model dependencies between the input (source)
    and output (target) sequences.

    Attributes:
        encoder (Encoder): Encoder model to process the source sequence.
        decoder (Decoder): Decoder model to generate the target sequence.
        src_pad_idx (int): Padding index for the source sequence.
        trg_pad_idx (int): Padding index for the target sequence.
        device (str): Device on which the model is run (CPU or GPU).
    """

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        """
        Initializes the Transformer model with an encoder and a decoder, each
        consisting of multiple layers. The model uses self-attention and
        cross-attention mechanisms to model relationships between tokens
        in the source and target sequences.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            trg_vocab_size (int): Size of the target vocabulary.
            src_pad_idx (int): Padding index for the source sequence.
            trg_pad_idx (int): Padding index for the target sequence.
            embed_size (int): Dimensionality of the embedding vector. Default is 512.
            num_layers (int): Number of layers in both the encoder and decoder. Default is 6.
            forward_expansion (int): Expansion factor for the feed-forward networks. Default is 4.
            heads (int): Number of attention heads in multi-head attention. Default is 8.
            dropout (float): Dropout rate for regularization. Default is 0.
            device (str): Device on which the model runs (CPU or GPU). Default is "cpu".
            max_length (int): Maximum length of the source/target sequences. Default is 100.
        """
        super(Transformer, self).__init__()

        # Initialize the encoder, which will process the source sequence
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        # Initialize the decoder, which will generate the target sequence
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        # Padding indices for the source and target sequences
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to avoid attending to padding tokens.

        Args:
            src (Tensor): Source sequence tensor of shape (N, src_len) where N is the batch size
                          and src_len is the length of the source sequence.

        Returns:
            Tensor: A mask tensor of shape (N, 1, 1, src_len) where padding tokens are masked.
        """
        # Create a mask where padding tokens (src_pad_idx) are set to 0, others to 1
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # Mask shape: (N, 1, 1, src_len) - this will be used in self-attention to avoid attending to pad tokens
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        Creates a mask for the target sequence to prevent attending to future tokens during training.

        Args:
            trg (Tensor): Target sequence tensor of shape (N, trg_len) where N is the batch size
                          and trg_len is the length of the target sequence.

        Returns:
            Tensor: A mask tensor of shape (N, 1, trg_len, trg_len) where future tokens are masked.
        """
        N, trg_len = trg.shape
        # Create a lower triangular matrix for masking future tokens in the target sequence (causal mask)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # trg_mask shape: (N, 1, trg_len, trg_len) - used for masked self-attention in the decoder
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        """
        Forward pass through the Transformer model. Processes the source sequence through the encoder,
        generates the target sequence through the decoder using self-attention and cross-attention,
        and returns the final output.

        Args:
            src (Tensor): Source sequence tensor of shape (N, src_len) where N is the batch size.
            trg (Tensor): Target sequence tensor of shape (N, trg_len) where N is the batch size.

        Returns:
            Tensor: Output tensor of shape (N, trg_len, trg_vocab_size) representing
                    the predicted next tokens in the target sequence.
        """
        # Generate the source mask to ignore padding tokens in the source sequence
        src_mask = self.make_src_mask(src)

        # Generate the target mask to prevent the model from attending to future tokens in the target sequence
        trg_mask = self.make_trg_mask(trg)

        # Encode the source sequence
        enc_src = self.encoder(src, src_mask)

        # Decode the target sequence using the encoded source and the generated masks
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Example input tensors for source and target sequences
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # Define model parameters
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    # Initialize the Transformer model
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device
    ).to(device)

    # Perform a forward pass with the source and target sequences (shifted right)
    out = model(x, trg[:, :-1])

    print(out.shape)  # Output tensor shape will be (N, trg_len, trg_vocab_size)
    # OUTPUT: torch.Size([2, 7, 10])
    # The output tensor represents the predicted next tokens in the target sequence
