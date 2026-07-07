import torch
import torch.nn as nn
from torch import Tensor
from encoder_layer import Encoder
from decoder_layer import Decoder
from embeddings import Embeddings

class Transformer(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: Embeddings,
            trg_embed: Embeddings,
            src_pad_idx: int,
            trg_pad_idx: int, device):
        """
        Args:
            encoder:      encoder stack
            decoder:      decoder stack
            src_embed:    source embeddings and encodings
            trg_embed:    target embeddings and encodings
            src_pad_idx:  padding index
            trg_pad_idx:  padding index
            device:       cuda or cpu
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src: Tensor):
        """
        Args:
            src:          raw sequences with padding        (batch_size, seq_length)

        Returns:
            src_mask:     mask for each sequence            (batch_size, 1, 1, seq_length)
        """
        # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg: Tensor):
        """
        Args:
            trg:          raw sequences with padding        (batch_size, seq_length)

        Returns:
            trg_mask:     mask for each sequence            (batch_size, 1, seq_length, seq_length)
        """

        seq_length = trg.shape[1]

        # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)

        # generate subsequent mask
        trg_sub_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=self.device)).bool()  # (batch_size, 1, seq_length, seq_length)

        # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
        trg_mask = trg_mask & trg_sub_mask

        return trg_mask

    def forward(self, src: Tensor, trg: Tensor):
        """
        Args:
            trg:          raw target sequences              (batch_size, trg_seq_length)
            src:          raw src sequences                 (batch_size, src_seq_length)

        Returns:
            output:       sequences after decoder           (batch_size, trg_seq_length, output_dim)
        """

        # create source and target masks
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, src_seq_length)
        trg_mask = self.make_trg_mask(trg)  # (batch_size, 1, trg_seq_length, trg_seq_length)

        # push the src through the encoder layers
        src = self.encoder(self.src_embed(src), src_mask)  # (batch_size, src_seq_length, d_model)

        # decoder output and attention probabilities
        output = self.decoder(self.trg_embed(trg), src, trg_mask, src_mask)

        return output