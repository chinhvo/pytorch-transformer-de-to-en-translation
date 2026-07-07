import math
import torch
import torch.nn as nn
from torch import Tensor
#Maps token IDs → dense vectors (nn.Embedding).
#Scales the result by √d_model, per Vaswani et al. Section 3.4.
#Exactly what’s used in the Transformer Encoder-Decoder.

class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
          vocab_size:     size of vocabulary
          d_model:        dimension of embeddings
        """
        # inherit from nn.Module
        super().__init__()

        # embedding look-up table (lut)
        # Example: "house" → [0.23, -0.18, ..., 0.91].
        self.lut = nn.Embedding(vocab_size, d_model)

        # dimension of embeddings
        self.d_model = d_model

    def forward(self, x: Tensor):
        """
        Args:
          x:              input Tensor (batch_size, seq_length)

        Returns:
                          embedding vector
        """
        # embeddings by constant sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)