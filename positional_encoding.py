import math
import torch
import torch.nn as nn
from torch import Tensor
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      probability of dropout applied after adding the positional encodings (regularization)
          max_length:   maximum sequence length supported by the precomputed encoding table (default 5000)
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # perform dropout
        return self.dropout(x)