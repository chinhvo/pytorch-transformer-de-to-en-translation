import torch
import torch.nn as nn
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            d_model:      dimension of embeddings
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x:            output from attention (batch_size, seq_length, d_model)

        Returns:
            expanded-and-contracted representation (batch_size, seq_length, d_model)
        """
        # w_1(x).relu(): (batch_size, seq_length, d_model) x (d_model,d_ffn) -> (batch_size, seq_length, d_ffn)
        # w_2(w_1(x).relu()): (batch_size, seq_length, d_ffn) x (d_ffn, d_model) -> (batch_size, seq_length, d_model)
        return self.w_2(self.dropout(self.w_1(x).relu()))