import torch
import torch.nn as nn
from torch import Tensor
from multi_head_attention import MultiHeadAttention
from positionwise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()
        # multi-head attention sublayer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for multi-head attention
        self.attn_layer_norm = nn.LayerNorm(d_model)

        # position-wise feed-forward network
        self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        # layer norm for position-wise ffn
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor):
        """
        Args:
            src:          positionally embedded sequences   (batch_size, seq_length, d_model)
            src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)
        Returns:
            src:          sequences after self-attention    (batch_size, seq_length, d_model)
        """
        # pass embeddings through multi-head attention
        _src, attn_probs = self.attention(src, src, src, src_mask)

        # residual add and norm
        src = self.attn_layer_norm(src + self.dropout(_src))

        # position-wise feed-forward network
        _src = self.positionwise_ffn(src)

        # residual add and norm
        src = self.ffn_layer_norm(src + self.dropout(_src))

        return src, attn_probs


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int,
                 n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            d_model:      dimension of embeddings
            n_layers:     number of encoder layers
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()

        # create n_layers encoders
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropout)
                                     for layer in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor):
        """
        Args:
            src:          embedded sequences                (batch_size, seq_length, d_model)
            src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)

        Returns:
            src:          sequences after self-attention    (batch_size, seq_length, d_model)
        """

        # pass the sequences through each encoder
        for layer in self.layers:
            src, attn_probs = layer(src, src_mask)

        self.attn_probs = attn_probs

        return src