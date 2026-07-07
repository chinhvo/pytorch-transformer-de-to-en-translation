import torch
import torch.nn as nn
from torch import Tensor
from multi_head_attention import MultiHeadAttention
from positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        """
        Args:
            d_model:      dimension of embeddings
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()
        # masked multi-head attention sublayer
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for masked multi-head attention
        self.masked_attn_layer_norm = nn.LayerNorm(d_model)

        # multi-head attention sublayer
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        # layer norm for multi-head attention
        self.attn_layer_norm = nn.LayerNorm(d_model)

        # position-wise feed-forward network
        self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        # layer norm for position-wise ffn
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        Args:
            trg:          embedded sequences                (batch_size, trg_seq_length, d_model)
            src:          embedded sequences                (batch_size, src_seq_length, d_model)
            trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
            src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

        Returns:
            trg:                sequences after self-attention    (batch_size, trg_seq_length, d_model)
            attn_probs:         cross-attention softmax scores    (batch_size, n_heads, trg_seq_length, src_seq_length)
            masked_attn_probs:  self-attention softmax scores     (batch_size, n_heads, trg_seq_length, trg_seq_length)
        """
        # pass trg embeddings through masked multi-head attention
        _trg, masked_attn_probs = self.masked_attention(trg, trg, trg, trg_mask)

        # residual add and norm
        trg = self.masked_attn_layer_norm(trg + self.dropout(_trg))

        # pass trg and src embeddings through multi-head attention
        _trg, attn_probs = self.attention(trg, src, src, src_mask)

        # residual add and norm
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        # position-wise feed-forward network
        _trg = self.positionwise_ffn(trg)

        # residual add and norm
        trg = self.ffn_layer_norm(trg + self.dropout(_trg))

        return trg, attn_probs, masked_attn_probs


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ffn: int, dropout: float = 0.1):
        """
        Args:
            vocab_size:   size of the target vocabulary
            d_model:      dimension of embeddings
            n_layers:     number of encoder layers
            n_heads:      number of heads
            d_ffn:        dimension of feed-forward network
            dropout:      probability of dropout occurring
        """
        super().__init__()

        # create n_layers encoders
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffn, dropout)
                                     for layer in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # set output layer
        self.Wo = nn.Linear(d_model, vocab_size)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        Args:
            trg:          embedded sequences                (batch_size, trg_seq_length, d_model)
            src:          encoded sequences from encoder    (batch_size, src_seq_length, d_model)
            trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
            src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

        Returns:
            output:             sequences after decoder           (batch_size, trg_seq_length, vocab_size)
            attn_probs:         cross-attention softmax scores    (batch_size, n_heads, trg_seq_length, src_seq_length)
            masked_attn_probs:  self-attention softmax scores     (batch_size, n_heads, trg_seq_length, trg_seq_length)
        """

        # pass the sequences through each decoder
        for layer in self.layers:
            trg, attn_probs, masked_attn_probs = layer(trg, src, trg_mask, src_mask)

        self.attn_probs = attn_probs
        self.masked_attn_probs = masked_attn_probs

        return self.Wo(trg)