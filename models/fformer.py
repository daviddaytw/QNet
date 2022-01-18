import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False,
                 n_qubits: int = 8,
                 n_qlayers: int = 1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(max_seq_len, max_seq_len, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        x = self.k_linear(x)
        x = x.transpose(1, 2)
        x = self.q_linear(x)
        x = x.transpose(1, 2)
        output = self.v_linear(x)

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, bias=False),
            nn.ReLU(ffn_dim),
            nn.Linear(ffn_dim, embed_dim, bias=False)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 num_heads: int,
                 ff_dim: int,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim=embed_dim, max_seq_len=max_seq_len, num_heads=num_heads, dropout=dropout, mask=mask)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        attn_output = self.attn(x)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(attn_output + x)

        ff_output = self.ffn(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(ff_output + x)
        return x

class TextClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
                 ffn_dim: int = 32,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout=0.1):
        super(TextClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        transformer_blocks = [
            TransformerBlock(embed_dim, max_seq_len, num_heads, ffn_dim) for _ in range(num_blocks)
        ]

        self.transformers = nn.Sequential(*transformer_blocks)
        self.class_logits = nn.Linear(embed_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = self.pos_embedding(torch.arange(end=x.size(1), dtype=torch.int64).to(x.device))
        x = tokens + positions
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)
        
