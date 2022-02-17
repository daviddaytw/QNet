import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class FlatNetBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False,
                 n_qlayers: int = 1):
        super(FlatNetBlock, self).__init__()
        self.embed_dim = embed_dim
        
        ffn_dim = embed_dim * max_seq_len
        self.layers = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim),
            nn.BatchNorm1d(ffn_dim)
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        x = torch.reshape(x, (batch_size, -1))
        x = self.layers(x)
        x = torch.reshape(x, (batch_size, seq_len, embed_dim))
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
                 normalize: bool = False,
                 dropout=0.1):
        super(TextClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.normalize = normalize

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(FlatNetBlock(embed_dim, max_seq_len))
        self.ff = nn.Linear(embed_dim, embed_dim)
        self.class_logits = nn.Linear(embed_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = self.pos_embedding(torch.arange(end=x.size(1), dtype=torch.int64).to(x.device))
        x = tokens + positions
        
        if self.normalize:
            x = self.norm(x)

        for attn in self.layers:
            x = attn(x) + x
        x = self.ff(x)

        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)
        
