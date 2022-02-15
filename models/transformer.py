import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# see also:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/
# https://github.com/pbloem/former
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


class MultiHeadAttentionBase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttentionBase, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        self.k_linear = None
        self.q_linear = None
        self.v_linear = None
        self.combine_heads = None
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def separate_heads(self, x):
        '''
        split into N heads
        from (batch_size, seq_len, embed_dim)
        to   (batch_size, seq_len, num_heads, embed_dim)
        then transpose (1,2) to (batch_size, num_heads, seq_len, embed_dim)
        to make mat mult straightforward for each head
        '''
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def attention(self, query, key, value, mask=None, dropout=None):
        '''
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # see also: https://tensorchiefs.github.io/dlday2018/tutorial/einsum.html
        #scores = torch.einsum('bijh, bkjh -> bikh', query, key) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        attn = torch.matmul(scores, value)
        return attn, scores
    
    def downstream(self, query, key, value, batch_size, mask=None):
        Q = self.separate_heads(query)
        K = self.separate_heads(key)
        V = self.separate_heads(value)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return concat
        # output = self.combine_heads(concat)
        # return output

    def forward(self, x, mask=None):
        raise NotImplementedError("Base class does not execute forward function.")


class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask, use_bias=use_bias)

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        x = self.downstream(Q, K, V, batch_size, mask)
        output = self.combine_heads(x)
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
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask)
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
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
                 max_seq_len: int,
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

        transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)
        ]

        self.norm = nn.LayerNorm(embed_dim)
        self.transformers = nn.Sequential(*transformer_blocks)
        self.class_logits = nn.Linear(embed_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = self.pos_embedding(torch.arange(end=x.size(1), dtype=torch.int64).to(x.device))
        x = tokens + positions
        if self.normalize:
            x = self.norm(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)
        
