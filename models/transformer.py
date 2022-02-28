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
                 mask=None,
                 use_bias=True):
        super(MultiHeadAttentionBase, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        self.k_linear = None
        self.q_linear = None
        self.v_linear = None
        self.combine_heads = None
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

    def attention(self, query, key, value, mask=None):
        '''
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        attn = torch.matmul(scores, value)
        return attn, scores
    
    def downstream(self, query, key, value, batch_size, mask=None):
        Q = self.separate_heads(query)
        K = self.separate_heads(key)
        V = self.separate_heads(value)

        x, self.attn_weights = self.attention(Q, K, V, mask)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return concat
        # output = self.combine_heads(concat)
        # return output

    def forward(self, x, mask=None):
        raise NotImplementedError("Base class does not execute forward function.")


class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 mask=None,
                 use_bias=True):
        super(MultiHeadAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads, mask=mask, use_bias=use_bias)

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
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, mask=mask)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(attn_output + x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :])

class TextClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
                 max_seq_len: int,
                 ffn_dim: int = 32,
                 dropout=0.1):
        super(TextClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, maxlen=max_seq_len)

        transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)
        ]

        self.transformers = nn.Sequential(*transformer_blocks)
        self.class_logits = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pos_encoding(self.token_embedding(x))
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)

class Seq2Seq(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 num_heads: int,
                 num_blocks: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 ffn_dim: int = 32,
                 unk_idx: int = 0,
                 dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.unk_idx = unk_idx

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, maxlen=max_seq_len)

        self.encoders = nn.ModuleList([])
        for _ in range(num_blocks):
            self.encoders.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim),
                FeedForward(embed_dim, ffn_dim),
                nn.LayerNorm(embed_dim),
            ]))

        self.decoders = nn.ModuleList([])
        for _ in range(num_blocks):
            self.decoders.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim),
                FeedForward(embed_dim, ffn_dim),
                nn.LayerNorm(embed_dim),
            ]))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask):
        memory = self.encode(src)
        x = self.decode(tgt, memory, tgt_mask)
        x = self.linear(self.dropout(x))
        return x

    def encode(self, src):
        x = self.pos_encoding(self.src_embedding(src))
        for attn, norm1, ff, norm2 in self.encoders:
            x = norm1(attn(x, x, x)[0] + x)
            x = norm2(ff(x) + x)
        return x

    def decode(self, tgt, memory, tgt_mask):
        x = self.pos_encoding(self.tgt_embedding(tgt))
        for mattn, norm1, attn, norm2, ff, norm3 in self.decoders:
            x = norm1(mattn(x, x, x, attn_mask=tgt_mask)[0] + x)
            x = norm2(attn(x, memory, memory)[0] + x)
            x = norm3(ff(x) + x)
        return x
