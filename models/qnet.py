import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch.nn.utils.rnn import pad_sequence
from .embedding import QEmbedding

class QNetBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 mask=None,
                 n_qlayers: int = 1):
        super(QNetBlock, self).__init__()
        self.embed_dim = embed_dim
        
        n_wires = math.ceil(math.log2(embed_dim * max_seq_len))
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.encoder = tq.StateEncoder()
        self.trainable_gates = [
            [
                tq.Rot(has_params=True,
                       trainable=True)
                for __ in range(n_wires)
            ]
            for _ in range(n_qlayers)
        ]

    def applyOps(self, idx):
        for i in range(self.n_wires):
            self.trainable_gates[idx][i](self.q_device, wires=i)

    def vqc(self, idx):
        self.applyOps(idx)
        for i in range(self.n_wires):
            tqf.cnot(self.q_device, [(i+idx) % self.n_wires, (i+idx+1) % self.n_wires])

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        x = torch.reshape(x, (batch_size, -1))
        self.encoder(self.q_device, x)

        for i in range(self.n_qlayers):
            self.vqc(i)

        x = self.q_device.states.reshape(batch_size, seq_len, embed_dim).abs()
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(ffn_dim),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        x = self.layers(x)
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
                 max_seq_len: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
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

        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(nn.ModuleList([
                QNetBlock(embed_dim, max_seq_len),
                FeedForward(embed_dim, ffn_dim),
            ]))
        self.class_logits = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pos_encoding(self.token_embedding(x))

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

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

        self.src_embedding = QEmbedding(src_vocab_size, math.floor(math.log2(embed_dim)))
        self.tgt_embedding = QEmbedding(tgt_vocab_size, math.floor(math.log2(embed_dim)))
        self.pos_encoding = PositionalEncoding(embed_dim, maxlen=max_seq_len)

        self.encoders = nn.ModuleList([])
        for _ in range(num_blocks):
            self.encoders.append(nn.ModuleList([
                QNetBlock(embed_dim, max_seq_len),
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

    def forward(self, src, tgt, tgt_mask_size, memory=None, decode=True):
        if memory == None:
            memory = self.encode(src)
            x = memory
        if decode:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_mask_size).to(tgt.device)
            x = self.decode(tgt, memory, tgt_mask)
            x = self.linear(self.dropout(x))
        return x

    def encode(self, src):
        x = self.pos_encoding(self.src_embedding(src))
        for attn, norm1, ff, norm2 in self.encoders:
            x = norm1(attn(x) + x)
            x = norm2(ff(x) + x)
        return x

    def decode(self, tgt, memory, tgt_mask):
        x = self.pos_encoding(self.tgt_embedding(tgt))
        for mattn, norm1, attn, norm2, ff, norm3 in self.decoders:
            x = norm1(mattn(x, x, x, attn_mask=tgt_mask)[0] + x)
            x = norm2(attn(x, memory, memory)[0] + x)
            x = norm3(ff(x) + x)
        return x
