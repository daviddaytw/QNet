import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch.nn.utils.rnn import pad_sequence

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class QFNetBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_seq_len: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False,
                 n_qlayers: int = 1):
        super(QFNetBlock, self).__init__()
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
            tqf.cnot(self.q_device, [i+idx, (i+idx+1) % self.n_wires])
        
    def qft(self):
        for n in range(self.n_wires - 1, 0, -1):
            tqf.hadamard(self.q_device, wires=n)
            for i in range(n):
                psi = torch.tensor([[math.pi/2**(n-i), 0]])
                psi = torch.view_as_complex(psi)
                tqf.cp(self.q_device, wires=[i, n], params=psi)

        for i in range(self.n_wires//2):
            tqf.swap(self.q_device, wires=[i, self.n_wires-i-1])

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        x = torch.reshape(x, (batch_size, -1))
        self.encoder(self.q_device, x)

        for i in range(self.n_qlayers):
            self.qft()
            self.vqc(i)

        x = self.q_device.states.reshape(batch_size, seq_len, embed_dim).abs()
        return x

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
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, QFNetBlock(embed_dim, max_seq_len)),
                PreNorm(embed_dim, FeedForward(embed_dim, ffn_dim))
            ]))
        self.class_logits = nn.Linear(embed_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        positions = self.pos_embedding(torch.arange(end=x.size(1), dtype=torch.int64).to(x.device))
        x = tokens + positions

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)
        
