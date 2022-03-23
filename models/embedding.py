import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# class QEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, embed_dim: int, affine: bool=True):
#         super(QEmbedding, self).__init__()
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.n_wires = int(math.floor(math.log2(embed_dim)))
#         self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
#         self.rot_params = nn.Embedding(vocab_size, 3 * self.n_wires)
#         self.norm = nn.LayerNorm(embed_dim, elementwise_affine=affine)

#     def applyRots(self, x):
#         for i in range(self.n_wires):
#             tqf.rot(self.q_device, wires=i, params=x[:, i, :])

#     def forward(self, x):
#         bsz, seq_len = x.shape
#         p = self.rot_params(x)
#         self.q_device.reset_states(bsz * seq_len)
# #         self.q_device.states = torch.view_as_complex(self.q_device.states)

#         self.applyRots(p.reshape(bsz * seq_len, self.n_wires, 3))
#         for i in range(self.n_wires):
#                 tqf.cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])

#         x = self.q_device.states.abs().reshape(bsz, seq_len, 2 ** self.n_wires)
#         return self.norm(x)

class QEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, wires_limit: int=32, affine: bool=True):
        super(QEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pack_size = min(vocab_size * embed_dim, 2 ** wires_limit)
        self.num_packs = math.ceil((vocab_size * embed_dim) / self.pack_size)
        self.n_wires = math.ceil(math.log2(self.pack_size))
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.rot_params1 = nn.Parameter(torch.rand(self.num_packs, self.n_wires, 3))
        self.rot_params2 = nn.Parameter(torch.rand(self.num_packs, self.n_wires, 3))
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=affine)

    def applyRots(self, x):
        for i in range(self.n_wires):
            tqf.rot(self.q_device, wires=i, params=x[:, i, :])

    def forward(self, x):
        bsz, seq_len = x.shape

        self.q_device.reset_states(self.num_packs)
        self.q_device.states = torch.view_as_complex(self.q_device.states)

        self.applyRots(self.rot_params1)
        for i in range(self.n_wires):
                tqf.cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])
        self.applyRots(self.rot_params2)

        y = self.q_device.states.abs().reshape(-1)[:self.vocab_size * self.embed_dim]
        y = y.reshape(self.vocab_size, self.embed_dim)
        x = x.reshape(-1)
        x = torch.index_select(y, 0, x).reshape(bsz, seq_len, -1)
        return self.norm(x)
