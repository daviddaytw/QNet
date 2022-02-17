import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(nn.Module):
    def __init__(self, n_wires, arch=None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.encoder = tq.StateEncoder()
        self.trainable_u = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=n_wires)
        self.trainable_u1 = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=n_wires)

    def forward(self, x):
        bsz = x.shape[0]
        self.encoder(self.q_device, x)
        self.trainable_u(self.q_device, wires=list(range(self.n_wires)))
        for i in range(self.n_wires):
            tqf.cnot(self.q_device, [i, (i+1) % self.n_wires])
        self.trainable_u1(self.q_device, wires=list(range(self.n_wires)))
        x = self.q_device.states.view(bsz, -1).abs()

        return x