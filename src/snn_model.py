import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class EventSNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(2 * 128 * 128, 512)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc3 = nn.Linear(256, num_classes)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        b, t, c, h, w = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        out_spikes = []

        for step in range(t):
            xt = x[:, step].reshape(b, -1)
            cur1 = self.fc1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            out_spikes.append(spk3)

        return torch.stack(out_spikes).sum(dim=0)