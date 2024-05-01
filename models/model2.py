import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, layers=7):
        super(Model, self).__init__()
        pow_2 = [2**i for i in range(layers)][::-1]
        # Add i layers like input_size -> 2^i -> 2^(i-1) -> ... -> 2^0 -> 1
        self.fcs = nn.ModuleList([nn.Linear(input_size, pow_2[0])])
        self.fcs.extend([nn.Linear(pow_2[i], pow_2[i+1]) for i in range(layers-1)])
        self.init_weights()

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i != len(self.fcs) - 1:
                x = torch.relu(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)