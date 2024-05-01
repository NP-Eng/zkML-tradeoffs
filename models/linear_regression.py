from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, inputSize):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(inputSize, 1)

    def forward(self, x):
        return self.fc(x)