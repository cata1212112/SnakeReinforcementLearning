from imports import *
from NoisyLinear import *

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions=4):
        super(DQN, self).__init__()
        # self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=4),
        #                           nn.ReLU(inplace=True),
        #                           nn.Conv2d(32, 64, kernel_size=3, stride=2),
        #                           nn.ReLU(inplace=True),
        #                           nn.Flatten())

        self.conv = nn.Sequential(nn.Flatten())

        self.value = nn.Sequential(NoisyLinear(2500, 512),
                                   nn.ReLU(inplace=True),
                                   NoisyLinear(512, 1))

        self.action = nn.Sequential(NoisyLinear(2500, 512),
                                    nn.ReLU(inplace=True),
                                    NoisyLinear(512, n_actions))

    def forward(self, x):
        out_conv = self.conv(x)
        val = self.value(out_conv)
        act = self.action(out_conv)
        return val + act - act.mean()
