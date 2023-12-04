import matplotlib.pyplot as plt
import torch

from imports import *
from NoisyLinear import *

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions=3):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(1, 32, kernel_size=3, stride=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                  nn.ReLU(inplace=True),
                                  # nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  # nn.ReLU(inplace=True),
                                  nn.Flatten())

        # self.conv = nn.Sequential(
        #     nn.Flatten())


        self.value = nn.Sequential(NoisyLinear(17, 512),
                                   nn.ReLU(inplace=True),
                                   NoisyLinear(512, 1))

        self.action = nn.Sequential(NoisyLinear(17, 512),
                                    nn.ReLU(inplace=True),
                                    NoisyLinear(512, n_actions))

    def forward(self, x):
        # images = x[:, :, :16, :]
        # positions = x[:, :, 16, [0, 1]].squeeze(1)
        #
        # out_conv = self.conv(images)
        # out_conv = torch.cat((positions, out_conv), dim=1)
        out_conv = x
        val = self.value(out_conv)
        act = self.action(out_conv)
        return val + act - act.mean()
