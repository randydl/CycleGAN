import torch
import torch.nn as nn
import torch.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False), # size / 1
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False), # size / 1
            nn.InstanceNorm2d(in_channels)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, blocks=6):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False), # size / 1
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        num = 2
        for i in range(num):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False), # size / 2
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** num
        for i in range(blocks):
            model += [ResBlock(ngf * mult)]

        for i in range(num):
            mult = 2 ** (num - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # size * 2
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7), # size / 1
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, layers=3):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1), # size / 2
            nn.LeakyReLU(0.2, True)
        ]

        mult = 1
        mult_prev = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            model += [
                nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=2, padding=1, bias=False), # size / 2
                nn.InstanceNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, True)
            ]

        mult_prev = mult
        mult = min(2 ** layers, 8)
        model += [
            nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=1, padding=1, bias=False), # size - 1
            nn.InstanceNorm2d(ndf * mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)] # size - 1

        self.model = nn.Sequential(*model)

    def forward(x):
        return self.model(x)
