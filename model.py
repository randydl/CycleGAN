import torch
import torch.nn as nn
import torch.functional as F


class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, bias=False), # size / 1
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, bias=False), # size / 1
            nn.InstanceNorm2d(features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_features, out_features, ngf=64, blocks=6):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, ngf, kernel_size=7, bias=False), # size / 1
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False), # size / 2
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** n_downsampling
        for i in range(blocks):
            model += [ResBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # size * 2
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_features, kernel_size=7, bias=True), # size / 1
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(x):
        return self.model(x)
