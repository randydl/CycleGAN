import os
import sys
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader

from models import *
from dataset import Dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='D:/Workspace/Resources/monet2photo')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument("--lambda_cyc", type=float, default=10.0)
parser.add_argument("--lambda_id", type=float, default=5.0)
opt = parser.parse_args()
print(opt)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

net_G_A2B = Generator(opt.in_channels, opt.out_channels)
net_G_B2A = Generator(opt.out_channels, opt.in_channels)
net_D_A = Discriminator(opt.in_channels)
net_D_B = Discriminator(opt.out_channels)

net_G_A2B = net_G_A2B.to(device)
net_G_B2A = net_G_B2A.to(device)
net_D_A = net_D_A.to(device)
net_D_B = net_D_B.to(device)

net_G_A2B.apply(weights_init)
net_G_B2A.apply(weights_init)
net_D_A.apply(weights_init)
net_D_B.apply(weights_init)

transform = tv.transforms.Compose([
    tv.transforms.Resize(32),
    # tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor()
    # tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = Dataset(opt.data, transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

fixed_A, fixed_B = iter(dataloader).next()

criterion = nn.MSELoss()
optimizer_G = torch.optim.Adam(itertools.chain(net_G_A2B.parameters(), net_G_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(net_D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(net_D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

plt.ion()
for epoch in range(opt.epochs):
    print('Epoch: {}/{}'.format(epoch + 1, opt.epochs))

    for i, data in enumerate(dataloader):
        real_A = data[0].to(device) # 真实的A域图像
        real_B = data[1].to(device) # 真实的B域图像

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()

        # Identity Loss
        loss_id_A = criterion(net_G_B2A(real_A), real_A)
        loss_id_B = criterion(net_G_A2B(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN Loss
        fake_B = net_G_A2B(real_A)    # 生成的B域图像
        pred_fake_B = net_D_B(fake_B) # 对生成的B域图像的判别结果
        loss_GAN_AB = criterion(pred_fake_B, torch.ones_like(pred_fake_B, device=device))

        fake_A = net_G_B2A(real_B)    # 生成的A域图像
        pred_fake_A = net_D_A(fake_A) # 对生成的A域图像的判别结果
        loss_GAN_BA = criterion(pred_fake_A, torch.ones_like(pred_fake_A, device=device))

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle Loss
        same_A = net_G_B2A(fake_B) # 重建的A域图像
        loss_cycle_A = criterion(same_A, real_A)

        same_B = net_G_A2B(fake_A) # 重建的B域图像
        loss_cycle_B = criterion(same_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total Loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator A
        # ---------------------

        optimizer_D_A.zero_grad()

        # Real Loss
        pred_real_A = net_D_A(real_A) # 对真实的A域图像的判别结果
        loss_real_A = criterion(pred_real_A, torch.ones_like(pred_real_A, device=device))

        # Fake Loss
        pred_fake_A = net_D_A(fake_A.detach())
        loss_fake_A = criterion(pred_fake_A, torch.zeros_like(pred_fake_A, device=device))

        # Total Loss
        loss_D_A = (loss_real_A + loss_fake_A) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # ---------------------
        # Train Discriminator B
        # ---------------------

        optimizer_D_B.zero_grad()

        # Real Loss
        pred_real_B = net_D_B(real_B) # 对真实的B域图像的判别结果
        loss_real_B = criterion(pred_real_B, torch.ones_like(pred_real_B, device=device))

        # Fake Loss
        pred_fake_B = net_D_B(fake_B.detach())
        loss_fake_B = criterion(pred_fake_B, torch.zeros_like(pred_fake_B, device=device))

        # Total Loss
        loss_D_B = (loss_real_B + loss_fake_B) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # ---------
        # Print Log
        # ---------

        print(
            '[{}/{}]'.format(epoch + 1, opt.epochs) +
            '[{}/{}]'.format(i + 1, len(dataloader)) + ', ' +
            'loss_D: {:.4f}, loss_G: {:.4f}'.format(loss_D.item(), loss_G.item()) + ', ' +
            'loss_GAN: {:.4f}, loss_cycle: {:.4f}, loss_identity: {:.4f}'.format(loss_GAN.item(), loss_cycle.item(), loss_identity.item())
        )

        # ---------
        # Visualize
        # ---------
        fake_A = net_G_B2A(fixed_B.to(device))
        fake_A = fake_A.detach().cpu()
        fake_B = net_G_A2B(fixed_A.to(device))
        fake_B = fake_B.detach().cpu()

        images = torch.cat((fixed_A, fake_A, fixed_B, fake_B), 0)
        images = tv.utils.make_grid(images, nrow=2)
        images = images.numpy().transpose(1, 2, 0)

        plt.imshow(images)
        plt.pause(0.1)

plt.ioff()
plt.show()
