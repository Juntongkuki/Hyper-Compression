# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models.resnet import resnet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mobilenet_v3():
    model = mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(1024, 10)
    return model.to(device)


def build_resnet18():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    return model.to(device)


def prepare_dataloader(batch_size: int = 10): # 128
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader

def prepare_dataloader_unet(batch_size: int = 2): # 128
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor()
        ]), download=True),
        batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader

def eval_mobilenetv3_cifar10(model: torch.nn.Module, test_loader):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


    return 100 * correct / len(test_loader.dataset)