'''
Author: Jiayi Liu
Date: 2022-11-01 15:26:11
LastEditors: Jiayi Liu
LastEditTime: 2022-11-01 16:37:03
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch, torchvision
from torchvision import datasets, transforms
from resnet import BottleNeckBlock

CIFAR_PATH = "/home/ubuntu/MyFiles/private/DL-paper-implementation/Basic-CV/cifar100"
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers= 2

def cifar100_dataset():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=100, shuffle=True, num_workers=num_workers)
        
    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False, num_workers=num_workers)
    
    return trainloader,testloader

trainloader,testloader = cifar100_dataset()

inputs, targets = next(iter(trainloader))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BottleNeckBlock(3, 64)
model = model.to(device)
inputs = inputs.to(device)
print(inputs.shape)
# print(model(inputs).shape)