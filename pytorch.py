"""
In this file we use Pytorch to train both mnist and miniplaces.

We apply a simplified version of ResNet-18 for miniplaces.

@author: Hongxu Chen
"""

# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class SimpleFCNet(nn.Module):
    """
    A simple neural network with fully connected layers to train mnist
    
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        super(SimpleFCNet, self).__init__()
        # create the model by adding the layers
        layers = []

        ###################################
        #     fill in the code here       #
        ###################################
        # Add a Flatten layer to convert the 2D pixel array to a 1D vector
        layers.append(nn.Flatten())
        # Add a fully connected / linear layer with 128 nodes
        layers.append(nn.Linear(28*28, 128))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with 64 nodes
        layers.append(nn.Linear(128, 64))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with num_classes (10) nodes
        layers.append(nn.Linear(64, num_classes))
        
        self.layers = nn.Sequential(*layers)

        self.reset_params()

    def forward(self, x):
        # the forward propagation
        out = self.layers(x)
        if self.training:
            # softmax is merged into the loss during training
            return out
        else:
            # attach softmax during inference
            out = nn.functional.softmax(out, dim=1)
            return out

    def reset_params(self):
        # to make our model a faithful replica of the Keras version
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)






# 28.13%
class SimpleConvNet(nn.Module):
    """
    A simplified version of Resnet-18 to train miniplaces
    
    """
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(SimpleConvNet, self).__init__()
        ####################################################
        # you can start from here and create a better model
        ####################################################
        layers = []
        # 1 conv
        layers.append(nn.BatchNorm2d(3))
        layers.append(nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Maxpool
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(24))
        
        # 2 convs
        layers.append(nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # 2 convs
        layers.append(nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(48))
        # 2 convs
        layers.append(nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, dilation=1, bias=True))
        layers.append(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(96))
        # 2 convs
        layers.append(nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(192))
        
        


        self.layers = nn.Sequential(*layers)
        # Average pool
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
        # FC
        self.fc = nn.Linear(192, num_classes)










    def forward(self, x):
        #################################
        # Update the code here as needed
        #################################
        # the forward propagation
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ######################################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ######################################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        # Compute the output
        out = model(input)
        # Compute the loss
        loss = criterion(out, target)
        # Back propogation
        loss.backward()
        # Apply optimizer
        optimizer.step()
        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
