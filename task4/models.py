import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import torch.optim as optim
import numpy as np
import time


import matplotlib.pyplot as plt
class LogReg(nn.Module):
    def __init__(self,size=(3, 32, 32)):
        super(LogReg, self).__init__()
        self.size = size
        self.linear = nn.Linear(size[0] * size[1] * size[2], 10)

    def forward(self, x):
        ## softmax in criterion
        x = x.view(-1, self.size[0] * self.size[1] * self.size[2])
        
        x = self.linear(x)

        return x



class Net(nn.Module):
    def __init__(self, input_size=(3,32,32),conv_layers_num=2, conv_out_channels=(6, 6), conv_kernel_size=(2, 3),
                conv_stride=(1, 1), conv_padding=(0, 0), 
                 pool_kernel_size=(2, 2), pool_stride=(1, 1), pool_padding=(0, 0)):
        super(Net, self).__init__()
        size = (input_size[1], input_size[2])
        conv_blocks = [nn.Conv2d(3, conv_out_channels[0],
                                kernel_size=conv_kernel_size[0], stride=conv_stride[0], padding=conv_padding[0])]
        size = self.resize(size, conv_kernel_size[0], conv_stride[0], conv_padding[0])

        conv_blocks.append(nn.ReLU())
        conv_blocks.append(nn.MaxPool2d(pool_kernel_size[0], stride=pool_stride[0], padding=pool_padding[0], return_indices=True))
        size = self.resize(size, pool_kernel_size[0], pool_stride[0], pool_padding[0])
              
        for i in range(1, conv_layers_num):
            conv_blocks.append(nn.Conv2d(conv_out_channels[0], conv_out_channels[i],
                                kernel_size=conv_kernel_size[i], stride=conv_stride[i], padding=conv_padding[i]))
            size = self.resize(size, conv_kernel_size[i], conv_stride[i], conv_padding[i])
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(pool_kernel_size[i], stride=pool_stride[i], padding=pool_padding[i]))
            size = self.resize(size, pool_kernel_size[i], pool_stride[i], pool_padding[i])
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.size = size[0]*size[1]*conv_out_channels[conv_layers_num-1]

        self.fc = nn.Linear(self.size, 10)
    def forward(self, x):
        for i, l in enumerate(self.conv_blocks):
            if i % 3 == 2:
                x, index = l(x)
            else:
                x = l(x)
        x = x.view(-1, self.size)
        x = self.fc(x)
        return x
    def resize(self, size, kernel, stride, padding):
        new_size = [1, 2]
        for i in range(2):
            new_size[i] = floor((size[i] + 2*padding - (kernel - 1) -1)/stride + 1)
        return tuple(new_size)
        

class ConvAutoEncoder(nn.Module):
    def __init__(self, input_size=(3, 32, 32), layers_num=2, conv_out_channels=(6, 6), conv_kernel_size=(2, 3),
                conv_stride=(1, 1),conv_padding=(0,0), pool_kernel_size=(2, 2),
                 pool_stride=(1, 1), pool_padding=(0, 0)):
        super(ConvAutoEncoder, self).__init__()
        size = (input_size[1], input_size[2])
        self.conv_out_channels = [conv_out_channels[0]]
        layer_enc = [nn.Conv2d(3, conv_out_channels[0],  kernel_size=conv_kernel_size[0], stride=conv_stride[0])]
        self.size = [self.resize(size, conv_kernel_size[0], conv_stride[0], conv_padding[0])]
        size = self.resize(size, conv_kernel_size[0], conv_stride[0], conv_padding[0])
        layer_enc.append(nn.Tanh())
        layer_enc.append(nn.MaxPool2d(pool_kernel_size[0], stride=pool_stride[0], padding=pool_padding[0], return_indices=True))
        size = self.resize(size, pool_kernel_size[0], pool_stride[0], pool_padding[0])
        layer_dec = [nn.MaxUnpool2d(pool_kernel_size[0], stride=pool_stride[0], padding=pool_padding[0])]
        layer_dec.append(nn.ConvTranspose2d(conv_out_channels[0], 3, conv_kernel_size[0],
                                            stride=conv_stride[0], padding=conv_padding[0]))
        layer_dec.append(nn.Tanh())
        for i in range(1, layers_num - 1):
            layer_enc.append(nn.Conv2d(conv_out_channels[i-1], conv_out_channels[i],
                                                 conv_kernel_size[i], stride=conv_stride[i]))
            self.conv_out_channels.append(conv_out_channels[i-1])
            self.size = [self.resize(size, conv_kernel_size[0], conv_stride[0], conv_padding[0])]
            size = self.resize(size, conv_kernel_size[0], conv_stride[0], conv_padding[0])
            layer_enc.append(nn.ReLu())
            layer_enc.append(nn.MaxPool2d(pool_kernel_size[i], stride=pool_stride[i], padding=0, return_indices=True))
            size = self.resize(size, pool_kernel_size[0], pool_stride[0], pool_padding[0])
            layer_dec.append(nn.ConvTranspose2d(conv_out_channels[i], conv_out_channels[i-1],
                                                          conv_kernel_size[i], stride=conv_stride[i]))
            layer_dec.append(nn.ReLu())
            layer_dec.append(nn.MaxUnpool2d(pool_kernel_size[i], stride=pool_stride[i], padding=0))
        if layers_num-1 > 0:
            i = layers_num - 1  
            layer_enc.append(nn.Conv2d(conv_out_channels[i], conv_out_channels[i-1],
                                       conv_kernel_size[i], stride=conv_stride[i]))
            layer_enc.append(nn.Tanh())
            layer_enc.append(nn.MaxPool2d(pool_kernel_size[i], stride=pool_stride[i], padding=0, return_indices=True))
            layer_dec.append(nn.ConvTranspose2d(conv_out_channels[i], conv_out_channels[i-1],
                                                          conv_kernel_size[i], stride=conv_stride[i]))
            layer_dec.append(nn.Tanh())
            layer_dec.append(nn.MaxUnpool2d(pool_kernel_size[i], stride=pool_stride[i], padding=0))
            layer_dec = layer_dec [::-1]
        self.encoder = nn.ModuleList(layer_enc)
        self.decoder = nn.ModuleList(layer_dec)
        self.layer_num = layers_num
        
    def resize(self, size, kernel, stride, padding):
        new_size = [1, 2]
        for i in range(2):
            new_size[i] = floor((size[i] + 2*padding - (kernel - 1) -1)/stride + 1)
        return tuple(new_size)    
            
    def forward(self, x, decode=True):
        ind = []
        for i, l in enumerate(self.encoder):
            if i % 3 == 2:
                x, index = l(x)
                ind.append(index)
            else:
                x = l(x)
        if decode is True:
            for i, l in enumerate(self.decoder):
                if i % 3 == 0:
                    s = self.size[i // 3]
                    o = torch.IntTensor([x.size()[0], self.conv_out_channels[i // 3], s[0], s[1]])
                    x = l(x, ind[i // 3], output_size=o)

                else:
                    x = l(x)       
            
        return x
def fit(model, trainloader, criterion, optimizer, epoch=3):
    start = time.time()
    for epoch in range(epoch):
        #fit
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return time.time() - start
def calculate_loss(model, validloader, criterion):
    running_loss = 0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
    return running_loss/len(validloader)
def calculate_accuracy(model, validloader, criterion):
    correct = 0
    total = 0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), labels
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total

class dataloader():
    def __init__(self, dataset):
        self.data = dataset
        self.len = dataset.__len__()
        self.numbers = np.random.randint(0, self.len, self.len)
        self.i = 0
        self.size, y = dataset[0]
        self.size = tuple(self.size.size())
    def get_data(self, batch_size=4):
        if self.i + batch_size >= self.len:
            return (-1, -1)
        out = torch.Tensor(batch_size, *self.size)
        for i in range(batch_size):
            out[i, :], y = self.data[self.numbers[self.i]]
            self.i += 1
        return (out, 1)
def fit_autoenc(encoder,unlabelset, optimizer, criterion, epoch=3):
    start = time.time()
    running_loss = 0
    for epoch in range(3):
        data = dataloader(unlabelset)
        inputs, end = data.get_data()
        i = 0
        while(end==1):
            inputs = Variable(inputs)
            optimizer.zero_grad()
            outputs = encoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            inputs, end = data.get_data()
    return (time.time() - start, running_loss / len(unlabelset))
def loss_autoenc(autoenc, validloader, criterion):
    running_loss = 0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        inputs= Variable(inputs)
        outputs = autoenc(inputs)
        loss  = criterion(outputs, inputs)
        running_loss += loss.data[0]
    return running_loss / len(validloader)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))       