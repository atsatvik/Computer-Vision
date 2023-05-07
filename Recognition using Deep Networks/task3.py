# Satvik Tyagi


# This file uses the MNIST fashion dataset to train several neural network and save the one that performs best
# It also saves a yml file which has all the model's alteration combinations and their respective accuracies on test set 


# import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchsummary import summary
import random
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import time
import numpy as np
import yaml
import math


# Neural network structure and forward pass class definition
class MyNetwork(nn.Module):
    def __init__(self, unique_comb):
        super(MyNetwork, self).__init__()

        self.comb = unique_comb
        print(self.comb)
        #A convolution layer with specified number of filters with a specified shape of filters
        self.conv1 = nn.Conv2d(1, self.comb[1], kernel_size=self.comb[0])
        #A convolution layer with specified number of filters with a specified shape of filters
        self.conv2 = nn.Conv2d(self.comb[1], 60 , kernel_size=self.comb[0])
        #A dropout layer with a specified dropout rate
        self.conv2_drop = nn.Dropout2d(p=self.comb[2])
        #Fully connected layer

        self.fc1 = nn.Linear(self.comb[5]*60, 50)
        self.fc2 = nn.Linear(50, 10)
        #Flatten
        self.flatten = nn.Flatten()

    # computes a forward pass for the network
    def forward(self, x):
        #A max pooling layer with a specified window and a specified activation function applied on conv1.
        x = self.comb[3](F.max_pool2d(self.conv1(x), self.comb[4]))
        #Dropout layer and a max pooling layer with a specified window and a specified activation function applied on conv2
        x = self.comb[3](F.max_pool2d(self.conv2_drop(self.conv2(x)), self.comb[4]))
        #A flattening operation followed by a fully connected Linear layer with 50 nodes and a specified 
        #activation function on the output
        x = self.comb[3](self.fc1(self.flatten(x)))
        #A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
        x = F.log_softmax(self.fc2(x))

        return x



# Function for training the network
def train_network(params,network,train_loader,train_losses,train_counter,optimizer,epoch):

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % params["log_interval"] == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

# Function for testing the network
def test_network(network,test_loader,test_losses):

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    return round((100. * correct / len(test_loader.dataset)).item(),3)



#Setting hyperparameter values
def hyperparams():
    params = {
                "n_epochs" : 5,
                "batch_size_train" : 64,
                "batch_size_test" : 1000,
                "learning_rate" : 0.01,
                "momentum" : 0.5,
                "log_interval" : 10,
                "random_seed" : 1
             }
    return params


if __name__ == "__main__":


    #--------------------------Making a specified number of unique possible combinations--------------------------
    #Variations
    filter_sizes = [2,3,4]
    num_filters = [10, 20, 40]
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    activation_functions = [F.relu, F.sigmoid, F.tanh]
    pooling_sizes = [2,3]

    unique_comb = []
    num_unique_combinations = 100

    params = hyperparams()

    while num_unique_combinations != 0:
        comb = (random.choice(filter_sizes),random.choice(num_filters),
            random.choice(dropout_rates),random.choice(activation_functions),random.choice(pooling_sizes))

        #Formula to calculate for number of nodes to pass in FC1
        s_1 = 28-(comb[0]-1)#28*28 -> 24*24
        s_2 = math.floor((s_1 - comb[4])/comb[4]) + 1#24*24 -> 12*12
        s_3 = s_2 - (comb[0]-1)# 12*12 -> 8*8
        s_4 = math.floor((s_3 - comb[4])/comb[4]) + 1# 8*8 -> 4*4

        comb += (s_4*s_4,)

        if comb not in unique_comb:
            unique_comb.append(comb)
            num_unique_combinations-=1

    #-------------------------------------------------------------------------------------------------

    # Load the MNIST Fashion dataset
    train_data = FashionMNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_data = FashionMNIST(root="./data", train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    #-------------Training each combination for a set number of epoch and saving their test accruacies in a list--------

    test_acc = []
    first = True
    for i in range(len(unique_comb)):
        print("MODEL NUMBER",i+1,'\n')         
        network = MyNetwork(unique_comb[i])
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                              momentum=params["momentum"])

        train_losses = []
        train_counter = []
        test_losses = []

        test_network(network,test_loader,test_losses)
        for epoch in range(1, params["n_epochs"] + 1):
            train_network(params,network,train_loader,train_losses,train_counter,optimizer,epoch)
            test_network(network,test_loader,test_losses)

        accuracy_current = test_network(network,test_loader,test_losses) 
        test_acc.append(accuracy_current)
        print(test_acc)

    #------------------------------------------------------------------------------------------------------


    #--------Taking the combination that gives max accuracy and saving that network after training---------
    max_acc = max(test_acc)
    idx_max_acc = test_acc.index(max_acc)

    network = MyNetwork(unique_comb[idx_max_acc])
    optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                              momentum=params["momentum"])

    test_network(network,test_loader,test_losses)
    for epoch in range(1, params["n_epochs"] + 1):
        train_network(params,network,train_loader,train_losses,train_counter,optimizer,epoch)
        test_network(network,test_loader,test_losses)



    comb = unique_comb[idx_max_acc]

    if(comb[3] == F.relu):
        unique_comb_str = str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'relu_'+'poolsize' + str(comb[4])
    if(comb[3] == F.sigmoid):
        unique_comb_str = str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'sigmoid_'+'poolsize' + str(comb[4])
    if(comb[3] == F.tanh):
        unique_comb_str = str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'tanh_'+'poolsize' + str(comb[4])


    model_name ='./results/task3/model__' + unique_comb_str + '.pth'
    optimizer_name = './results/task3/optimizer__' + unique_comb_str + '.pth'

    torch.save(network.state_dict(), model_name)
    torch.save(optimizer.state_dict(), optimizer_name)

    #-------------------------------------------------------------------------------------------------

    #----------Saving all the models and their corresponding accuracies in a yml file------------------
    unique_comb_str = []
    for comb in unique_comb:
        if(comb[3] == F.relu):
            unique_comb_str.append(str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'relu_'+'poolsize' + str(comb[4]))
        if(comb[3] == F.sigmoid):
            unique_comb_str.append(str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'sigmoid_'+'poolsize' + str(comb[4]))
        if(comb[3] == F.tanh):
            unique_comb_str.append(str(comb[0])+'_'+str(comb[1])+'_'+str(comb[2])+'_'+'tanh_'+'poolsize' + str(comb[4]))

    dictionary = {k:v for k,v in zip(unique_comb_str, test_acc)}
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

    with open('model_and_accuracy.yml', 'w') as file:
        # Use the yaml.dump method to write the data to the file
        yaml.dump(sorted_dict, file)

    #-------------------------------------------------------------------------------------------------