# Satvik Tyagi


# This file takes in images of greek letters and a dataloader is created. This data is then used for transfer learning
# using saved neural network and optimizer from task 1. The new neural network trained on greek letters is then tested
# with a test set containing images taken by the user


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
import cv2
import numpy as np


# Neural network structure and forward pass class definition
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        #A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        #A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        #A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)

        #Fully connected layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        #Flatten
        self.flatten = nn.Flatten()

    # computes a forward pass for the network
    def forward(self, x):     

        #A max pooling layer with a 2x2 window and a ReLU function applied on conv1.   
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        #Dropout layer and a max pooling layer with a 2x2 window and a ReLU function applied on conv2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        #A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
        x = F.relu(self.fc1(self.flatten(x)))

        #A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
        x = F.log_softmax(self.fc2(x))

        return x


#Setting hyperparameter values
def hyperparams():
    params = {
                "n_epochs" : 30,
                "batch_size_train" : 64,
                "batch_size_test" : 1000,
                "learning_rate" : 0.01,
                "momentum" : 0.5,
                "log_interval" : 10,
                "random_seed" : 1
             }
    return params


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
          torch.save(network.state_dict(), './results/task2/model.pth')
          torch.save(optimizer.state_dict(), './results/task2/optimizer.pth')

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
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


#Loading data 
def loaddata(params):
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])),batch_size=params["batch_size_train"], shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))])),batch_size=params["batch_size_test"], shuffle=True)
    return train_loader, test_loader

#Making a dataloader for greek symbols data
def loaddata_greek(params, path,b_size,test=False):
    greek_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(test),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,))])),
        batch_size = b_size,
        shuffle = True )
    return greek_train


# greek data set transform
class GreekTransform:
    def __init__(self,test):
        self.test = test
        pass

    def __call__(self, x):

        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        if self.test:
            x = torchvision.transforms.Resize((230,230))(x)
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )



if __name__ == "__main__":  

    params = hyperparams()

    #For repeatability we are setting these values
    torch.backends.cudnn.enabled = False
    torch.manual_seed(params["random_seed"])

    #Initializing network and optimizer
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"], momentum=params["momentum"])

    network_state_dict = torch.load('./results/task1/model.pth')
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('./results/task1/optimizer.pth')
    optimizer.load_state_dict(optimizer_state_dict)

    summary(network, (1,28,28))

    conv1_tensor = (network.conv1.weight).detach().numpy()
    print(conv1_tensor.shape)

    #----------------------------------Print and Visualize filters in first layer----------------------------------
    for i in range(conv1_tensor.shape[0]):
        print("filter {}".format(i+1),conv1_tensor[i], '\n')

    fig = plt.figure()
    for i in range(conv1_tensor.shape[0]):
        plt.subplot(4,3,i+1)
        plt.tight_layout()
        plt.imshow(conv1_tensor[i][0,:,:])
        plt.title("Filter: {}".format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    #-------------------------------------------------------------------------------------------------

    #--------loading dataset nad showing filter and its corresponding result on a image---------------
    train_loader, test_loader = loaddata(params)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    count = 2
    count_odd = 1

    img = example_data[0][0].detach().numpy()


    #Filter and its corresponding result on a image
    for i in range(conv1_tensor.shape[0]):

        filt = conv1_tensor[i][0,:,:]
        dst = cv2.filter2D(img, -1, filt)

        plt.subplot(5,4,count_odd)
        plt.tight_layout()
        plt.imshow(filt)

        plt.subplot(5,4,count)
        plt.imshow(dst)
        plt.xticks([])
        plt.yticks([])

        count += 2
        count_odd += 2         

    plt.show()
    #-------------------------------------------------------------------------------------------------

    # --------------------------Freeze and unfreezing all of the network weights----------------------
    for param in network.parameters():
        param.requires_grad = False

    network.fc2 = nn.Linear(50, 3)

    for param in network.parameters():
        param.requires_grad = True  

    #-------------------------------------------------------------------------------------------------

    #---------------------------------------DataLoader for the Greek data set-------------------------
    training_set_path = "./greek_train/greek_train/"
    test_set_path = "./dataset_greek/"

    greek_train = loaddata_greek(params, training_set_path,5)
    greek_test = loaddata_greek(params,test_set_path,9,test=True)

    #-------------------------------------------------------------------------------------------------


    #-----------------------------------Visualizing images after cropping----------------------------
    examples = enumerate(greek_test)

    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #Visualizing after cropping
    for i in range(5):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.xticks([])
      plt.yticks([])

    # Uncomment below to visualize
    # plt.show()

    #-------------------------------------------------------------------------------------------------


    # ----------------Further training and then testing of pretrained network from task 1------------------------------
    train_losses = []
    train_counter = []
    test_losses = []

    test_counter = [i*len(greek_test.dataset) for i in range(params["n_epochs"] + 1)]

    test_network(network,greek_test,test_losses)
    for epoch in range(1, params["n_epochs"] + 1):
        train_network(params,network,greek_train,train_losses,train_counter,optimizer,epoch)
        test_network(network,greek_test,test_losses)

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(greek_test)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = network(example_data)


    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    #-------------------------------------------------------------------------------------------------

    summary(network, (1,28,28))














