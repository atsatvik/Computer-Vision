# Satvik Tyagi


# This file takes command line inputs and then performs actions accordingly. Main function performs the following operations
# based on the input given:
# >Train a neural network on MNIST dataset from scratch 
# >Continue the training for a given number of epochs 
# > Output the structure of nerual network
# > Test the trained neural network
# > Output the first six images in the dataset
# > Get loss graph for training')
# > Test the trained network and print results



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

# class definitions
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
    # methods need a summary comment
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
          torch.save(network.state_dict(), './results/task1/model.pth')
          torch.save(optimizer.state_dict(), './results/task1/optimizer.pth')

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


#Plotting the first six images in the "test" MNIST dataset
def firstsix(test_loader):

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    plt.show()

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


#Making a dataloader for custom numbers data
def loaddata_custom_nums(params, path,b_size,test=False):
    custom_num_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       numtransform(test),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,))])),
        batch_size = b_size,
        shuffle = True )
    return custom_num_train


# Custom number data set transform
class numtransform:

    def __init__(self,test):
        self.test = test
        pass

    def __call__(self, x):

        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        if self.test:
            x = torchvision.transforms.Resize((100,100))(x)
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


# main function (yes, it needs a comment too)
def main(argv):

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Task 1')
    
    parser.add_argument('--siximg', type=bool, default=False, help='Output first six images with their ground label')
    parser.add_argument('--scratch_train', type=bool, default=False, help='Train the NN from scratch')
    parser.add_argument('--cont_train', type=bool, default=False, help='Continue NN training from a saved model and optimizer')
    parser.add_argument('--epochs_cont', type=int, default=3, help='# of epochs for continued training')
    parser.add_argument('--network_diag', type=bool, default=False, help='Print diagram structure')
    parser.add_argument('--loss_diag', type=bool, default=False, help='Get loss graph for training')
    parser.add_argument('--test_net', type=bool, default=False, help='Test the trained network and print results')
    parser.add_argument('--test_net_custom', type=bool, default=False, help='Test the pre trained network and print results on custom data')


    args = parser.parse_args()


    #Setting hyperparameters
    params = hyperparams()

    #For repeatability we are setting these values
    torch.backends.cudnn.enabled = False
    torch.manual_seed(params["random_seed"])

    #loading dataset
    train_loader, test_loader = loaddata(params)

    train_losses = []
    train_counter = []
    test_losses = []

    #output first 6 examples in MNIST
    if (args.siximg == True):
        firstsix(test_loader)

    #Initializing network and optimizer 
    if (args.scratch_train == True):
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                          momentum=params["momentum"])
        test_counter = [i*len(train_loader.dataset) for i in range(params["n_epochs"] + 1)]

        test_network(network,test_loader,test_losses)
        for epoch in range(1, params["n_epochs"] + 1):
            train_network(params,network,train_loader,train_losses,train_counter,optimizer,epoch)
            test_network(network,test_loader,test_losses)
        

    if(args.cont_train == True):
        test_counter = [i*len(train_loader.dataset) for i in range(args.epochs_cont + 1)]

        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                                    momentum=params["momentum"])

        network_state_dict = torch.load('./results/model.pth')
        network.load_state_dict(network_state_dict)

        optimizer_state_dict = torch.load('./results/optimizer.pth')
        optimizer.load_state_dict(optimizer_state_dict)

        test_counter = []
        test_network(network,test_loader,test_losses)
        for i in range(1,args.epochs_cont + 1):
            test_counter.append(i*len(train_loader.dataset))
            train_network(params,network,train_loader,train_losses,train_counter,optimizer,i)
            test_network(network,test_loader,test_losses)

    if(args.network_diag == True):
        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                          momentum=params["momentum"])
        summary(network, (1,28,28))

    if(args.loss_diag == True):        
        fig = plt.figure()
        plt.plot(train_counter, train_losses, color='blue')
        plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        plt.show()

    if(args.test_net == True):
        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                                    momentum=params["momentum"])

        network_state_dict = torch.load('./results/model.pth')
        network.load_state_dict(network_state_dict)

        optimizer_state_dict = torch.load('./results/optimizer.pth')
        optimizer.load_state_dict(optimizer_state_dict)

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

    if(args.test_net_custom == True):

        test_set_path = './dataset_digits/'
        test_loader = loaddata_custom_nums(params, test_set_path,9,test=True)

        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        network = MyNetwork()
        optimizer = optim.SGD(network.parameters(), lr=params["learning_rate"],
                                    momentum=params["momentum"])

        network_state_dict = torch.load('./results/task1/model.pth')
        network.load_state_dict(network_state_dict)

        optimizer_state_dict = torch.load('./results/task1/optimizer.pth')
        optimizer.load_state_dict(optimizer_state_dict)

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

    
    return

if __name__ == "__main__":
    main(sys.argv)


    


