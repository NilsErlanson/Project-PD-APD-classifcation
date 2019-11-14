import pandas as pd
import numpy as np
import sklearn
import torch
#Our own files
import Get_dataset
import visFuncs
import preprocessing

# To rotate the images
import scipy.ndimage

#To transform the images

import matplotlib.pyplot as plt

import torchvision

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader, random_split


def training_session(model, optimizer, cost_function, train_data, test_data):
    # track the training and test loss
    training_loss = []
    test_loss = []

    # optimize parameters for 3 epochs
    for i in range(6):

        # for each minibatch
        for x, y in train_data:
            x = x.permute(0,4,1,2,3)


            # evaluate the cost function on the training data set
            #loss = cost_function(x, model, 5)

            #loss = cost_function(model(x), torch.max(y, 1)[1]) #need model output and target
            #loss = criterion(outputs, torch.max(labels, 1)[1])
            target = torch.max(y,1)[1] #needed for crossentropy
            print("target: ",target)
            loss = cost_function(model.forward(x), torch.max(y,1)[1])
            #print("Loss: " ,loss)
            # update the statistics
            training_loss.append(loss.item())
            test_loss.append(float('nan'))

            # perform backpropagation
            loss.backward()

            # perform a gradient descent step
            optimizer.step()
            
            # reset the gradient information
            optimizer.zero_grad()

        #evaluate the model after every epoch
        with torch.no_grad():

            #evaluate the cost function on the test data set
           accumulated_loss = 0
           for x, y in test_data:
               x = x.permute(0,4,1,2,3)
               loss = cost_function(model.forward(x), torch.max(y,1)[1])
               accumulated_loss += loss.item()
                
            #update the statistics
           test_loss[-1] = accumulated_loss / len(test_data)
                
        print(f"Epoch {i + 1:2d}: training loss {training_loss[-1]: 9.3f},"f"test loss {test_loss[-1]: 9.3f}")

    return model

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (128, 128, 128, 2)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 2, output channels = 18
        self.conv1 = torch.nn.Conv3d(in_channels = 2, out_channels = 2, kernel_size = (3,3,3), stride = 10, padding = 1)
        #self.pool = torch.nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 0)
        self.Flatten = torch.nn.Flatten()
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(2704, 200)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(200, 6)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        #x = self.pool(x)
        x = self.Flatten(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        #x = x.view(-1, 18 * 16 *16)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = F.softmax(self.fc2(x))

        return (x)
        
# ************************** Ligger i main ***************************
if __name__ == "__main__":
    # Split into training and test dataset
    dataset = preprocessing.predata()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(len(train_dataset), len(test_dataset))
    
    print(train_dataset[0][1])

    # Try to learn and validate simple CNN
    # define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 5, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 5)
    # # test data for the checking the network
    testtensor = torch.rand((1,2,128,128,80),dtype=torch.float)
    print(testtensor.shape)
   
    # define the model
    #model = simple_network()
    model = SimpleCNN()
    output = model.forward(testtensor)
    print(output)
    # USES FLOAT
    from torchsummary import summary
    summary(model, input_size=(2, 128, 128, 80))

    # TRAINING USES DOUBLE
    model = model.double()
 
    # define the cost function
    #cost_function = nn.MSELoss()
    cost_function = torch.nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # run training
    model = training_session(model, optimizer, cost_function, train_data, test_data)
    test = dataset[9]
    scan = test[0]
    scan = torch.from_numpy(scan).unsqueeze_(0)
    
    print(scan.shape)
    scan = scan.permute(0,4,1,2,3)
    label = test[1]
    print(visFuncs.get_name(label))
    print(label)
    predicttion = model.forward(scan)
    val,ind = torch.max(predicttion,0)
    print(val)
    print(predicttion)
    
