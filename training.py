#
# Scipt which loads a model, splits the loaded datasets into training/test data
#
# Saves the trained model into a file which can be used to predict using the file predict.py
# 
# Could be used to validate the model using validate.py
#

# To use the parameters specified in config.py
import config
#Our own files
import model
import load_dataset
import visFuncs
from create_dataset import ScanDataSet

import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split


def training_session(model, optimizer, cost_function, train_data, test_data):
    # track the training and test loss
    training_loss = []
    test_loss = []

    # optimize parameters for 3 epochs
    for i in range(config.epochs):

        # for each minibatch
        for x, y in train_data:
            x = x.permute(0,4,1,2,3)

            # evaluate the cost function on the training data set

            #loss = cost_function(model(x), torch.max(y, 1)[1]) #need model output and target
            #loss = criterion(outputs, torch.max(labels, 1)[1])
            target = torch.max(y,1)[1] #needed for crossentropy

            #print("target: ",target)
            loss = cost_function(model.forward(x), target)
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

        
# ************************** Ligger i main ***************************
if __name__ == "__main__":
    # Split into training and test dataset
    original_dataset, augmented_dataset = load_dataset.load_datasets()
    dataset = original_dataset

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(len(train_dataset), len(test_dataset))
    
    #print(train_dataset[0][1])

    #Load the data and create test/train sets
    #original_dataset, augmented_dataset = preprocessing.load_datasets()
    #train_dataset, test_dataset = Get_dataset.split_dataset_one_random_sample_from_each_class(original_dataset, augmented_dataset)

    #print("len(train_dataset): ", len(train_dataset))
    #print("len(test_dataset): ", len(test_dataset))
    #sample = train_dataset.__getitem__(0)
    #print("sample[0].type: ", type(sample[0][...,1]))


    # Try to learn and validate simple CNN
    # define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config.batchSize, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config.batchSize)
    # # test data for the checking the network
    
    #testtensor = torch.rand((1,2,128,128,80), dtype=torch.float)
    #print(testtensor.shape)
   
    # define the model
    #model = simple_network()
    model = model.SimpleCNN()
    #output = model.forward(testtensor)
    #print(output)
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
    
    scan = scan.permute(0,4,1,2,3)
    label = test[1]
    print(visFuncs.get_name(label))
    print(label)
    prediction = model.forward(scan)
    
    val, ind = torch.max(input = prediction, axis=1)
    #print(val)
    #print(predicttion)

    print(prediction)
    print(ind)
    
