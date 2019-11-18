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


# TEST PRETRAINED MODELS
import torchvision


def training_session(model, optimizer, cost_function, train_data, test_data):
    print("Training...")
    # track the training and test loss
    training_loss = []
    test_loss = []

    # Convert the data
    #if torch.cuda.is_available() and config.USE_CUDA:
    #    dtype = torch.cuda.FloatTensor()
    #else:
    #    dtype = torch.FloatTensor()

    # optimize parameters for 3 epochs
    for i in range(config.epochs):

        # for each minibatch
        for x, y in train_data:

            # Convert the data into the right format
            x = x.permute(0,4,1,2,3)
            
            #x = x.dtype
            #y = y.dtype

            # MED EGEN SPLITTNING
            x = x.float()
            y = y.float()

            # evaluate the cost function on the training data set
            target = torch.max(y,1)[1] #needed for crossentropy
                    
        
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
                x = x.float()
                y = y.double()
                target = torch.max(y,1)[1]
                loss = cost_function(model.forward(x), target)
                accumulated_loss += loss.item()
                
            #update the statistics
            test_loss[-1] = accumulated_loss / len(test_data)
                
        print(f"Epoch {i + 1:2d}: training loss {training_loss[-1]: 9.3f},"f"test loss {test_loss[-1]: 9.3f}")
    
    print("Done!")
    
    plot_training_test_loss(training_loss, test_loss)

    return model

def plot_training_test_loss(training_loss, test_loss):
    import matplotlib.pyplot as plt
    # plot loss
    plt.figure()
    iterations = np.arange(1, len(training_loss) + 1)
    plt.scatter(iterations, training_loss, label='training loss')
    plt.scatter(iterations, test_loss, label='test loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()   


# ************************** Ligger i main ***************************
if __name__ == "__main__":
    # Load the datasets
    original_dataset, augmented_dataset = load_dataset.load_datasets()

    # Split into training and test dataset
    train_dataset, test_dataset = load_dataset.split_dataset_one_random_sample_from_each_class(original_dataset, augmented_dataset)

    print("len(original_dataset): ", len(original_dataset), " type ", original_dataset)
    print("len(augmented_dataset): ", len(augmented_dataset), " type ", augmented_dataset)
    #print("len(train_dataset): ", len(train_dataset), " type ", train_dataset)
    #print("len(test_dataset): ", len(test_dataset), " type ", test_dataset)


    print("\n\n")
    # THIS DATASPLIT WORKS
    #dataset = augmented_dataset
    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #print(len(train_dataset), len(test_dataset))
    #print("FROM ORIGINAL: len(train_dataset): ", len(train_dataset), " type ", train_dataset)
    #print("FROM ORIGINAL: len(test_dataset): ", len(test_dataset), " type ", test_dataset)
    #print(type(train_dataset.__getitem__(0)[0][1,1,1,0]))
    #print("\n\n")

    # define the data loaders
    #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config.batchSize, shuffle=True)
    #test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config.batchSize)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 1)


    # define the model
    model = model.SimpleCNN() 
    #model = model.UNet3D(2, config.nrOfDifferentDiseases)  
    #model = torchvision.models.googlenet(pretrained=False, progress=True)
 

    # USES FLOAT
    from torchsummary import summary
    summary(model, input_size = (2, 128, 128, 80))


    # TRAINING USES DOUBLE
    #model = model.double()

    # define the cost function
    #cost_function = nn.MSELoss()
    cost_function = torch.nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # run training
    trained_model = training_session(model, optimizer, cost_function, train_data, test_data)

    #torch.save(trained_model)

    trained_model.eval()
    
    """
    # TEST
    test_input = train_dataset.__getitem__(0)[0]
    test_target = train_dataset.__getitem__(0)[1]
    #Convert input
    test_input = torch.tensor(test_input).float()
    test_input = test_input.unsqueeze(0)
    test_input = test_input.permute(0,4,1,2,3) 
    print(type(test_input), test_input)
    #Convert target
    print(type(test_target), test_target)
    test_target = torch.tensor([2])

    model_test = model(test_input)
    cost_test = cost_function(model(test_input), test_target)
    # TEST
    """


    """
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
    
    """

