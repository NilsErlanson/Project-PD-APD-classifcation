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
import create_dataset
import visFuncs
from create_dataset import ScanDataSet

import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split, Subset

def find_class_sample_count(original_dataset):
    class_sample_count = []
    for i in range(len(original_dataset)):
        sample = original_dataset.__getitem__(i)
        label = sample[1]

        class_sample_count.append((np.asscalar(np.where(label == 1)[0])))

    #Count how many times a class occured
    zeros = class_sample_count.count(0)
    ones = class_sample_count.count(1)
    twos = class_sample_count.count(2)
    threes = class_sample_count.count(3)

    for n, i in enumerate(class_sample_count):
        if i == 0:
          class_sample_count[n] = zeros
        elif i == 1:
          class_sample_count[n] = ones
        elif i == 1:
          class_sample_count[n] = twos
        else:
          class_sample_count[n] = threes

    return class_sample_count


def training_session(device, model, optimizer, cost_function, train_data, test_data):
    print("Training...")
    # track the training and test loss
    training_loss = []
    test_loss = []

    # optimize parameters for the number of epochs in the configuration file
    for i in range(config.epochs):

        # for each minibatch
        for x, y in train_data:

            # Convert the data into the right format
            x = x.permute(0,4,1,2,3)
            x = x.float().to(device)
            y = y.float().to(device)
            target = torch.max(y,1)[1] #Needed transform for crossentropy

            # evaluate the cost function on the training data set
            loss = cost_function(model.forward(x), target)

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
                x = x.float().to(device)
                y = y.float().to(device)
                target = torch.max(y,1)[1]
                loss = cost_function(model.forward(x), target)
                accumulated_loss += loss.item()
                
            #update the statistics
            test_loss[-1] = accumulated_loss / len(test_data)
                
        print(f"Epoch {i + 1:2d}: training loss {training_loss[-1]: 9.3f},"f"test loss {test_loss[-1]: 9.3f}")
    
    print("Done!")
    
    return model, training_loss, test_loss

def kfold(device, original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 3))

    # Apply the tran/test transforms defined in config_2D
    original_dataset = create_dataset.ApplyTransform(original_dataset, gammaTransform = config.applyGammaTransformation)
    
    # Count the number of samples in each class and create a vector in order to create weigths for the weighted sampling
    class_sample_count_original = find_class_sample_count(original_dataset)
    
    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range( len(original_dataset) ):
        print(j, "...")

        # For unbalanced dataset we create a weighted sampler                       
        class_sample_count = class_sample_count_original.copy()
        class_sample_count.pop(j)

        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = len(weights) - 3, replacement = False)

        # Split the dataset into training/test, 1-fold validation
        test_index = [j]  
        all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
        train_indices = np.delete(all_indices,test_index)
        train_dataset, test_dataset = [Subset(dataset = original_dataset, indices = train_indices),Subset(dataset = original_dataset, indices = test_index)]

        # define the data loaders
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config.batchSize, sampler = sampler)     
        #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batchSize, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config.batchSize)

        # define the model
        model2 = model.SimpleCNN().to(device)

        # define the cost function
        cost_function = torch.nn.CrossEntropyLoss()

        # define the optimizer
        optimizer = torch.optim.Adam(model2.parameters(), lr = config.learning_rate)

        # run training
        trained_model,_,test_loss = training_session(device, model2, optimizer, cost_function, train_data, test_data)

        # Check if the model do the right predictions
        for x, y in test_data:
            # Convert the data into the right format
            x = x.permute(0,4,1,2,3)
            x = x.float().to(device)
            y = y.float().to(device)

            prediction = torch.argmax(F.softmax(trained_model.forward(x), dim = 1))
            label = torch.max(y,1)[1][0]

            # Check if we predicted the right, save result
            if torch.eq(prediction, label):
                accuracy[j,0] = 1

            accuracy[j,1] = prediction
            accuracy[j,2] = label

    return accuracy

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
    # Check if cuda is available 
    if config.USE_CUDA and torch.cuda.is_available():
        print("Cuda is available, using CUDA")
        device = torch.device("cuda:0")
    else:
        print("Cuda is not available, using CPU")
        device = torch.device("cpu")

    # Load the datasets
    original_dataset = load_dataset.load_datasets()
    
    # Try to apply gamma transformation
    original_dataset = create_dataset.ApplyTransform(original_dataset, config.applyGammaTransformation)

    """
    # Define the test and train sizes
    train_size = int(0.8 * len(original_dataset))
    test_size = len(original_dataset) - train_size
    # Split into training and test dataset
    train_dataset, test_dataset = random_split(original_dataset, [train_size, test_size])

    # For unbalanced dataset we create a weighted sampler                       
    class_sample_count = find_class_sample_count(train_dataset)
    weights = 1 / torch.Tensor(class_sample_count)
    weights = weights.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights) - 3, replacement = False)

    # Define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config.batchSize, sampler = sampler)  
    #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = 1)

    # define the model
    model = model.SimpleCNN().to(device)

    # define the cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    # run training
    trained_model, training_loss, test_loss = training_session(device, model, optimizer, cost_function, train_data, test_data)

    # plot the test and training loss
    plot_training_test_loss(training_loss, test_loss)
    """
    # Validate the robustness of the model
    accuracy = kfold(device, original_dataset)

    print(accuracy)
    

