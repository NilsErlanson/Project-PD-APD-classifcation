#
# Scipt which loads a model, splits the loaded datasets into training/test data and trains one model with the given parameters specified in the configuration file
#

# To use the parameters specified in config.py
import config_2D
#Our own files
import model2d
import load_dataset_2D
import visFuncs_2D
import transformations_2D
from create_dataset_2D import ScanDataSet

# Standard libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset


def training_session(model, optimizer, cost_function, train_data, test_data):
    print("Training...")
    # track the training and test loss
    training_loss = []
    test_loss = []

    # optimize parameters for 3 epochs
    for i in range(config_2D.epochs):

        # for each minibatch
        for x, y in train_data:

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
            #evaluate the number of correct predictions
            #nrCorrectPredictions = 0
            for x, y in test_data:
                x = x.float()
                y = y.float()
                target = torch.max(y,1)[1]
                loss = cost_function(model.forward(x), target)
                accumulated_loss += loss.item()
                
            #update the statistics
            test_loss[-1] = accumulated_loss / len(test_data)

            #print("Number of correct predictions: ", nrCorrectPredictions)
                
        print(f"Epoch {i + 1:2d}: training loss {training_loss[-1]: 9.3f},"f" test loss {test_loss[-1]: 9.3f}")
    
    print("Done!")
    
    return model, training_loss, test_loss

def plot_training_test_loss(training_loss, test_loss):
    # plot loss
    plt.figure()
    iterations = np.arange(1, len(training_loss) + 1)
    plt.scatter(iterations, training_loss, label='training loss')
    plt.scatter(iterations, test_loss, label='test loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()   

def validate(test_data, model):
    nrCorrectPredictions = 0

    for x,y in test_data:
        x = x.float()

        prediction = model.forward(x)

        prediction = torch.max(prediction, 1)[1]
        y = y.float()

        label = torch.max(y,1)[1] #needed transformation for crossentropy

        print("Prediction: ", torch.max(model.forward(x), 1)[1], " label: ",  label)

def kfold(original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 1))

    # Apply train and test transforms
    transform = config_2D.transform

    # Apply the tran/test transforms defined in config_2D
    original_dataset = transformations_2D.ApplyTransform(original_dataset, sliceNr = config_2D.sliceSample, applyMean = config_2D.addMeanImage, normalbrain=config_2D.adddiffNormal, transform = transform)

    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range(k):
        print(j, "...")
        # Split the dataset into training/test, 1-fold validation
        test_index = [j]  
        all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
        train_indices = np.delete(all_indices,test_index)
        train_dataset, test_dataset= [Subset(dataset = original_dataset, indices = train_indices),Subset(dataset = original_dataset, indices = test_index)]

        # define the data loaders
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

        # define the model
        model = model2d.resnet() 

        # define the cost function
        cost_function = torch.nn.CrossEntropyLoss()

        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)
        
        # run training
        trained_model,_,test_loss = training_session(model, optimizer, cost_function, train_data, test_data)

        # Check if the model do the right predictions
        for x,y in test_data:
            x = x.float()
            prediction = torch.argmax(torch.nn.functional.softmax(trained_model.forward(x), dim=1))

            y = y.float()
            label = torch.max(y,1)[1][0] 

            # Check if we predicted the right, save result
            if torch.eq(prediction, label):
                accuracy[j] = 1

                print("Correct!")

    return accuracy
        
if __name__ == "__main__":
    # Load the datasets from the pickle file   
    original_dataset = load_dataset_2D.load_original_dataset()
    """
    # Apply train and test transforms
    transform = config_2D.transform
    original_dataset = transformations_2D.ApplyTransform(original_dataset, sliceNr = config_2D.sliceSample, applyMean = config_2D.addMeanImage, normalbrain = config_2D.adddiffNormal, useMultipleSlices = config_2D.useMultipleSlices, transform = transform)
    
    # Split the original dataset into two subsets, training/testing
    train_size = int(0.8 * len(original_dataset))
    test_size = len(original_dataset) - train_size
    train_dataset, test_dataset = random_split(original_dataset, [train_size, test_size])

    # Define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

    # Define the model
    model = model2d.resnet() 

    # Define the cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)
    
    # Run training
    trained_model, training_loss, test_loss = training_session(model, optimizer, cost_function, train_data, test_data)

    # Evaluate the run
    plot_training_test_loss(training_loss, test_loss)

    # Print predictions and the labels in the test data
    validate(test_data, trained_model)
    """
    # Validate the robustness of the model
    accuracy = kfold(original_dataset)

    print(accuracy)


