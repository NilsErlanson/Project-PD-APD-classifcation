#
# Scipt which loads a model, splits the loaded datasets into training/test data
#
# Saves the trained model into a file which can be used to predict using the file predict.py
# 
# Could be used to validate the model using validate.py
#

# To use the parameters specified in config.py
import config_2D
#Our own files
import model2d
import load_dataset_2D
import visFuncs_2D
import transformations_2D
from create_dataset_2D import ScanDataSet

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
    import matplotlib.pyplot as plt
    # plot loss
    plt.figure()
    iterations = np.arange(1, len(training_loss) + 1)
    plt.scatter(iterations, training_loss, label='training loss')
    plt.scatter(iterations, test_loss, label='test loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()   


from torch.utils.data import Dataset

def validate(test_data, model):
    nrCorrectPredictions = 0

    for x,y in test_data:
        x = x.float()

        prediction = model.forward(x)

        prediction = torch.max(prediction, 1)[1]
        y = y.float()

        label = torch.max(y,1)[1] #needed transformation for crossentropy

        print("Prediction: ", torch.max(model.forward(x), 1)[1], " label: ",  label)
        #print("Prediction: ", torch.nn.functional.softmax(model.forward(x), dim=1), " label: ",  label)

        for i in range(config_2D.batchSize):
            if torch.eq(torch.max(model.forward(x), 1)[1][i], label[i]):
                print("torch.max(model.forward(x), 1)[1][i] ", torch.max(model.forward(x), 1)[1][i])
                nrCorrectPredictions = nrCorrectPredictions + 1

    return nrCorrectPredictions

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

def kfold(original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 1))
    test_losses = np.zeros((len(original_dataset), 1))


    # Apply train and test transforms
    train_transform = config_2D.train_transform
    test_transform = config_2D.test_transform

    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range(k):
        print(j, "...")
        train_size = int(0.95 * len(original_dataset))
        test_size = len(original_dataset) - train_size
        train_dataset, test_dataset = random_split(original_dataset, [train_size, test_size])

        # Apply the tran/test transforms defined in config_2D
        train_dataset = transformations_2D.ApplyTransform(train_dataset, sliceNr = config_2D.sliceSample, applyMean = config_2D.addMeanImage, transform = train_transform)
        test_dataset = transformations_2D.ApplyTransform(test_dataset, sliceNr = config_2D.sliceSample, applyMean = config_2D.addMeanImage, transform = test_transform)

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
            
        #Store the test_loss 
        test_losses[j] = test_loss

    return accuracy
        

# ************************** Ligger i main ***************************
if __name__ == "__main__":
    # Load the datasets    
    print("Load dataset without transforms...")
    original_dataset = load_dataset_2D.load_original_dataset()
    print("Done!\n")

    #train_transform = config_2D.train_transform
    #original_dataset = transformations_2D.ApplyTransform(original_dataset, sliceNr = 64, applyMean = False, transform = train_transform)

    accuracy = kfold(original_dataset, len(original_dataset))

    """
    print("Apply transformations on train and test dataset!")
    # Apply train and test transforms
    train_transform = config_2D.train_transform
    test_transform = config_2D.test_transform

    train_dataset = transformations_2D.ApplyTransform(train_dataset, sliceNr = 64, applyMean = config_2D.addMeanImage, transform = train_transform)
    test_dataset = transformations_2D.ApplyTransform(test_dataset, sliceNr = 64, applyMean = config_2D.addMeanImage, transform = test_transform)
    print("Done!\n")

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
    trained_model, training_loss, test_loss = training_session(model, optimizer, cost_function, train_data, test_data)

    # evaluate the run
    plot_training_test_loss(training_loss, test_loss)

    nrCorrectPredicitons = validate(test_data, trained_model)
    print(nrCorrectPredicitons)
    #print("Number of correct predictions: ", nrCorrectPredicitons)
    #print("Validation rate: ", nrCorrectPredicitons / len(test_data))

    """

    """
    sliceNr = 60
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(10,10))

    #SUVR first row
    axs[0,0].imshow(images[:,:,sliceNr,0], cmap='hot') #SUVr
    axs[0,0].set_title(['SUVR, Slicenumber: ', sliceNr])
    axs[0,1].imshow(images[:,:,sliceNr+20,0], cmap = 'hot') #SUVr
    axs[0,1].set_title(['suvr, slicenumber: ', sliceNr+20])
    axs[0,2].imshow(images[60,:,:,0], cmap = 'hot')
    axs[0,2].set_title('rotated')

    #RBF secod row
    axs[1, 0].imshow(images[:,:,sliceNr,1], cmap = 'hot') #rCBF
    axs[1, 0].set_title(['rcbf, slicenumber: ', sliceNr])
    axs[1, 1].imshow(images[:,:,sliceNr+20,1], cmap = 'hot') #rCBF
    axs[1, 1].set_title(['rcbf, slicenumber ', sliceNr+20])
    axs[1, 2].imshow(images[60,:,:,1], cmap = 'hot') #rCBF
    axs[1, 2].set_title(['rcbf, rotated '])
    plt.show()    
    """

