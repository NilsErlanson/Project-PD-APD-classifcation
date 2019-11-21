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

            # Convert the data into the right format
            #x = x.permute(0,4,1,2,3)
            
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
            #evaluate the number of correct predictions
            #nrCorrectPredictions = 0
            for x, y in test_data:
                x = x.float()
                y = y.double()
                target = torch.max(y,1)[1]
                loss = cost_function(model.forward(x), target)
                accumulated_loss += loss.item()


                y_pred = cross_val_predict(model, x, y, cv=3)
                print("y_pred: ", y_pred)
                
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

        for i in range(config_2D.batchSize):
            if torch.eq(torch.max(model.forward(x), 1)[1][i], label[i]):
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

# ************************** Ligger i main ***************************
if __name__ == "__main__":
    # Load the datasets    
    print("Load dataset without transforms...")
    original_dataset = load_dataset_2D.load_original_dataset()
    print("Done!\n")

    original_sample = original_dataset.__getitem__(1)
    visFuncs_2D.scroll_slices(original_sample, True)

    train_size = int(0.8 * len(original_dataset))
    test_size = len(original_dataset) - train_size
    train_dataset, test_dataset = random_split(original_dataset, [train_size, test_size])

    print("Apply transformations on train and test dataset!")
    # Apply train and test transforms
    train_transform = config_2D.train_transform
    test_transform = config_2D.test_transform

    train_dataset = ApplyTransform(train_dataset, transform = train_transform)
    test_dataset = ApplyTransform(test_dataset, transform = test_transform)

    train_sample = train_dataset.__getitem__(0)
    #visFuncs_2D.show_scan(train_sample)

    print(len(train_dataset), len(test_dataset))
    print("Done!\n")

    # define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

    # define the model
    model = model2d.resnet() 
    #model = model.UNet3D(2, config.nrOfDifferentDiseases)  
    #model = torchvision.models.googlenet(pretrained=False, progress=True)
 
    # USES FLOAT
    #from torchsummary import summary
    #summary(model, input_size = (2, 128, 128, 80))

    # define the cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)
    
    from sklearn.model_selection import cross_val_predict
    # run training
    trained_model, training_loss, test_loss = training_session(model, optimizer, cost_function, train_data, test_data)

    # evaluate the run
    #plot_training_test_loss(training_loss, test_loss)

    #nrCorrectPredicitons = validate(test_data, trained_model)
    #print(nrCorrectPredicitons)
    #print("Number of correct predictions: ", nrCorrectPredicitons)
    #print("Validation rate: ", nrCorrectPredicitons / len(test_data))
    
    
    """
    # Plot the training and test results
    plot_training_test_loss(training_loss, test_loss)

    sample = original_dataset_not_normalized.__getitem__(0)
    images = sample[0]
    print("Mean: ", images[np.nonzero(images[:,:,:,0])].mean())
    print("Std: ", np.sqrt(images[np.nonzero(images[:,:,:,0])].var()))
    #visFuncs.show_scan(sample)
    
    
    """

