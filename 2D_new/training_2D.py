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


def training_session(device, model, optimizer, cost_function, train_data, test_data):
    print("Training...")
    # track the training and test loss
    training_loss = []
    test_loss = []

    # optimize parameters for 3 epochs
    for i in range(config_2D.epochs):

        # for each minibatch
        for x, y in train_data:

            x = x.float().to(device)
            y = y.float().to(device)

            # evaluate the cost function on the training data set
            target = torch.max(y,1)[1] #needed for crossentropy
                    
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
            #evaluate the number of correct predictions
            #nrCorrectPredictions = 0
            for x, y in test_data:
                x = x.float().to(device)
                y = y.float().to(device)
                target = torch.max(y,1)[1]
                loss = cost_function(model.forward(x), target)
                accumulated_loss += loss.item()
                
            #update the statistics
            test_loss[-1] = accumulated_loss / len(test_data)

            #print("Number of correct predictions: ", nrCorrectPredictions)
                
        print(f"Epoch {i + 1:2d}: training loss {training_loss[-1]: 9.3f},"f" test loss {test_loss[-1]: 9.3f}")
    
    print("Done!")
    
    return model, training_loss, test_loss 

def kfold(device, original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 3))
    test_losses = np.zeros((len(original_dataset), config_2D.epochs * int((16*config_2D.nrAugmentations)/config_2D.batchSize) ))
    train_losses = np.zeros((len(original_dataset), config_2D.epochs * int((16*config_2D.nrAugmentations)/config_2D.batchSize) ))

    #class_sample_count_original = find_class_sample_count(original_dataset)

    test_samples_vector = random_test_sets(original_dataset) 
    print("The list order before going into k-fold loop: ", test_samples_vector)

    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range(0, len(original_dataset), config_2D.nrOfDifferentDiseases):
        print(j," ", j+1," ", j+2," ", j+3, "...")
        print("Testing on: ", test_samples_vector[j], " ", test_samples_vector[j+1], " ", test_samples_vector[j+2], " ", test_samples_vector[j+3])

        # Split the dataset into training/test, 1-fold validation
        test_index = [test_samples_vector[j], test_samples_vector[j+1], test_samples_vector[j+2], test_samples_vector[j+3]]  
        all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
        train_indices = np.delete(all_indices,test_index)
        train_dataset, test_dataset = [Subset(dataset = original_dataset, indices = train_indices),Subset(dataset = original_dataset, indices = test_index)]

        #Count the weights according to the training dataset 
        class_sample_count = find_class_sample_count(train_dataset)

        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = len(weights)*8, replacement = True)

        # Normalize the training/test datasets
        train_dataset = transformations_2D.ApplyNormalization(train_dataset, None, None, True)
        test_dataset = transformations_2D.ApplyNormalization(test_dataset, train_dataset.max_suvr_values, train_dataset.max_rcbf_values, False)

        # Apply transforms to the training/test datasets
        train_dataset = transformations_2D.ApplyTransform(train_dataset, 
                                            sliceNr = config_2D.sliceSample, 
                                            applyDiffnormal = config_2D.applydiffnormal,
                                            meantrainbrain_rcbf = None, 
                                            meantrainbrain_suvr = None, 
                                            useMultipleSlices = config_2D.useMultipleSlices, 
                                            mirrorImage = config_2D.applyMirrorImage, 
                                            gammaTransform = config_2D.gamma, 
                                            transform = config_2D.transform_train,
                                            randomSlice = True)

        meannormalbrain_rcbf = train_dataset.meannormalbrain_rcbf
        meannormalbrain_suvr = train_dataset.meannormalbrain_suvr

        test_dataset = transformations_2D.ApplyTransform(test_dataset, 
                                        sliceNr = config_2D.sliceSample, 
                                        applyDiffnormal = config_2D.applydiffnormal,
                                        meantrainbrain_rcbf = meannormalbrain_rcbf, 
                                        meantrainbrain_suvr = meannormalbrain_suvr, 
                                        useMultipleSlices = config_2D.useMultipleSlices, 
                                        mirrorImage = config_2D.applyMirrorImage, 
                                        gammaTransform = config_2D.gamma, 
                                        transform = config_2D.transform_test,
                                        randomSlice = False)

        # define the data loaders
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, sampler = sampler, drop_last = True)     
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

        # define the model
        model = model2d.resnet().to(device)

        # define the cost function
        cost_function = torch.nn.CrossEntropyLoss()

        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)

        # run training
        trained_model,train_loss,test_loss = training_session(device, model, optimizer, cost_function, train_data, test_data)

        # Check if the model do the right predictions
        for x,y in test_data:
            x = x.float().to(device)
            y = y.float().to(device)

            probs = F.softmax(trained_model.forward(x), dim = 1)
            for i in range(len(test_index)):
                label = torch.max(y,1)[1][i]
                prediction = torch.argmax(probs[i], dim=0)
                print("prediction: ", prediction, "label: ", label, "probabilities ", probs[i])
                #Store the result
                if torch.eq(prediction, label):
                    accuracy[i + j,0] = 1

                accuracy[i + j,1] = prediction
                accuracy[i + j,2] = label    
        
        #Store a metric for the train/test loss of the training
        test_losses[j,:] = np.asarray(test_loss)
        train_losses[j,:] = np.asarray(train_loss)

    return accuracy, train_losses, test_losses

def validate(device, test_data, model):
    nrCorrectPredictions = 0

    for x,y in test_data:
        x = x.float().to(device)

        prediction = model.forward(x)

        prediction = torch.max(prediction, 1)[1]
        y = y.float().to(device)

        label = torch.max(y,1)[1] #needed transformation for crossentropy

        #print("Prediction: ", torch.max(model.forward(x), 1)[1], " label: ",  label)
        print(torch.max(model.forward(x), 1)[1], " Prediction: ", torch.nn.functional.softmax(model.forward(x), dim=1), " label: ",  label)
 

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

def random_test_sets(original_dataset):
    indices = np.arange(len(original_dataset))
    nrOfLeastSamples = config_2D.nrOfDifferentDiseases
    class_sample_count = []
    test_matrix = np.zeros((nrOfLeastSamples,nrOfLeastSamples))
    test_vector = np.zeros((nrOfLeastSamples*nrOfLeastSamples))
    
    for i in range(len(original_dataset)):
        sample = original_dataset.__getitem__(i)
        label = sample[1]

        #class_sample_count.append((np.asscalar(np.where(label == 1)[0])))
        class_sample_count.append((np.where(label == 1)[0]).item(0))
    
    test_matrix[:,0] = np.random.choice(np.array([i for i, x in enumerate(class_sample_count) if x == 0]),nrOfLeastSamples, replace = False)
    test_matrix[:,1] = np.random.choice(np.array([i for i, x in enumerate(class_sample_count) if x == 1]),nrOfLeastSamples, replace = False)
    test_matrix[:,2] = np.random.choice(np.array([i for i, x in enumerate(class_sample_count) if x == 2]),nrOfLeastSamples, replace = False)
    test_matrix[:,3] = np.random.choice(np.array([i for i, x in enumerate(class_sample_count) if x == 3]),nrOfLeastSamples, replace = False)    
    # Randomize the order
    np.random.shuffle(test_matrix)
    
    x = 0
    for i in range(nrOfLeastSamples):  
        for j in range(nrOfLeastSamples):
            test_vector[x] = test_matrix[i,j]
            x = x + 1
    
    remaining = np.setxor1d(test_vector, indices).astype(int)
    
    test_vector = np.append(test_vector, remaining).astype(int)
        
    return test_vector.astype(int)
        
if __name__ == "__main__":
    # Check if cuda is available 
    if config_2D.USE_CUDA and torch.cuda.is_available():
        print("Cuda is available, using CUDA")
        device = torch.device("cuda:0")
    else:
        print("Cuda is not available, using CPU")
        device = torch.device("cpu")

    # Load the datasets from the pickle file   
    original_dataset = load_dataset_2D.load_original_dataset()
    """
    # Split the original dataset into two subsets, training/testing
    train_size = int(0.9* len(original_dataset))
    test_size = len(original_dataset) - train_size
    train_dataset, test_dataset = random_split(original_dataset, [train_size, test_size])

    # Normalize the training/test datasets
    train_dataset = transformations_2D.ApplyNormalization(train_dataset, None, None, True)
    test_dataset = transformations_2D.ApplyNormalization(test_dataset, train_dataset.max_suvr_values, train_dataset.max_rcbf_values, False)
    
    # Apply transforms to the training/test datasets
    train_dataset = transformations_2D.ApplyTransform(train_dataset, 
                                                        sliceNr = config_2D.sliceSample, 
                                                        applyDiffnormal = config_2D.applydiffnormal,
                                                        meantrainbrain_rcbf = None, 
                                                        meantrainbrain_suvr = None, 
                                                        useMultipleSlices = config_2D.useMultipleSlices, 
                                                        mirrorImage = config_2D.applyMirrorImage, 
                                                        gammaTransform = config_2D.gamma, 
                                                        transform = config_2D.transform_train,
                                                        randomSlice = True)

    meannormalbrain_rcbf = train_dataset.meannormalbrain_rcbf
    meannormalbrain_suvr = train_dataset.meannormalbrain_suvr

    test_dataset = transformations_2D.ApplyTransform(test_dataset, 
                                                        sliceNr = config_2D.sliceSample, 
                                                        applyDiffnormal = config_2D.applydiffnormal,
                                                        meantrainbrain_rcbf = meannormalbrain_rcbf, 
                                                        meantrainbrain_suvr = meannormalbrain_suvr, 
                                                        useMultipleSlices = config_2D.useMultipleSlices, 
                                                        mirrorImage = config_2D.applyMirrorImage, 
                                                        gammaTransform = config_2D.gamma, 
                                                        transform = config_2D.transform_test,
                                                        randomSlice = False)
    
    # For unbalanced dataset we create a weighted sampler                       
    class_sample_count = find_class_sample_count(train_dataset)
    weights = 1 / torch.Tensor(class_sample_count)
    weights = weights.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset)*4, replacement = True)

    # Define the data loaders
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, sampler = sampler)     
    #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

    # Define the model
    model = model2d.resnet().to(device)

    # Define the cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)
    
    # Run training
    trained_model, training_loss, test_loss = training_session(device, model, optimizer, cost_function, train_data, test_data)

    # Evaluate the run
    visFuncs_2D.plot_training_test_loss(training_loss, test_loss)

    # Print predictions and the labels in the test data
    validate(device, test_data, trained_model)
    """
    # Validate the robustness of the model
    accuracy, train_losses, test_losses = kfold(device, original_dataset)
    print("Accuracy ", accuracy)
    #print(accuracy)
    
