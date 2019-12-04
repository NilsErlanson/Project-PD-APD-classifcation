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

def kfold_v2(device, original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 3))
    test_losses = np.zeros((len(original_dataset), epochs * 10))  # there are 
    train_losses = np.zeros((len(original_dataset), epochs * 10))

    #class_sample_count_original = find_class_sample_count(original_dataset)

    # RANDOMIZE THE ORDER IN THE DATASET, DONE ONCE !
    indices = np.arange(20)
    np.random.shuffle(indices)
    #indices = [0,2,6,13,1,4,7,14,3,5,10,15,9,11,12,16,8,17,18,19]
    print("The list order before going into k-fold loop: ", indices)

    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range(0, len(original_dataset), 4):
        print(j," ", j+1," ", j+2," ", j+3, "...")
        print("Testing on: ", indices[j], " ", indices[j+1], " ", indices[j+2], " ", indices[j+3])

        # Split the dataset into training/test, 1-fold validation
        test_index = [indices[j], indices[j+1], indices[j+2], indices[j+3]]  
        all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
        train_indices = np.delete(all_indices,test_index)
        train_dataset, test_dataset = [Subset(dataset = original_dataset, indices = train_indices),Subset(dataset = original_dataset, indices = test_index)]

        #Count the weights according to the training dataset 
        class_sample_count = find_class_sample_count(train_dataset)

        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = len(weights)*8, replacement = True)

        # Normalize the training/test datasets
        train_dataset = ApplyNormalization(train_dataset, None, None, True)
        test_dataset = ApplyNormalization(test_dataset, train_dataset.max_suvr_values, train_dataset.max_rcbf_values, False)

        # Apply transforms to the training/test datasets
        train_dataset = ApplyTransform(train_dataset, 
                                            sliceNr = config_2D.sliceSample, 
                                            applyMean = config_2D.addMeanImage, 
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

        test_dataset = ApplyTransform(test_dataset, 
                                        sliceNr = config_2D.sliceSample, 
                                        applyMean = config_2D.addMeanImage, 
                                        applyDiffnormal = config_2D.applydiffnormal,
                                        meantrainbrain_rcbf = config_2D.meannormalbrain_rcbf, 
                                        meantrainbrain_suvr = config_2D.meannormalbrain_suvr, 
                                        useMultipleSlices = config_2D.useMultipleSlices, 
                                        mirrorImage = config_2D.applyMirrorImage, 
                                        gammaTransform = config_2D.gamma, 
                                        transform = config_2D.transform_test,
                                        randomSlice = False)

        # define the data loaders
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, sampler = sampler, drop_last = True)     
        #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batchSize, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size = config_2D.batchSize)

        # define the model
        model = resnet().to(device)

        # define the cost function
        cost_function = torch.nn.CrossEntropyLoss()

        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = config_2D.learning_rate)

        # run training
        trained_model,train_loss,test_loss = training_session(device, model, optimizer, cost_function, train_data, test_data)

        # Check if the model do the right predictions
        for x,y in test_data:
            x = x.float().to(device)
            #prediction = torch.argmax(F.softmax(trained_model.forward(x), dim=1))

            y = y.float().to(device)
            #label = torch.max(y,1)[1][0]

            label = torch.max(y,1)[1]
            probs = F.softmax(trained_model.forward(x), dim = 1)

            for i in range(len(test_index)):
                prediction = torch.argmax(probs[i], dim=0)
                print("prediction: ", prediction, "label: ", label[i])

                #Store the result
                if torch.eq(prediction, label):
                    accuracy[i + (j * 4),0] = 1

                accuracy[i + (j * 4),1] = prediction
                accuracy[i + (j * 4),2] = label    
      
        #Store a metric for the train/test loss of the training
        test_losses[j,:] = np.asarray(test_loss)
        train_losses[j,:] = np.asarray(train_loss)

    return accuracy, train_losses, test_losses

def kfold(device, original_dataset, k = 20):
    # Create matrix to save the accuracy and the test losses
    accuracy = np.zeros((len(original_dataset), 3))
    test_losses = np.zeros((len(original_dataset), config_2D.epochs))
    train_losses = np.zeros((len(original_dataset), config_2D.epochs))

    # Count the number of samples in each class and create a vector in order to create weigths for the weighted sampling
    class_sample_count_original = find_class_sample_count(original_dataset)
    print("class_sample_count_original", class_sample_count_original)
    

    # Split the data into k-times
    print("Performing k-fold cross validation...")
    for j in range(k):
        print(j, "...")

        # Split the dataset into training/test, 1-fold validation
        test_index = [j]  
        all_indices = np.linspace(0, len(original_dataset)-1, len(original_dataset)).astype(int)
        train_indices = np.delete(all_indices,test_index)
        train_dataset, test_dataset= [Subset(dataset = original_dataset, indices = train_indices),Subset(dataset = original_dataset, indices = test_index)]

        # Normalize the training/test datasets
        train_dataset = transformations_2D.ApplyNormalization(train_dataset, None, None, True)
        test_dataset = transformations_2D.ApplyNormalization(test_dataset, train_dataset.max_suvr_values, train_dataset.max_rcbf_values, False)
        
        # Apply transforms to the test and training datasets
        #Normal plot
        train_dataset = transformations_2D.ApplyTransform(train_dataset, 
                                                            sliceNr = config_2D.sliceSample, 
                                                            applyMean = config_2D.addMeanImage, 
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
                                                            applyMean = config_2D.addMeanImage, 
                                                            applyDiffnormal = config_2D.applydiffnormal,
                                                            meantrainbrain_rcbf = meannormalbrain_rcbf, 
                                                            meantrainbrain_suvr = meannormalbrain_suvr, 
                                                            useMultipleSlices = config_2D.useMultipleSlices, 
                                                            mirrorImage = config_2D.applyMirrorImage, 
                                                            gammaTransform = config_2D.gamma, 
                                                            transform = config_2D.transform_train,
                                                            randomSlice = False)

        # For unbalanced dataset we create a weighted sampler                       
        class_sample_count = class_sample_count_original.copy()
        class_sample_count.pop(j)

        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()  
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = len(train_dataset)*4, replacement = True)
        
        #seed = 2147483647 + j + 1
        #torch.manual_seed(seed
        #random.seed(seed)  # Python random module.

        # define the data loaders
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, sampler = sampler)     
        #train_data = torch.utils.data.DataLoader(train_dataset, batch_size = config_2D.batchSize, shuffle=True)
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
            prediction = torch.argmax(torch.nn.functional.softmax(trained_model.forward(x), dim=1))

            y = y.float().to(device)
            label = torch.max(y,1)[1][0] 

            # Check if we predicted the right, save result
            if torch.eq(prediction, label):
                accuracy[j,0] = 1

            accuracy[j,1] = prediction
            accuracy[j,2] = label
        
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
                                                        applyMean = config_2D.addMeanImage, 
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
                                                        applyMean = config_2D.addMeanImage, 
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
    
    # Validate the robustness of the model
    #accuracy, train_losses, test_losses = kfold(device, original_dataset)
    #print(accuracy)
    """

    """
    # Multiple plots to examine the data
    fig, axs = plt.subplots(4, 4, figsize = (15,6))
    fig, axs2 = plt.subplots(4, 4, figsize = (15,6))
    fig, axs3 = plt.subplots(4, 2, figsize = (15,6))
    fig, axs4 = plt.subplots(4, 2, figsize = (15,6))
    cmap = 'CMRmap'
    sliceNumber = 67


    for i in range(4):
        for j in range(4):
            sample2 = train_dataset.__getitem__(j+i*4)
            #sample2 = original_dataset.__getitem__(1)

            name = visFuncs_2D.get_name(sample2[1])
            axs[i, j].imshow(sample2[0][0,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs[i, j].set_title([name])
            axs[i,j].axis('off')

            axs2[i, j].imshow(sample2[0][1,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs2[i, j].set_title([name])
            axs2[i,j].axis('off')
    
    for i in range(4):
            sample2 = test_dataset.__getitem__(i)
            name = visFuncs_2D.get_name(sample2[1])

            axs3[i,0].imshow(sample2[0][1,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs3[i,1].imshow(sample2[0][3,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs3[i,0].set_title([name])
            axs3[i,1].set_title([name])
            axs3[i,0].axis('off')
            axs3[i,1].axis('off')

            axs4[i,0].imshow(sample2[0][0,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs4[i,1].imshow(sample2[0][2,:,:], cmap = cmap, vmin = 0, vmax = 1)
            axs4[i,0].set_title([name])
            axs4[i,1].set_title([name])
            axs4[i,0].axis('off')
            axs4[i,1].axis('off')

    plt.show()
    """
    # Normalize the training/test datasets
    original_dataset2 = transformations_2D.ApplyNormalization(original_dataset, None, None, True)
    
    # Apply transforms to the training/test datasets
    original_dataset2 = transformations_2D.ApplyTransform(original_dataset2, 
                                                        sliceNr = config_2D.sliceSample, 
                                                        applyMean = config_2D.addMeanImage, 
                                                        applyDiffnormal = config_2D.applydiffnormal,
                                                        meantrainbrain_rcbf = None, 
                                                        meantrainbrain_suvr = None, 
                                                        useMultipleSlices = config_2D.useMultipleSlices, 
                                                        mirrorImage = config_2D.applyMirrorImage, 
                                                        gammaTransform = config_2D.gamma, 
                                                        transform = config_2D.transform_train,
                                                        randomSlice = False)
    
    from copy import copy
    import cv2
    import matplotlib.colors as colors

    palette = copy(plt.cm.jet)
    palette.set_under('darkblue', 0.8)
    palette.set_under('black', 0.7)
    palette.set_under('black', 1.0)

    # Multiple plots to examine the data
    fig, axs = plt.subplots(4, 5, figsize = (15,6))
    fig, axs2 = plt.subplots(4, 5, figsize = (15,6))
    fig, axs3 = plt.subplots(4, 5, figsize = (15,6))
    fig, axs4 = plt.subplots(4, 5, figsize = (15,6))

    cmap = 'jet'
    for i in range(4):
        for j in range(5):
            sample2 = original_dataset2.__getitem__(j+i*5)

            name = visFuncs_2D.get_name(sample2[1])
            #axs[i, j].imshow(sample2[0][0,:,:], cmap = cmap, vmin = 0.005, vmax = 1)
            axs[i, j].imshow(sample2[0][0,:,:], cmap = palette, vmin = 0.0005, vmax = 1)  

            axs[i, j].set_title([name])
            axs[i, j].axis('off')

            #test2 = axs2[i, j].imshow(sample2[0][1,:,:], cmap = cmap, vmin = 0.005, vmax = 1)
            axs2[i, j].imshow(sample2[0][1,:,:], cmap = palette)#, vmin = 0.0005, vmax = 1)  
            axs2[i, j].set_title([name])
            axs2[i, j].axis('off')

            #images = sample2[0].numpy()
            #images[0,:,:] = cv2.flip(images[0,:,:], 0)
            #images[1,:,:] = cv2.flip(images[1,:,:], 0)

            axs3[i, j].imshow(sample2[0][2,:,:], cmap = palette, vmin = 0.0005, vmax = 1)  
            axs3[i, j].set_title([name])
            axs3[i, j].axis('off')

            #test2 = axs2[i, j].imshow(sample2[0][1,:,:], cmap = cmap, vmin = 0.005, vmax = 1)
            axs4[i, j].imshow(sample2[0][3,:,:], cmap = palette, vmin = 0.0005, vmax = 1)  
            axs4[i, j].set_title([name])
            axs4[i, j].axis('off')    

    plt.show()


    """
    # plot to save for the report
    from copy import copy
    import matplotlib.colors as colors

    # Get the sample to be saved
    sample2 = original_dataset2.__getitem__(7)
    zero_healthy = sample2[0][0,:,:]
    one_healthy = sample2[0][1,:,:]

    
    # set the colormap
    cmap = 'jet'
    palette = copy(plt.cm.jet)
    palette.set_under('darkblue', 0.8)
    palette.set_under('black', 0.7)
    palette.set_under('black', 1.0)


    fig = plt.figure(num=None, figsize=(8, 12), dpi = 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    #name = visFuncs_2D.get_name(sample2[1])
    #ax.set_title(name)


    #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
    
    image_report = ax.imshow(one_healthy, cmap = palette, vmin = 0, vmax = 1)  
    cb = plt.colorbar(image_report, cmap = palette, fraction = 0.046, pad = 0.04, shrink = 0.9)

    #cb = plt.colorbar(image_report, cmap = palette, cax = cbaxes, fraction = 0.046, pad = 0.04)
    cb.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])  # vertically oriented colorbar
    cb.ax.tick_params(labelsize=14)


    ax.remove()
    #plt.savefig("../figures/colorbar.png", bbox_inches = 'tight', pad_inches = 0)

    plt.show()

    #fig = plt.figure(num=None, figsize=(8, 6), dpi = 80, facecolor='w', edgecolor='k')
    #ax = fig.add_subplot(111)
    #ax.axis('off')
    #image_report = ax.imshow(zero_healthy, cmap = palette, vmin = 0.007, vmax = 1)  
    #plt.savefig("../figures/PD_radioactivity.eps", bbox_inches = 'tight', pad_inches = 0)

    #plt.show()
    """



    # STACKED IMAGES
    """
    from copy import copy
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from scipy.ndimage.interpolation import affine_transform


    fig = plt.figure(num=None, figsize=(16, 6), dpi = 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.axis('off')

    # set the colormap
    cmap = 'jet'
    palette = copy(plt.cm.jet)
    palette.set_under('darkblue', 0.8)
    palette.set_under('black', 0.7)
    palette.set_under('black', 1.0)

    # Specify the number of images
    nimages = 4
    img_height, img_width = 128, 128
    bg_val = -0.1 # Some flag value indicating the background.

    # Random test images.
    sample = original_dataset2.__getitem__(16)
    # Put the images into a list
    images = [sample[0][3,:,:], sample[0][2,:,:], sample[0][1,:,:], sample[0][0,:,:]]

    stacked_height = 2*img_height
    stacked_width  = int(img_width + (nimages-1) * img_width * 1.25)
    stacked = np.full((stacked_height, stacked_width), bg_val)

    # Affine transform matrix.
    T = np.array([[1, -0.7],
                [0, 0.7]])

    for i in range(nimages):
        # The first image will be right most and on the "bottom" of the stack.
        #o = (nimages-i-1) * img_width/2
        o = (nimages-i-1) * img_width / 1.3

        out = affine_transform(images[i], T, offset = [o,-o], output_shape = stacked.shape, cval = bg_val)
        stacked[out != bg_val] = out[out != bg_val]

    plt.imshow(stacked, cmap = palette, vmin = 0.0035, vmax = 1)
    plt.savefig("../figures/multiple_slices_healthy.png", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    """ 
   
   
    """
    # TRAIN THE NETWORK USING ALL SAMPLES PRETRAINED

    # Extracting features
    layer = model._modules.get('layer1')
    model.eval()


    def getFeatures(sample):
        scans, label = sample 
        scans = scans.unsqueeze(0).to(device).float() # take it to the correct form for the model
        my_embedding = torch.zeros((1,256,56,56)) # this is depending on which layer the model looks at CHANGE THIS IF WANT TO USE ANOTHER LAYER
        # 4. Define a function that will copy the output of a layer

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
            
        # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        model.forward(scans)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding.squeeze(0) #remove batch dimension for visualization

    #plot my_embedding_squeeze(0)   
    """