import torchvision.models as models
import torch.nn as nn
import config_2D

def resnet():
    # Model with 18 layers
    #model = models.resnet18(pretrained = True) # pretained and settin number of classes
    #model.fc = nn.Linear(512, config_2D.nrOfDifferentDiseases)
    
    # Model with 152 layers
    model = models.resnet152(pretrained = True) # pretained and settin number of classes
    model.fc = nn.Linear(2048, config_2D.nrOfDifferentDiseases)


    #changing the initial layer from 3 channels to 160
    inChannels = config_2D.inChannels
    model.conv1 = nn.Conv2d(inChannels, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

    #Freezing layers that are already trained, perhaps freeze layer 

    for layers in [model.layer1, model.layer2, model.layer3, model.layer4]:
        layers.requires_grad = False

    return model
    
if __name__ == "__main__":
    mod = resnet()
    print(mod)
