import torchvision.models as models
import torch.nn as nn

def resnet():
    model = models.resnet18(pretrained=True) # pretained and settin number of classes
    model.fc = nn.Linear(512,6)
    #changing the initial layer from 3 channels to 160
    inChannels = 160
    model.conv1 = nn.Conv2d(inChannels,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    #Freezing layers that are already trained, perhaps freeze layer 

    for layers in [model.layer1, model.layer2, model.layer3]:
        layers.requires_grad = False

    return model
    
if __name__ == "__main__":
    mod = resnet()
    print(mod)
