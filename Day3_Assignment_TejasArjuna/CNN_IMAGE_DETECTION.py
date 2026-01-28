import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms,models
import torch.optim as optim
import torch.utils
from  torch.utils.data import Dataloader
train_transform=transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.Normalize(),transforms.ToTensor()])
test_transform=transforms.Compose([transforms.Resize(256),transforms.Crop(224),transforms.Normalize(),transforms.ToTensor()])
train_dataset=datasets.ImageFolder("path",transform=train_transform)

model =models.resnet18(pretrained=True)
criterion=nn.CrossEntropyLoss()
for param in model.parameters():
    param.requires_grad=False
model.fc=nn.Linear(model.fc.in_features,2)
print(f"weights random initial {model.weights()}")
for epoch in range(10):
    train_data=Dataloader(train_dataset,batch_size=16,shuffle=True)
    for images,labels in train_data:
       outputs=model(images)
       
    

