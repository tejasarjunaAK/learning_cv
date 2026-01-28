import torch 
import torch.nn as nn
from torchvision import models,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import webdataset as wds

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
model=models.resnet18(pretrained=True)
criterion=nn.CrossEntropyLoss()
model.fc=nn.Linear(model.fc.in_features,2)
for param in model.parameters():
    param.requires_grad=False
for param in model.layer4.parameters():
    param.requires_grad=True
for param in model.fc.parameters():
    param.requires_grad=True
optimize=optim.Adam([{'params':model.layer4.parameters(),'lr':1e-4},{'params':model.fc.parameters(),'lr':1e-3}])
scheduler = optim . lr_scheduler . StepLR ( optimize, step_size =10 , gamma
=0.1)
def dataset_loading():
    train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    test_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    train_set=(wds.WebDataset("catsdogs-{000000..000023}.tar").shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
    .map_tuple(train_transform, int))
    test_set=(wds.WebDataset("catsdogs-{000024..000029}.tar").decode("pil")
        .to_tuple("jpg","cls")
        .map_tuple(test_transform, int))
    train_loader=DataLoader(train_set,shuffle=False,batch_size=32)
    test_loader=DataLoader(test_set,shuffle=False,batch_size=32)
    return train_loader,test_loader
def train_data(train_loader,test_loader,model):
    model=model.to(device)
    model.train()
    for epoch in range(10):
        loss_per_epoch=0
        total=0
        for images,labels in train_loader:
            images=images.to(device)
            labels=labels.to(device)
            output=model(images)
            loss=criterion(output,labels)
            optimize.zero_grad()
            loss.backward()
            optimize.step()
            loss_per_epoch+=loss.item()
            total+=labels.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1},Loss:{loss_per_epoch/total}")
        model.eval()
        correct=0
       
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            output=model(images)
            _,predicted=torch.max(output,1)
            correct+=(predicted==labels).sum().item()
            
        accuracy=correct/len(train_loader)
        print(f"Test Accuracy after epoch {epoch+1}:{accuracy*100}%")
def main():
    train_loader,test_loader=dataset_loading()
    train_data(train_loader,test_loader,model)  
main()                     
     
        
        