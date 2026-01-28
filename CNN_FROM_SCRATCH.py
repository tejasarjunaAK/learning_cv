import torch
import torch.nn as nn
from torchvision import models, datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
class cnn_scratch(nn.Module):
    train_transform=transforms.Compose([transforms.RandomHorizontalFlip,transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    test_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    def __init__(self):
        super().__init__()
        self.train_set=datasets.CIFAR10('./data',train=True,download=True,transform=cnn_scratch.train_transform)
        self.test_set=datasets.CIFAR10('./data',download=True,train=False,transform=cnn_scratch.test_transform)
        self.train_loader=DataLoader(self.train_set,shuffle=True,batch_size=16)
        self.test_loader=DataLoader(self.test_set,batch_size=16,shuffle=False)
        self.layer1=nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2,2))
        self.layer2=nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2,2))
        self.layer3=nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2))
        self.fc1=nn.Linear(64*4*4,256)
        self.fc2=nn.Linear(256,10)
        self.Flatten=nn.Flatten()
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.Flatten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x
model=cnn_scratch().to(device)
criterion=nn.CrossEntropyLoss()
optimize=optim.Adam(model.parameters(),lr=0.001)
model.train()
for epoch in range(10):
    total_loss=0
    for images,labels in model.train_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        loss=criterion(output,labels)
        optimize.zero_grad()
        loss.backward()# these are the four lines which have to be remembered for backpropagation
        optimize.step()
        total_loss+=loss.item()
    print(f'Epoch -{epoch+1}, Loss-{total_loss/len(model.train_loader)}')
model.eval()
with torch.no_grad():
    correct=0
    for images,labels in model.test_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        scores,predicted=torch.max(output,1)# torch.max will take input as tensor , dimension
        correct+=(predicted==labels).sum().item()
    print(f'Accuracy is {correct/len(model.test_set)*100}')
     
        
        