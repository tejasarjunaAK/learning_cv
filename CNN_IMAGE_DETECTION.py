import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms,models
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import os
checkpoint="checkpoint.pth"
def save_checkpoint(epoch,model,optimizer,accuracy,path):
    torch.save({"epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"best_accuracy":accuracy},path)
start_epoch=0
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform=transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
test_transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
train_dataset=datasets.ImageFolder("path",transform=train_transform)
train_data=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_set=datasets.ImageFolder("path",transform=test_transform)
test_loader=DataLoader(test_set,batch_size=16,shuffle=False)#onl train loader mmust be shuffled not test loader
model =models.resnet18(pretrained=True)
criterion=nn.CrossEntropyLoss()
for param in model.parameters():
    param.requires_grad=False
for param in model.layer4.parameters():
    param.requires_grad=True
model.fc=nn.Linear(model.fc.in_features,2)
model=model.to(device)
optimize=optim.SGD([{"params":model.layer4.parameters(),"lr":0.0001},{"params":model.fc.parameters(),"lr":0.001}],momentum=0.9)
print(f"weights random initial {model.fc.weight}")
best_accuracy=0
if os.path.exists(checkpoint):
    cpt = torch.load(checkpoint, map_location=device)

    model.load_state_dict(cpt["model_state"])
    optimize.load_state_dict(cpt["optimizer_state"])

    start_epoch = cpt["epoch"] + 1
    best_accuracy =cpt["best_accuracy"]

    print(f" Loaded checkpoint from epoch {start_epoch}")
else:
    print(" No checkpoint found")
for epoch in range(start_epoch,start_epoch+10):
    correct=0
    
    model.train()
    for images,labels in train_data:
       images=images.to(device)
       labels=labels.to(device)
       outputs=model(images)
       loss=criterion(outputs,labels)
       optimize.zero_grad()
       loss.backward()
       optimize.step()
    model.eval()
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        with torch.no_grad():
            outputs=model(images)#output will be the array of 16 arrays and each array with amount of resemblance to class
            outputs=outputs.argmax(dim=1)#since the image has scores for each class and we neeed to take the maximum close one
            correct+=(outputs==labels).sum().item()
    accuracy=correct/len(test_set)*100
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        save_checkpoint(epoch,model,optimize,accuracy,checkpoint)

        


            
       
    

