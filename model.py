
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from importlib import resources
import csv
import io


class OpiDetectionDatasetR(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        with resources.path("exposuremodelf", csv_file) as df: #comment out if not in pip mode
            self.annotations = pd.read_csv(df)      #comment out if not in pip mode
        #self.annotations = pd.read_csv(csv_file)   #Uncomment if not in pip mode
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 3])
        img = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(float((self.annotations.iloc[index, 2]) - 1) / 4) # 2 for "AvgRating" column
        #Subtracting by one then dividing by 4 normalizes the data to be from 0 to 1.
        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

def load(num_epochs, data_dir):
    cudnn.benchmark = True
    plt.ion()   


    input_size = 224
    batch_size = 32

    test_transforms = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ]
    )

    print("Initializing Datasets and Dataloaders...")


    dataset = OpiDetectionDatasetR(csv_file = 'ratings.csv', root_dir = data_dir, transform = test_transforms)
    dataset_size = dataset.__len__()
    print ("Dataset size =", dataset_size)


    training_size = 1100
    testing_size = dataset_size - training_size

    dataset_sizes = {'train':training_size, 'val':testing_size}
    train_set, test_set = torch.utils.data.random_split(dataset,[training_size,testing_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Dictionary with training / testing data
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = test_loader


    device = torch.device('cpu')

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    #model_ft.load_state_dict(torch.load('model.pth'))

    model = model.to(device)

    criterion = nn.MSELoss()


    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    num_epochs = num_epochs


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 1000.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        img_num = 0


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                labels = torch.FloatTensor(labels)
                img_num += len(labels.data.tolist())

                inputs = inputs.to(device)
                labels = labels.to(device)      
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs) # Constraining outputs to be in range [0, 1]

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.float().unsqueeze(1))

                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f}')


            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Loss: {lowest_loss:4f}')


    model.load_state_dict(best_model_wts)
    PATH = 'model.pth'
    torch.save(model.state_dict(), PATH)




def eval(imgs, pretrain = True):
    if pretrain:
        model_path = "p_model.pth" # Change as needed
    else:
        model_path = "model.pth"
    try:
        device = torch.device("cpu")
        model_ft = models.resnet18()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)
        with resources.path("exposuremodelf", model_path) as pp:
            model_ft.load_state_dict(torch.load(pp))
        model_ft = model_ft.to(device)

        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        predictions = []
        for img in imgs:
            model_ft.eval()
            pic = Image.open(img)
            pic = preprocess(pic)
            pic = pic.unsqueeze(0)
            #fig = plt.figure()
            with torch.no_grad(): # not training, no need for gradients
                output_tensor = model_ft(pic)
                #prediction = output_tensor
                prediction = torch.sigmoid(output_tensor) # Constraining output to be in range
                prediction = prediction.squeeze(1).tolist()[0]
                predictions.append(prediction)
                print(f'predicted: {prediction}')
        return (predictions)
    except:
        print("No model found. Load new model with model.load(num_epochs)")



def test():
    image_path = "test.jpg"
    eval( [image_path])

if __name__ == '__main__':
    load(1, "Data")


