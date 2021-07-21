from classifier import CNN_1d
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  TensorDataset, DataLoader, random_split, Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from extraction import FeatureExtractor

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
LR = 0.1
MODEL = CNN_1d
PATH = f".model/{MODEL.__name__}.pth"
print(PATH)


class BigDataset(torch.utils.data.Dataset):
    
    def __init__(self, labels, path):
        'Initialization'
        self.labels = labels
        self.list_IDs = range(len(labels))
        self.path = path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(f'{self.path}/{ID}.pt')
        y = self.labels[ID]

        return X, y

class CustomImageDataset(Dataset):#TODO Change X_file, Y_file
    def __init__(self, X_file = ".data/compressed/0.pt", Y_file = ".data/compressed/0_lab.pt", transform=None, target_transform=None):
        self.X = torch.load(X_file).permute(0, 1,-1, 2, 3).float()
        self.Y = torch.load(Y_file).long()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        clip = self.X[idx]
        label = self.Y[idx]
        if self.transform:
            clip = self.transform(clip)
        if self.target_transform:
            label = self.target_transform(label)
        return clip, label

    def getNumClasses(self):
        return len(self.Y.unique())


class Classifier:

    def __init__(self, num_classes, model=CNN_1d):
        self._extractor = FeatureExtractor().to(DEVICE)
        self._model = model(num_classes).to(DEVICE)


    def fit(self, train_loader, num_epochs=10, log_interval = 50, force_training = False):
        if not force_training:
            try:
                self._model.load_state_dict(torch.load(PATH))
                return
            except FileNotFoundError as e:
                pass

        print('Training')
        optimizer = optim.Adam(self._model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        self._model.train()
        self._extractor.eval()
        for epoch in range(num_epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    with torch.no_grad():
                        features = self._extractor(data)
                    print(features)
                    print(features.shape)
                    optimizer.zero_grad()  # zero the gradient buffers
                    outputs = self._model(features)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()  # Does the update
                    if batch_idx % log_interval == 0:
                        print(
                            f'Train Epoch: {epoch+1}, idx:{batch_idx} \tLoss: {loss.item()}')

        torch.save(self._model.state_dict(), PATH)


    def eval(self, test_loader):
        self._model.eval()
        correct = 0
        predictions = torch.Tensor([])
        with torch.no_grad():
            for data, target in test_loader:
                features = self._extractor.fit(data)
                output = self._model(features)
                pred = output.data.max(1, keepdim=True)[1]
                pred_unsq = torch.squeeze(pred)
                predictions = torch.cat((predictions, pred_unsq))
                correct += pred.eq(target.data.view_as(pred)).sum()
        
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return correct, predictions.int()

"""def readFeatures():
    data = torch.load(".data/compressed/0.pt")
    lab = torch.load(".data/compressed/0_lab.pt")
    return data, lab
"""

def splitDataset(dataset, train_perc = 0.8):
    N = len(dataset)
    n_train = round(N*train_perc)
    n_test = N - n_train
    train, test = random_split(dataset, [n_train, n_test])
    return train, test

def createDataLoader(dataset):
    return DataLoader(dataset, batch_size=1)#TODO CHANGE THIS


if __name__ == '__main__':
    #dataset = CustomImageDataset(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    labels = torch.load(".dataset/labels.pt")
    dataset = BigDataset(labels, ".data")
    
    #features, labels = readFeatures()
    #print(features.shape, labels.shape)
    #tds = TensorDataset((features, labels), transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    #num_classes = len(labels.unique())
    #model = MODEL(num_classes).to(DEVICE)# optimizer
    num_classes = dataset.getNumClasses()

    train, test = splitDataset(dataset)
    
    train_loader, test_loader = createDataLoader(train), createDataLoader(test)

    classifier = Classifier(num_classes)

    classifier.fit(train_loader)

    classifier.eval(test_loader)