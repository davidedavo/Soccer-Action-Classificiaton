from classifier import CNN_1d
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd

LR = 0.1
DEVICE = 'cuda:0'
MODEL = CNN_1d
PATH = f".model/{MODEL.__name__}.pth"
print(PATH)

def fit(model, train_loader, num_epochs=10, log_interval = 50, force_training = False):
    if not force_training:
        try:
            model.load_state_dict(torch.load(PATH))
            return
        except FileNotFoundError as e:
            pass

    print('Training')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()  # zero the gradient buffers
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()  # Does the update
                if batch_idx % log_interval == 0:
                    print(
                        f'Train Epoch: {epoch+1}, idx:{batch_idx} \tLoss: {loss.item()}')

    torch.save(model.state_dict(), PATH)


def eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    predictions = torch.Tensor([])
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_unsq = torch.squeeze(pred)
            predictions = torch.cat((predictions, pred_unsq))
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct, predictions.int()

def readFeatures():
    Y = np.load(".data/compressed/Y_1626087305931.npz")["arr_0"]
    label_unique = np.sort(np.unique(Y))
    df = pd.DataFrame(Y)
    df[0] = pd.Categorical(df[0], categories=label_unique)
    Y = df[0].cat.codes
    return torch.zeros(10,5,2), Y, label_unique

def splitDataset(dataset, train_perc = 0.8):
    N = len(features)
    n_train = N*train_perc
    n_test = N - n_train
    train, test = random_split(dataset, [n_train, n_test])
    return train, test

def createDataLoader(dataset):
    return DataLoader(dataset)


if __name__ == '__main__':
    features, labels, categories = readFeatures()
    print(features.shape, labels.shape)
    tds = TensorDataset(features, labels)
    
    num_classes = len(labels.unique())
    model = MODEL(num_classes).to(DEVICE)# optimizer
    
    train, test = splitDataset(tds)
    train_loader, test_loader = createDataLoader(train), createDataLoader(test)

    fit(model, train_loader)

    eval(model, test_loader)