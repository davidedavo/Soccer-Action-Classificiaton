from classifier import CNN_1d
import torch.nn as nn
import torch
import torch.optim as optim

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

def readFeatures():
    pass

if __name__ == '__main__':
    features, labels = readFeatures()
    num_classes = len(labels.unique())
    model = MODEL(num_classes).to(DEVICE)# optimizer
    

    fit(model, None)