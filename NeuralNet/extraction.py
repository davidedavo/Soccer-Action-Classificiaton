import numpy as np
from numpy.core.fromnumeric import ptp
import torch
from torchvision import transforms

def loadData(data_path, label_path=None):
    X = np.load(data_path, allow_pickle=True)['arr_0']
    #Y = np.load(label_path, allow_pickle=True)['arr_0']
    return X

def preprocessImages(data):
    data = np.swapaxes(np.swapaxes(data, 2, 4), 3, 4)
    data = torch.from_numpy(data).type(torch.float32)
    return data

def extractFeatures(data):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model.eval()

    tensor_list = []

    tran = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for action in range(data.shape[0]):
        net_output = []
        for frame in range(data.shape[1]):
            data[action, frame, :, :, :] = tran(data[action,frame,:,:,:])

            input_batch = data[action, frame, :, :, :].unsqueeze(0)
            with torch.no_grad():
                net_output.append(model(input_batch))
        net_output = torch.cat(net_output)
        print(f"Clip: {action}, Shape: {net_output.shape}")
        tensor_list.append(net_output)
    
    tensor_list = torch.stack(tensor_list)
            
    return tensor_list

if __name__ == "__main__":
    X_path = ".data/X.npz"
    #Y_path = "Y.npz"
    data = loadData(X_path)
    data = preprocessImages(data)
    features = extractFeatures(data)
    torch.save(features, ".data/features/features.pt")




    