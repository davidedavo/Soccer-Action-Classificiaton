import numpy as np
from numpy.core.fromnumeric import clip
import torch
from torchvision import transforms
import time
import os
from sklearn import preprocessing
import torchvision
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = torchvision.models.resnet18(pretrained=True)
        #self._model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self._model = torch.nn.Sequential(*(list(self._model.children())[:-2]))
        #self._model.eval()

    def forward(self, data):
        #with torch.no_grad():
        dim_0 = data.shape[0]
        dim_1 = data.shape[1]
        flatt = data.flatten(start_dim=0, end_dim=1)
        
        output = self._model(flatt) # output.shape = [30, 512, 7, 13]
        output = output.reshape(dim_0, output.shape[0], output.shape[1], -1)
        output = output.permute(0, 2, 1, 3)
        output = output.flatten(start_dim=2)
        print(output.shape)
        return output

def preprocessImages(clips_np):
    clips = torch.from_numpy(np.swapaxes(np.swapaxes(clips_np, 2, 4), 3, 4)).type(torch.float32)
    return clips

def fit(data):
    '''
        Args:
            data: Dummy arg for clips, needs to be replaced with data coming from dataLoader
    '''

    model1 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model1 = torch.nn.Sequential(*(list(model1.children())[:-2]))
    model1.eval()
    processed_clips = 0

    # Need to swapaxes in order to get torch-compatible shape
    clips = preprocessImages(data)

    tran = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for action in range(clips.shape[0]):
        input_batch = tran(clips[action]) # Input batch = 1 clip = 30 frames
        s = time.time()
        with torch.no_grad():
            output = model1(input_batch) # output.shape = [30, 512, 7, 13]
            processed_clips += 1
            st = time.time()
            print(f"Clip: {processed_clips} Time: {round((st-s), 2)} Shape: {output.shape}")

    # Add here the classifier using "output" as the network input






    