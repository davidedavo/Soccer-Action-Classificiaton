import numpy as np
from numpy.core.fromnumeric import clip
import torch
from torchvision import transforms
import time
import os
from sklearn import preprocessing

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

    





    