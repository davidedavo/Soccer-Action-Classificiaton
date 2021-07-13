import numpy as np
from numpy.core.fromnumeric import clip
import torch
from torchvision import transforms
import time
import os
from sklearn import preprocessing

def loadGameActions(filePath):
    clips = np.load(f"{filePath}/features.npz")['arr_0']
    labels = np.load(f"{filePath}/labels.npz")['arr_0']
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    targets = torch.as_tensor(targets)

    return clips, targets

def preprocessImages(clips_np):
    clips = torch.from_numpy(np.swapaxes(np.swapaxes(clips_np, 2, 4), 3, 4)).type(torch.float32)
    return clips

def fit(path):
    model1 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model1 = torch.nn.Sequential(*(list(model1.children())[:-2]))
    model1.eval()
    processed_clips = 0

    championships = os.listdir(path)
    for c in championships:
        cpath = path + f"/{c}"
        years = os.listdir(cpath)
        for y in years:
            ypath = cpath + f"/{y}"
            games = os.listdir(ypath)
            for g in games:
                gpath = ypath + f"/{g}"
                clips_np, labels = loadGameActions(gpath)
                clips = preprocessImages(clips_np)

                tran = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for action in range(clips.shape[0]):
                    input_batch = tran(clips[action])
                    s = time.time()
                    with torch.no_grad():
                        output = model1(input_batch)
                        processed_clips += 1
                        st = time.time()
                        print(f"Clip: {processed_clips} Time: {round((st-s), 2)} Shape: {output.shape}")


if __name__ == "__main__":
    fit(".data")




    