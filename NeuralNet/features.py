import torch
import torchvision
from torchvision import transforms
from PIL import Image

import downloader as dl

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
model.eval()

# Testing the feature extraction on an Image
filename = "test.jpeg"
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)
print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)