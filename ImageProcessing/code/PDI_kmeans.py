# player detection with kmeans for color detection
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def feature_hist(image):
    avg = np.mean(image, axis=(0,1))
    return avg

# thresholds
conf_th = 0.6
nms_th = 0.4

# COCO classes
classes = []
classesFile = 'res/YOLOcfg/coco.names'
with open(classesFile, 'r') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Weights and configurations
Cfg = 'res/YOLOcfg/yolov3.cfg'
Weights = 'res/YOLOcfg/yolov3.weights'

CfgTiny = 'res/YOLO-tiny/yolov3-tiny.cfg'
WeightsTiny = 'res/YOLO-tiny/yolov3-tiny.weights'

# Load yolo
net = cv2.dnn.readNet(Weights, Cfg)
print("Net loaded")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)

# Load images
img = cv2.imread("res/img_01.jpg")
copy = img.copy()
height, width, ch = img.shape

scaling = 1/255
blob = cv2.dnn.blobFromImage(img, scaling, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outputs:
    for detection in out:
        scores = detection[5:]
        id_class = np.argmax(scores)
        confidence = scores[id_class]
        if confidence > conf_th:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(id_class)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)

font = cv2.FONT_HERSHEY_PLAIN

feature_vector = []
count = 1
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        conf = str(round(confidences[i], 2))
        if label == 'person':
            player = img[y:y+h, x:x+w]
            cv2.putText(img, str(count), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            vector = feature_hist(player)
            feature_vector.append(vector)
            count = count + 1

feature_vector = np.array(feature_vector)
clf = KMeans(n_clusters=3)
predict = clf.fit_predict(feature_vector)
print(predict)
print(predict.shape)

count = 0
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        conf = str(round(confidences[i], 2))
        if label == 'person':
            player = img[y:y+h, x:x+w]
            if predict[count] == 0:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
            if predict[count] == 1:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0),1)
            if predict[count] == 2:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)
            count = count + 1

cv2.imshow("Img", img)
cv2.waitKey(0)