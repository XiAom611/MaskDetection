import numpy as np 
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model

def plot_image(img, annotation, threshold=None, flag=True):
    if not flag:
        return img

    framecolor={1: (0, 255, 0), 2: (0, 0, 255), 0: (0, 0, 0)} # bgr
    annotation = annotation[0]
    
    for box, label, score in zip(annotation["boxes"], annotation["labels"], annotation["scores"]):
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        label = int(label)
        score = float(score)

        if threshold and score < threshold:
            continue

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), framecolor[label], 2)
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(score[:4]), (xmin, ymin), label_font, 0.9, framecolor[label], 2)

    return img

if __name__ == '__main__':
    # load model weights
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load("src/model2.pt"))

    model.eval()

    cam = cv2.VideoCapture(0)
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    flag = False
    while True:
        check, origin_frame = cam.read()
        origin_frame = cv2.resize(origin_frame, (640, 320))
        pred = None
        if flag:
            frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = trans(frame)
            frame = [torch.Tensor(frame)]
            pred = model(frame)
        new_frame = plot_image(origin_frame, pred, 0.5, flag)
        cv2.imshow("video", new_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('1'):
            flag = True
            print("Start detect mask")
        elif key == ord('2'):
            flag = False
            print("Stop detect mask")

    cam.release()
    cv2.destroyAllWindows()
