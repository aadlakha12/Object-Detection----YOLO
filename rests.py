#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:57:29 2020

@author: akshayadlakha
"""

import os
import cv2
import numpy as np
#import time
from model.yolo_model import YOLO

# function to reaad classes form classes.txt file in data folder
def get_classes(file):
    with open(file) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]

    return classes

# image processing
def imagepreprocessing(img):
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

# function to draw a box when an object is detected
def drawBox(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

    print()

# function to detect an object in an image    
def imageDetection(image, yolo, all_classes):

    pimage = imagepreprocessing(image)

    boxes, classes, scores = yolo.predict(pimage, image.shape)


    if boxes is not None:
        drawBox(image, boxes, scores, classes, all_classes)

    return image

# function to detect an object in video, it processes image frame by frame 
#  and detects an object in each frame
    
def videoDetection(video, yolo, all_classes):

    video_path = os.path.join("videos", "test", video)
    cam = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    sz = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    
    videoout = cv2.VideoWriter()
    videoout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = cam.read()

        if not res:
            break

        image = imageDetection(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        videoout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    videoout.release()
    cam.release()

file = 'data/classes.txt'
all_classes = get_classes(file)
yolo = YOLO(0.6, 0.5)

# image detection
filepath = 'images/'+'person.jpg'

image = cv2.imread(filepath)
image = imageDetection(image, yolo, all_classes)

# saving image after detecting
cv2.imwrite('images/res/' + 'person.jpg', image)

# video detection
video = 'veidoo.mp4'
videoDetection(video, yolo, all_classes)
