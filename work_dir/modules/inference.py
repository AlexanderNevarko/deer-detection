import os
import cv2
import yaml
import mmdet
import requests
import numpy as np
from collections import defaultdict
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from mmdet.apis import (
    inference_detector,
    init_detector,
)

from PIL import Image

from work_dir.modules.classificator import ClassificationNet
from work_dir.modules.detector import DetectObjects




def convert_COCO2YOLO(bbox, img_w, img_h):
    x_l, y_l, x_r, y_r = bbox
    x_c = (x_l + x_r) / 2 / img_w
    y_c = (y_l + y_r) / 2 / img_h
    h = abs(x_l - x_r) / img_w
    w = abs(y_l - y_r) / img_h
    return {
        'x_c': x_c,
        'y_c': y_c,
        'h': h,
        'w': w
    }

class Inferencer:
    
    def __init__(self, detcetor_config,
                 detector_checkpoint,
                 classifier_checkpoint) -> None:
        
            
        self.detector = DetectObjects(detector_checkpoint, detcetor_config, 'cuda')
        self.classifier = ClassificationNet(classifier_checkpoint)
        
        
    def inference(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        bboxes = self.detector.detect_all_deers(img_path, threshold=1e-3)
        
        predictions = defaultdict(list)
        
        delta = np.zeros(2, dtype=np.int_)
        for bbox in bboxes:
            x_lu, y_lu, x_rb, y_rb, sc = bbox
            bbox = [x_lu, y_lu, x_rb, y_rb]
            yolo_bbox = convert_COCO2YOLO(bbox, img_w, img_h)
            
            delta[0] = (bbox[2] - bbox[0]) // 10
            delta[1] = (bbox[3] - bbox[1]) // 10
            bbox[:2] = np.clip(bbox[:2] - delta, (0,0), (img_w-1, img_h-1)) 
            bbox[2:] = np.clip(bbox[2:] + delta, (0,0), (img_w-1, img_h-1))
            area = abs((bbox[0]-bbox[2]) * (bbox[1]-bbox[3]))
            if area <= 0:
                continue
            
            crop = img.crop(bbox)
            crop = crop.convert('RGB')
            class_label, confidence = self.classifier.inference(crop)
            confidence = confidence.detch().cpu().item()
            
            img_name = img_path.split('/')[-1]
            predictions['filename'].append(img_name)
            for k, v in yolo_bbox.items():
                predictions[k].append(v)
            predictions['class_label'].append(class_label)
            predictions['confidence'].append(confidence)
            predictions['image_width'].append(img_w)
            predictions['image_height'].append(img_h)
        return predictions
                