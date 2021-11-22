import os
import cv2
import yaml
import mmdet
import requests
import numpy as np
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




def convert_COCO2YOLO(bbox):
    pass

class Detector:
    
    def __init__(self, config) -> None:
        with open(config, 'r') as file:
            kwargs = yaml.safe_load(file)
            
        self.detector = init_detector(**kwargs['Detector'])
        
        self.device = torch.device('cuda')
        self.classifier = timm.create_model('vit_small_patch16_224_in21k', pretrained=True)
        self.classifier.head = nn.Linear(in_features=self.classifier.head.in_features, out_features=kwargs['Classifier']['out_features'])
        self.classifier.load_state_dict(torch.load(kwargs['Classifier']['checkpoint']))
        self.classifier.eval()
        self.classifier.to(self.device)
        
        transform_test = A.Compose([
                A.LongestMaxSize(max_size=224),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                A.PadIfNeeded(
                    position=A.PadIfNeeded.PositionType.CENTER,
                    min_height=224,
                    min_width=224,
                    value=0,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                ToTensorV2(),
        ])
        self.transform = Transforms(transform_test)
        
        self.mapping = {
            0: 'fawn',
            1: 'reindeer'
        }
        
        
    def inference(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        preds = inference_detector(
            model = self.detector,
            imgs = np.asarray(img)
        )
        
        bboxes = []
        delta = np.zeros(2, dtype=np.int_)
        for class_pred in preds:
            for bbox in class_pred:
                x_lu, y_lu, x_rb, y_rb, sc = bbox
                bbox = [x_lu, y_lu, x_rb, y_rb]
                delta[0] = (bbox[2] - bbox[0]) // 10
                delta[1] = (bbox[3] - bbox[1]) // 10
                bbox[:2] = np.clip(bbox[:2] - delta, (0,0), (img_w-1, img_h-1)) 
                bbox[2:] = np.clip(bbox[2:] + delta, (0,0), (img_w-1, img_h-1))
                area = abs((bbox[0]-bbox[2]) * (bbox[1]-bbox[3]))
                if area <= 0:
                    continue
                
                crop = img.crop(bbox)
                crop = crop.convert('RGB')
                torch_crop = self.transform(crop).unsqueeze(0).to(self.device)
                class_idx = self.classifier(torch_crop)[0].argmax(dim=0)
                text_class = self.mapping[class_idx]
                