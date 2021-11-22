import torch 
import timm
import albumentations as A
import numpy as np
import cv2
import torch.nn as nn
from albumentations.pytorch import ToTensorV2

class Transforms:
    
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        return self.transforms(image=np.array(img))['image']



class ClassificationNet:
    def __init__(self, checkpoint, out_features=2, device='cuda'):
        self.model = timm.create_model('vit_small_patch16_224_in21k', pretrained=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.head = nn.Linear(in_features=self.model.head.in_features, out_features=out_features)
        self.model.head.requires_grad_(False)
        self.model.load_state_dict(torch.load(checkpoint))
        self.device = torch.device(device)
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
        self.softmax = nn.Softmax()
        
    def inference(self, img):
        torch_img = self.transform(img).unsqueeze(0).to(self.device)
        pred = self.model(torch_img)[0]
        probs = self.softmax(pred)
        class_idx = self.model(torch_img)[0].argmax(dim=0).detach().cpu().numpy()
        class_text = self.mapping[class_idx]
        return class_text, probs[class_idx]
