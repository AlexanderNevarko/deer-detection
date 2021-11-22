from mmdet.apis import inference_detector, init_detector
import numpy as np
from PIL import Image

class DetectObjects:
	
    def __init__(self, checkpoint='', config='', device=''):
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self.detector = init_detector(self.config, checkpoint=self.checkpoint, device=self.device)

    def select_deers(self, preds, threshold_confidence=1e-3):
        final_bboxes = []
        for class_pred in preds:
            for bbox in class_pred:
                x_lu, y_lu, x_rb, y_rb, sc = bbox
                if sc >= threshold_confidence:
                    final_bboxes.append(bbox)
        return final_bboxes

    def detect_all_deers(self, image_path, threshold=1e-3):
        image_pil = Image.open(image_path).convert('RGB')

        predictions = inference_detector(
                    model=self.detector,
                    imgs=np.array(image_pil),
            )
        
        deers_bboxes = self.select_deers(predictions, threshold)
        return deers_bboxes
