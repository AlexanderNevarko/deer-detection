from mmdet.apis import inference_detector, init_detector
import numpy as np
from PIL import Image

class DetectObjects:
	
	def __init__(self):
		self.checkpoint = '../work_dir/pretrained/epoch_48.pth'
		self.config = 'vfnet_aug_v2.py'
		self.detect_instances = init_detector(self.config, checkpoint=self.checkpoint, device='cuda')
	
	def select_deer(self, predictions, taget_classes='deer', treshold_confidence=1e-5):
		bboxes, masks = predictions
		target_bboxes, target_masks = [], []
		for target_idx in self.select_classes[taget_classes]:
			filtered_bboxes, filtered_masks = [], []
            		for bbox in bboxes[target_idx]:
                		if len(bbox) == 0:
                    			continue
                		x_l, y_l, x_r, y_r, sc = bbox
                		if sc > treshold_confidence:
                    			filtered_bboxes.append(bbox)
					filtered_masks.append(bbox)

			target_bboxes.append(np.array(filtered_bboxes))
			target_masks.append(np.array(filtered_masks))
		return (target_bboxes, target_masks)

	def detect_all_deers(self, image_path, threshold):
		image_pil = Image.open(image_path).convert('RGB')

		predictions = inference_detector(
            		model=self.detect_instances,
            		imgs=np.array(image_pil),
        	)
		
		deers_bbox, deers_mask = self.select_deer(predictions=predictions,
							taget_classes='deer',
							treshold_confidence=threshold)
		final_bbox, final_masks = [], []
		for i in range(len(deers_bbox)):
			x_l, y_l, x_r, y_r, sc = deers_bbox[i][0]
			mask = deers_mask[i][0]
			x_c = (x_l + x_r) / 2
			y_c = (y_l + y_r) / 2
			h = abs(x_l - x_r) / 2
			w = abs(y_l - y_r) / 2
			final_bbox.append((x_c, y_c, h, w))
			final_masks.append(mask)
		return final_bbox, final_masks 		
