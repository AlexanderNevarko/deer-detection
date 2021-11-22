import torch 
import timm
import albumentations as A


class ClassificationNet:
  def __init__(self):
    self.model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True)
    self.model.train()
    for param in self.model.parameters():
        param.requires_grad_(False)
    self.model.head = nn.Linear(in_features=self.model.head.in_features, out_features=2)
    self.model.head.requires_grad_(False)
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.transforms = A.Compose([
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