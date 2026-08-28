import torch
import timm
from PIL import Image
import numpy as np
from torchvision import transforms

class TimmClassificationModel:
    def __init__(self, model_name: str, weights: None, num_classes: int = 1, device: torch.device = None,):
        """
        Wrapper for creating and managing a classification model using timm.

        :param model_name: Name of the model architecture from timm.
        :param device: PyTorch device to move the model to. Defaults to 'cuda' if available.
        :param num_classes: Number of output classes. Default is 1.
        :param pretrained: Whether to load pretrained weights. Default is True.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(weights, map_location=self.device), strict=True)
        self.model.to(self.device).eval()
        self.transform = self.default_transforms()


    def predict(self, images: list[np.ndarray]):
        """
        Accepts a list of numpy images (HWC, uint8 or float),
        converts them to PIL Images, applies transforms, and runs inference.
        """
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        probs = []
        for img in pil_images:
            img = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
            with torch.no_grad():
                logit = self.model(img)
                prob = torch.sigmoid(logit).squeeze().cpu().item()

            probs.append(prob)

        return probs

    @staticmethod
    def default_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

