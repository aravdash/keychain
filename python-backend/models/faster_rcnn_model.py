import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class KeychainFasterRCNN:
    def __init__(self, num_classes=10, model_path=None):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path, optimizer=None, epoch=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch:
            checkpoint['epoch'] = epoch
            
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
    
    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            return image
        
        if image.max() > 1:
            image = image / 255.0
        
        if len(image.shape) == 3:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
        
        return image
    
    def predict(self, image, confidence_threshold=0.5):
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            predictions = self.model([image_tensor])
            
            prediction = predictions[0]
            keep = prediction['scores'] > confidence_threshold
            
            filtered_prediction = {
                'boxes': prediction['boxes'][keep].cpu().numpy(),
                'labels': prediction['labels'][keep].cpu().numpy(),
                'scores': prediction['scores'][keep].cpu().numpy(),
            }
            
            return filtered_prediction
    
    def extract_features(self, image):
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            features = self.model.backbone(image_tensor)
            
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(
                features['pool'], (1, 1)
            ).flatten(1)
            
            return pooled_features.cpu().numpy()

KEYCHAIN_CLASSES = [
    'background',
    'heart',
    'star',
    'circle',
    'square',
    'triangle',
    'lightning',
    'flower',
    'animal',
    'letter'
]

def load_trained_model(model_path=None):
    model = KeychainFasterRCNN(
        num_classes=len(KEYCHAIN_CLASSES),
        model_path=model_path
    )
    
    return model