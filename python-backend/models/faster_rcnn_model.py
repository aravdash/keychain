import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class KeychainFasterRCNN:
    """
    Faster R-CNN model for keychain object detection and classification
    
    In a real implementation, this would be trained on a dataset of keychain designs
    with bounding boxes and class labels for different keychain types.
    """
    
    def __init__(self, num_classes=10, model_path=None):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path, optimizer=None, epoch=None):
        """Save model weights"""
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
        """Preprocess image for model input"""
        # Convert PIL image or numpy array to tensor
        if isinstance(image, torch.Tensor):
            return image
        
        # Normalize to [0, 1] range if needed
        if image.max() > 1:
            image = image / 255.0
        
        # Convert to tensor and add batch dimension
        if len(image.shape) == 3:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # HWC to CHW
        
        return image
    
    def predict(self, image, confidence_threshold=0.5):
        """
        Predict objects in image
        
        Args:
            image: Input image (PIL, numpy array, or tensor)
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            Dictionary with boxes, labels, scores, and features
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # Add batch dimension if needed
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference
            predictions = self.model([image_tensor])
            
            # Filter by confidence
            prediction = predictions[0]
            keep = prediction['scores'] > confidence_threshold
            
            filtered_prediction = {
                'boxes': prediction['boxes'][keep].cpu().numpy(),
                'labels': prediction['labels'][keep].cpu().numpy(),
                'scores': prediction['scores'][keep].cpu().numpy(),
            }
            
            return filtered_prediction
    
    def extract_features(self, image):
        """
        Extract feature embeddings for similarity search
        
        This would extract features from the backbone network
        for use in template matching and similarity search.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Extract features from backbone
            features = self.model.backbone(image_tensor)
            
            # Global average pooling to get fixed-size feature vector
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(
                features['pool'], (1, 1)
            ).flatten(1)
            
            return pooled_features.cpu().numpy()

# Keychain class labels for the model
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
    """
    Load a trained Faster R-CNN model for keychain detection
    
    In production, this would load a model trained specifically
    on keychain designs with proper labels and bounding boxes.
    """
    model = KeychainFasterRCNN(
        num_classes=len(KEYCHAIN_CLASSES),
        model_path=model_path
    )
    
    return model