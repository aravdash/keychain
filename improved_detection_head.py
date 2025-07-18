import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, pool_size=(7,7), hidden_dim=1024):
        super().__init__()
        # Calculate the flattened feature size
        self.out_h, self.out_w = pool_size
        input_dim = in_channels * self.out_h * self.out_w
        
        # Two fully-connected layers with dropout for regularization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        
        # Classification layer: num_classes + 1 for background
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        
        # BBox regression layer: CLASS-AGNOSTIC (only 4 coordinates)
        # This is the critical fix - was previously 4 * (num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for classification layer
        nn.init.normal_(self.cls_score.weight, 0, 0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        
        # Special initialization for bbox regression
        nn.init.normal_(self.bbox_pred.weight, 0, 0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        # x: Tensor of shape [N, C, out_h, out_w]
        N = x.size(0)
        
        # Flatten
        x = x.view(N, -1)
        
        # FC layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Predict class logits and bbox deltas
        class_logits = self.cls_score(x)       # [N, num_classes+1]
        bbox_deltas = self.bbox_pred(x)        # [N, 4] - CLASS-AGNOSTIC
        
        return class_logits, bbox_deltas