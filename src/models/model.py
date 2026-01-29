"""
Model architectures for COVID-19 diagnosis
Supports multiple backbone architectures with transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights
)


class COVID19Model(nn.Module):
    """
    Transfer learning model for COVID-19 diagnosis
    Supports: ResNet50, DenseNet121, EfficientNet-B0
    """
    
    def __init__(self, num_classes=4, model_name='resnet50', pretrained=True, dropout=0.5):
        """
        Args:
            num_classes (int): Number of output classes
            model_name (str): Backbone architecture ('resnet50', 'densenet121', 'efficientnet_b0')
            pretrained (bool): Use ImageNet pretrained weights
            dropout (float): Dropout rate
        """
        super(COVID19Model, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Build backbone
        if model_name == 'resnet50':
            self.backbone = self._build_resnet50(pretrained)
        elif model_name == 'densenet121':
            self.backbone = self._build_densenet121(pretrained)
        elif model_name == 'efficientnet_b0':
            self.backbone = self._build_efficientnet_b0(pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"✅ Built {model_name} with {num_classes} output classes")
    
    def _build_resnet50(self, pretrained):
        """Build ResNet50 backbone"""
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            print("  Using ImageNet pretrained weights")
        else:
            model = models.resnet50(weights=None)
            print("  Training from scratch")
        
        # Get number of features
        num_features = model.fc.in_features
        
        # Replace final layer with custom classifier
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def _build_densenet121(self, pretrained):
        """Build DenseNet121 backbone"""
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
            print("  Using ImageNet pretrained weights")
        else:
            model = models.densenet121(weights=None)
            print("  Training from scratch")
        
        # Get number of features
        num_features = model.classifier.in_features
        
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def _build_efficientnet_b0(self, pretrained):
        """Build EfficientNet-B0 backbone"""
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            print("  Using ImageNet pretrained weights")
        else:
            model = models.efficientnet_b0(weights=None)
            print("  Training from scratch")
        
        # Get number of features
        num_features = model.classifier[1].in_features
        
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        return model
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


class BaselineCNN(nn.Module):
    """
    Simple baseline CNN for comparison
    Built from scratch (no transfer learning)
    """
    
    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        print(f"✅ Built Baseline CNN with {num_classes} output classes")
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_name='resnet50', num_classes=4, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name (str): Model architecture ('baseline', 'resnet50', 'densenet121', 'efficientnet_b0')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights (ignored for baseline)
        
    Returns:
        nn.Module: PyTorch model
    """
    if model_name == 'baseline':
        return BaselineCNN(num_classes=num_classes)
    else:
        return COVID19Model(
            num_classes=num_classes, 
            model_name=model_name, 
            pretrained=pretrained
        )


# Test module
if __name__ == "__main__":
    print("Testing model architectures...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test different architectures
    models_to_test = ['baseline', 'resnet50', 'densenet121', 'efficientnet_b0']
    
    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()}")
        print('='*70)
        
        model = create_model(model_name, num_classes=4, pretrained=True)
        model = model.to(device)
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Clear memory
        del model, dummy_input, output
        torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("✅ All models working correctly!")
    print("="*70)