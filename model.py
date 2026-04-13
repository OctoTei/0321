# model.py - FedTGP Model Definition
import torch
import torch.nn as nn


class FedTGPModel(nn.Module):
    """FedTGP Model with feature extractor and classifier head"""
    
    def __init__(self, feature_extractor, classifier_head):
        super(FedTGPModel, self).__init__()
        # Use 'base' as the primary name for feature extractor
        self.base = feature_extractor
        self.head = classifier_head
    
    def forward(self, x):
        """Forward pass - returns features for prototype computation"""
        features = self.base(x)
        return features
    
    def classify(self, x):
        """Forward pass with classification"""
        features = self.forward(x)
        logits = self.head(features)
        return logits


def create_cnn_model(num_classes=10, feature_dim=512, input_channels=1):
    """Create a CNN model for MNIST/CIFAR"""
    
    class CNNFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, 5)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 5)
            self.bn2 = nn.BatchNorm2d(64)
            
            # Use adaptive pooling to ensure consistent output size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
            # Fixed input size after adaptive pooling
            fc_input = 64 * 4 * 4  # Always 64 * 4 * 4 = 1024
            
            self.fc1 = nn.Linear(fc_input, feature_dim)
            self.fc_input = fc_input
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.pool(torch.relu(x))
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.pool(torch.relu(x))
            
            # Apply adaptive pooling to ensure consistent size
            x = self.adaptive_pool(x)
            
            x = x.view(-1, self.fc_input)
            x = torch.relu(self.fc1(x))
            return x
    
    feature_extractor = CNNFeatureExtractor()
    classifier_head = nn.Linear(feature_dim, num_classes)
    
    model = FedTGPModel(feature_extractor, classifier_head)
    return model


def create_simple_model(num_classes=10, feature_dim=256, input_channels=1):
    """Create a simplified model for resource-constrained devices"""
    
    class SimpleFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, 16, 5)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.bn2 = nn.BatchNorm2d(32)
            
            # Use adaptive pooling to ensure consistent output size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            
            # Fixed input size after adaptive pooling
            fc_input = 32 * 4 * 4  # Always 32 * 4 * 4 = 512
            
            self.fc1 = nn.Linear(fc_input, feature_dim)
            self.fc_input = fc_input
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.pool(torch.relu(x))
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.pool(torch.relu(x))
            
            # Apply adaptive pooling to ensure consistent size
            x = self.adaptive_pool(x)
            
            x = x.view(-1, self.fc_input)
            x = torch.relu(self.fc1(x))
            return x
    
    feature_extractor = SimpleFeatureExtractor()
    classifier_head = nn.Linear(feature_dim, num_classes)
    
    model = FedTGPModel(feature_extractor, classifier_head)
    return model


def create_model(model_args):
    """创建FedTGP模型 - 标准接口"""
    # 从配置中获取参数
    num_classes = model_args.get('num_classes', 10)
    feature_dim = model_args.get('feature_dim', 512)
    input_channels = model_args.get('in_channels', 1)
    
    # 根据模型类型选择创建函数
    model_type = model_args.get('model', 'custom_cnn')
    
    if model_type == 'simple_cnn':
        return create_simple_model(num_classes, feature_dim, input_channels)
    else:
        return create_cnn_model(num_classes, feature_dim, input_channels)
