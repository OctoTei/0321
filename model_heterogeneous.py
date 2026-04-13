# model_heterogeneous.py - FedTGP异构模型定义
import torch
import torch.nn as nn
import torchvision.models as models


class FedTGPModel(nn.Module):
    """FedTGP Model with feature extractor and classifier head"""
    
    def __init__(self, feature_extractor, classifier_head):
        super(FedTGPModel, self).__init__()
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


# ==================== 8种特征提取器 ====================

class CNN4FeatureExtractor(nn.Module):
    """4-layer CNN (McMahan et al. 2017)"""
    def __init__(self, in_channels=3, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(torch.relu(x))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(torch.relu(x))
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(torch.relu(x))
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GoogLeNetFeatureExtractor(nn.Module):
    """GoogLeNet (Szegedy et al. 2015)"""
    def __init__(self, feature_dim=512, pretrained=False):
        super().__init__()
        # 关键修复：禁用aux_logits避免训练时的多输出问题
        googlenet = models.googlenet(pretrained=pretrained, aux_logits=False)
        
        # 移除分类层，保留特征提取部分
        self.conv1 = googlenet.conv1
        self.maxpool1 = googlenet.maxpool1
        self.conv2 = googlenet.conv2
        self.conv3 = googlenet.conv3
        self.maxpool2 = googlenet.maxpool2
        self.inception3a = googlenet.inception3a
        self.inception3b = googlenet.inception3b
        self.maxpool3 = googlenet.maxpool3
        self.inception4a = googlenet.inception4a
        self.inception4b = googlenet.inception4b
        self.inception4c = googlenet.inception4c
        self.inception4d = googlenet.inception4d
        self.inception4e = googlenet.inception4e
        self.maxpool4 = googlenet.maxpool4
        self.inception5a = googlenet.inception5a
        self.inception5b = googlenet.inception5b
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # GoogLeNet最后一层是1024维
        self.fc = nn.Linear(1024, feature_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MobileNetV2FeatureExtractor(nn.Module):
    """MobileNet_v2 (Sandler et al. 2018)"""
    def __init__(self, feature_dim=512, pretrained=False):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # 移除分类层
        self.features = mobilenet.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # MobileNetV2最后一层是1280维
        self.fc = nn.Linear(1280, feature_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetFeatureExtractor(nn.Module):
    """通用ResNet特征提取器 (ResNet18/34/50/101/152)"""
    def __init__(self, resnet_type='resnet18', feature_dim=512, pretrained=False):
        super().__init__()
        
        # 加载对应的ResNet模型
        if resnet_type == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            base_dim = 512
        elif resnet_type == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            base_dim = 512
        elif resnet_type == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            base_dim = 2048
        elif resnet_type == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            base_dim = 2048
        elif resnet_type == 'resnet152':
            resnet = models.resnet152(pretrained=pretrained)
            base_dim = 2048
        else:
            raise ValueError(f"Unknown resnet_type: {resnet_type}")
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_dim, feature_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== 4种分类器 ====================

class Classifier1(nn.Module):
    """Classifier 1: 直接输出 (512 -> 10)"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class Classifier2(nn.Module):
    """Classifier 2: 512 -> 256 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier3(nn.Module):
    """Classifier 3: 512 -> 128 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier4(nn.Module):
    """Classifier 4: 512 -> 256 -> 128 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier5(nn.Module):
    """Classifier 5: 512 -> 384 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 384)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(384, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier6(nn.Module):
    """Classifier 6: 512 -> 256 -> 64 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier7(nn.Module):
    """Classifier 7: 512 -> 512 -> 256 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier8(nn.Module):
    """Classifier 8: 512 -> 384 -> 192 -> 10"""
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 384)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(384, 192)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(192, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== 模型创建函数 ====================

def create_feature_extractor(extractor_type, in_channels=3, feature_dim=512, pretrained=False):
    """创建特征提取器"""
    if extractor_type == 'cnn4':
        return CNN4FeatureExtractor(in_channels, feature_dim)
    elif extractor_type == 'googlenet':
        return GoogLeNetFeatureExtractor(feature_dim, pretrained)
    elif extractor_type == 'mobilenet_v2':
        return MobileNetV2FeatureExtractor(feature_dim, pretrained)
    elif extractor_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return ResNetFeatureExtractor(extractor_type, feature_dim, pretrained)
    else:
        raise ValueError(f"Unknown feature extractor: {extractor_type}")


def create_classifier(classifier_type, feature_dim=512, num_classes=10):
    """创建分类器"""
    if classifier_type == 'classifier1':
        return Classifier1(feature_dim, num_classes)
    elif classifier_type == 'classifier2':
        return Classifier2(feature_dim, num_classes)
    elif classifier_type == 'classifier3':
        return Classifier3(feature_dim, num_classes)
    elif classifier_type == 'classifier4':
        return Classifier4(feature_dim, num_classes)
    elif classifier_type == 'classifier5':
        return Classifier5(feature_dim, num_classes)
    elif classifier_type == 'classifier6':
        return Classifier6(feature_dim, num_classes)
    elif classifier_type == 'classifier7':
        return Classifier7(feature_dim, num_classes)
    elif classifier_type == 'classifier8':
        return Classifier8(feature_dim, num_classes)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")


def create_heterogeneous_model(config, client_id=None):
    """
    创建异构模型
    
    Args:
        config: 配置字典
        client_id: 客户端ID (1-based)，如果为None则使用默认配置
    """
    feature_dim = config['model_args']['feature_dim']
    num_classes = config['model_args']['num_classes']
    pretrained = config['model_args'].get('pretrained', False)
    
    # 自动检测输入通道数（从数据集推断）
    dataset_name = config['data_args']['dataset'].lower()
    if 'mnist' in dataset_name:
        in_channels = 1
    elif 'cifar' in dataset_name or 'imagenet' in dataset_name:
        in_channels = 3
    else:
        in_channels = 3  # 默认RGB
    
    # 优先从model_args中读取特征提取器和分类器配置
    if 'feature_extractor' in config['model_args'] and 'classifier' in config['model_args']:
        extractor_type = config['model_args']['feature_extractor']
        classifier_type = config['model_args']['classifier']
    # 如果没有，尝试从client_configs中读取（兼容旧配置）
    elif client_id is not None and 'client_configs' in config:
        client_key = f'client_{client_id}'
        if client_key in config['client_configs']:
            client_config = config['client_configs'][client_key]
            extractor_type = client_config['feature_extractor']
            classifier_type = client_config['classifier']
        else:
            # 使用默认配置
            extractor_type = config['model_args']['model']
            classifier_type = 'classifier1'
    else:
        # 使用默认配置
        extractor_type = config['model_args']['model']
        classifier_type = 'classifier1'
    
    # 创建特征提取器和分类器
    feature_extractor = create_feature_extractor(
        extractor_type, in_channels, feature_dim, pretrained
    )
    classifier_head = create_classifier(classifier_type, feature_dim, num_classes)
    
    # 创建完整模型
    model = FedTGPModel(feature_extractor, classifier_head)
    
    return model, extractor_type, classifier_type
