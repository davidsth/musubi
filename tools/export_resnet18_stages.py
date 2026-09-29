import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

os.makedirs("models", exist_ok=True)

base = resnet18(weights=ResNet18_Weights.DEFAULT)
base.eval()

class Stage1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return self.layer2(x)

class Stage2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

stage1 = Stage1(base).eval()
stage2 = Stage2(base).eval()

image_input = torch.randn(1, 3, 224, 224)
feature_input = stage1(image_input)

torch.onnx.export(
    stage1,
    image_input,
    "models/resnet18_stage1.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=17,
)

torch.onnx.export(
    stage2,
    feature_input,
    "models/resnet18_stage2.onnx",
    input_names=["features"],
    output_names=["logits"],
    opset_version=17,
)

print("Created models/resnet18_stage1.onnx")
print("Created models/resnet18_stage2.onnx")
print("Intermediate feature shape:", tuple(feature_input.shape))
