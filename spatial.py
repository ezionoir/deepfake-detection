import torch
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1).to(device='cuda')
summary(model, input_size=(3, 256, 256))