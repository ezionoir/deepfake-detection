import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MotionDiff(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.conv_3d = nn.Conv3d(in_channels=3, out_channels=out_channels, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, x):
        x = self.conv_3d(x)
        return x
    
# class ShortTerm(nn.Module):
#     def __init__(self, out_channels_conv_3d=3):
#         super().__init__()
#         self.motion_diff = MotionDiff(out_channels=out_channels_conv_3d)
#         self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', in_channels=out_channels_conv_3d, num_classes = 1)

#     def forward(self, x):
#         x = self.motion_diff(x)
#         x = self.to_2d(x)
#         x = self.efficient_net(x)
#         return x

#     def to_2d(self, x):
#         return x.permute(1, 0, 2, 3)


    
class EfficientNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)

    def forward(self, x):
        x = self.efficient_net(x)
        return x
    
class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.dense(x)
        return x
    
class Spatial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Spatiotemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_diff = MotionDiff()
        self.dense = nn.Linear(in_features=3, out_features=1)
        self.efficient_net = EfficientNetBlock()

    def forward(self, x):
        x_original = x
        x = self.motion_diff(x)        

class ProposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = Spatial()
        self.spatiotemporal = Spatiotemporal()
        self.decision_maker = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        '''
            x: shape(batch_size, channels, num_frames_per_video, h, w)
        '''
        spatial_res = self.spatial(x)
        spatiotemporal_res = self.spatiotemporal(x)
        x = torch.stack([spatial_res, spatiotemporal_res]).view(2)
        x = self.decision_maker(x)
        return x