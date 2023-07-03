import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
    
class EfficientNetBlock(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config

        self.efficient_net = EfficientNet.from_pretrained(model_name=f'efficientnet-{self.config["scale"]}', num_classes=self.config["num-classes"])

    def forward(self, x):
        output_ = self.efficient_net(x)
        return output_

class Spatial(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config

        self.efficient_net = EfficientNetBlock(config=self.config["EfficientNet"])

    def forward(self, x):
        '''
        x: shape = (batch_size * num_frames_per_video, 3, h, w)
        '''
        output_ = self.efficient_net(x)
        return output_

class Spatiotemporal(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.motion_diff = nn.Conv3d(
            in_channels=3,
            out_channels=self.config["motion-diff"]["features"],
            kernel_size=(self.config["input-shape"]["frames-per-group"], 3, 3),
            stride=1,
            padding=(0, 1, 1)
        )
        self.efficient_net = EfficientNetBlock(config=self.config["EfficientNet"])

    def forward(self, x):
        '''
        x: shape = (batch_size, num_frames_per_video, channels, h, w)
        '''

        # Convert to (batch_size, channels, depth, height, width)
        motion_diff_input = x.permute(0, 2, 1, 3, 4)
        motion_diff_output = self.motion_diff(motion_diff_input)

        # Convert back to (batch_size, channels, height, width)
        efficient_net_input = motion_diff_output.squeeze()
        output_ = self.efficient_net(efficient_net_input)
        return output_

class ProposedModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.input_shape = config["input-shape"]

        self.spatial = Spatial(config=self.config["spatial"])
        self.spatiotemporal = Spatiotemporal(config=self.config["spatiotemporal"])
        self.decision_maker = nn.Linear(
            in_features=self.input_shape["groups-per-video"] * self.input_shape["frames-per-group"] + self.input_shape["groups-per-video"],
            out_features=1
        )

    def forward(self, x):
        '''
        x: shape = (batch_size, num_groups, num_frames_per_group, channels, h, w) = (8, 32, 2, 3, 224, 224)
        '''

        # Spatial branch
        spatial_input = x.view(
            self.input_shape["batch-size"] * self.input_shape["groups-per-video"] * self.input_shape["frames-per-group"],
            self.input_shape["channels"],
            self.input_shape["height"],
            self.input_shape["width"]
        )
        spatial_output = self.spatial(spatial_input)
        spatial_result = spatial_output.view(
            self.input_shape["batch-size"],
            self.input_shape["groups-per-video"] * self.input_shape["frames-per-group"]
        )

        # Spatiotemporal branch
        spatiotemporal_input = x.view(
            self.input_shape["batch-size"] * self.input_shape["groups-per-video"],
            self.input_shape["frames-per-group"],
            self.input_shape["channels"],
            self.input_shape["height"],
            self.input_shape["width"]
        )
        spatiotemporal_output = self.spatiotemporal(spatiotemporal_input)
        spatiotemporal_result = spatiotemporal_output.view(
            self.input_shape["batch-size"],
            self.input_shape["groups-per-video"]
        )

        # Make decision (merge two branches)
        decision_make_input = torch.cat([spatial_result, spatiotemporal_result], dim=1).view(self.input_shape["batch-size"], -1)
        decision_maker_output = self.decision_maker(decision_make_input)
        return decision_maker_output