import torch
from torchsummary import summary

from modules import MotionDiff, ShortTerm

def main():
    frame1 = torch.rand((320, 320, 3))
    frame2 = torch.rand((320, 320, 3))
    input_ = torch.stack([frame1, frame2])
    input_ = input_.permute(3, 0, 1, 2)

    short_term = ShortTerm(out_channels_conv_3d=12)

    output_ = short_term(input_)

    print(output_.size())

if __name__ == '__main__':
    main()