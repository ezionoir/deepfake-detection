import torch
import cv2
from torchsummary import summary

import utils

class ShortTerm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._3d_conv = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, input_):
        self.input_ = self._stack_frames(input_)
        output_ = self._3d_conv(self.input_)
        return output_
    
    def backward(self, input_):
        pass

    def _stack_frames(self, input_):
        input_ = torch.stack(input_, dim=0)
        input_ = input_.permute(3, 0, 2, 1)
        return input_.to(torch.float)

'''For testing'''
if __name__ == '__main__':

    img1 = cv2.imread('./input/frame/1.jpg')
    img2 = cv2.imread('./input/frame/2.jpg')

    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)

    block = ShortTerm().to('cpu')

    output_ = block([img1, img2])
    print(output_.size())