import torch
from efficientnet_pytorch import EfficientNet
from short_term import ShortTerm

def main():
    frame1 = torch.rand((1920, 1080, 3))
    frame2 = torch.rand((1920, 1080, 3))

    # Spatial - (A)
    spatial_block = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1).to(device='cuda')
    output_spatial = spatial_block(frame1)

    # Short-term - (B)
    frame_diff = ShortTerm().to(device='cuda')
    diff = frame_diff([frame1, frame2])

    short_term_ln = torch.nn.Linear(in_features=6, out_features=1, device='cuda')
    short_term = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1).to(device='cuda')

    output_short_term = short_term_ln(torch.stack(frame1, diff, dim=2))
    output_short_term = short_term(output_short_term)

    # Combine (A + B)
    combine = torch.nn.Linear(in_features=2, out_features=1, device='cuda')
    input_combine = torch.stack(output_spatial, output_short_term, dim=0)
    output_ = combine(input_combine)

if __name__ == '__main__':
    main()