from argparse import ArgumentParser
import json
from tqdm import tqdm

import torch

from modules import ProposedModel

def random_input():
    return torch.randn(8, 32, 2, 3, 64, 64)

def label():
    return torch.Tensor(
        [
            [1.],
            [1.],
            [0.],
            [1.],
        ]
    )

def train(options=None, config=None):
    model = ProposedModel(config=config["model"])
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(options.num_epochs):
        print(f'Epoch {epoch + 1}')

        # Load data
        input_ = random_input()
        labels = label()

        # Forward pass
        output_ = model(input_)
        preds = torch.sigmoid(output_)

        # Calculate loss
        loss = loss_func(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--num_epochs', type=int, default=10000, help="Number of epochs")

    options = parser.parse_args()

    config = {}

    with open('./config.json', 'r') as f:
        config = json.load(f)

    train(options=options, config=config)