from argparse import ArgumentParser
import json
from tqdm import tqdm

from modules import ProposedModel

def train(options=None, config=None):
    model = ProposedModel(config=config["model"])

    for epoch in range(options.num_epochs):
        print(f'Epoch {epoch + 1}')

        print('Loading data')

        print('Forwarding')

        print('Backwarding')

        print('Updating weights')
        
        print('Saving model')

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