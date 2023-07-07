from argparse import ArgumentParser
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from modules import ProposedModel
from dataset import DFDCDataset
from utils import get_ids, load_config

def train(opt=None, config=None, evaluation=True):
    model = ProposedModel(config=config["model"])

    if config["loss-function"]["name"] == "BCE":
        loss_func = torch.nn.BCELoss()

    if config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning-rate"])

    training_dataset = DFDCDataset(
        ids=get_ids(), # Chua sua
        frames_path='/home/ezionoir/Downloads/crops',   # Chua sua
        labels_path='/home/ezionoir/Downloads/test_set/metadata',   # Chua sua
        sampling=config["sampling"]
    )
    training_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True)

    validation_dataset = DFDCDataset(
        ids=get_ids(), # Chua sua
        frames_path='/home/ezionoir/Downloads/crops', # Chua sua
        labels_path='/home/ezionoir/Downloads/test_set/metadata', # Chua sua
        sampling=config["sampling"]
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    for epoch in range(opt.num_epochs):
        print(f'Epoch {epoch + 1}')

        # Iteration on training dataset
        for item in tqdm(training_dataloader, desc=f'Epoch {epoch + 1}/{opt.num_epochs}'):
            x, y = item

            optimizer.zero_grad()

            preds = model(x)

            loss = loss_func(y, preds)

            loss.backward()

            optimizer.step()

            # Log training infos

        # Validation
        if evaluation:
            with torch.no_grad:
                val_loss = 0.0
                for item in validation_dataloader:
                    x, y = item
                    preds = model(x)
                    val_loss += loss_func(y, preds).item()
                loss = val_loss / len(validation_dataloader)
                print(f'Validation loss: {loss:.8f}')

        # Save model every 10 epochs
        if epoch % 10 == 9:
            # save the model
            pass

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--num_epochs', type=int, default=10000, help="Number of epochs")
    parser.add_argument('--metadata_path', help='Path to metadata folder')

    opt = parser.parse_args()

    config = load_config(json_path='./config.json')

    train(opt=opt, config=config, evaluation=True)