from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch

from pipeline.modules import TheModel
from pipeline.dataset import DFDCDataset
from pipeline.utils import get_ids, load_config

def to_cuda(x):
    return x.to('cuda')

def train(opt=None, config=None):
    # Load the model
    if opt.device == 'cuda':
        if torch.cuda.is_available():
            model = TheModel(config=config["model"]).to(device='cuda')
        else:
            raise ValueError('CUDA not available. Please try using CPU instead.')
    else:
        model = TheModel(config=config["model"]).to(device='cpu')

    # Initialize function
    if config["loss-function"]["name"] == "BCE":
        loss_func = torch.nn.BCELoss()
    else:
        raise ValueError('Inapropriate loss function {}.'.format(config["loss-function"]["name"]))

    # Initialize optimizer
    if config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning-rate"])
    else:
        raise ValueError('Unavailable optimizer {}.'.format(config["optimizer"]["name"]))

    # Training dataset and data loader
    training_dataset = DFDCDataset(
        ids=get_ids(path=os.path.join(opt.data_path, 'training')),
        frames_path=os.path.join(opt.data_path, 'training'),
        labels_path=os.path.join(opt.metadata_path, 'training'),
        sampling=config["sampling"],
        img_size=config["input-size"]
    )
    training_dataloader = DataLoader(training_dataset, batch_size=config["batch-size"], shuffle=True)


    # Validation dataset and data loader
    if opt.validation:
        validation_dataset = DFDCDataset(
            ids=get_ids(path=os.path.join(opt.data_path, 'validation')),
            frames_path=os.path.join(opt.data_path, 'validation'),
            labels_path=os.path.join(opt.metadata_path, 'validation'),
            sampling=config["sampling"],
            img_size=config["input-size"]
        )
        validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch-size"], shuffle=False)


    # Training loop
    for epoch in range(opt.num_epochs):
        # Iteration on training dataset
        for item in tqdm(training_dataloader, desc=f'Epoch {epoch + 1}/{opt.num_epochs}'):
            x, y = item
            x = x.to(opt.device)
            y = torch.unsqueeze(y, 1).to(torch.float32).to(opt.device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        if opt.validation:
            with torch.no_grad():
                val_loss = 0.0
                for item in validation_dataloader:
                    x, y = item
                    x = x.to(opt.device)
                    y = torch.unsqueeze(y, 1).to(torch.float32).to(opt.device)

                    pred = model(x)
                    val_loss += loss_func(pred, y).item()
                loss = val_loss / len(validation_dataloader)
                print(f'Validation loss ({config["loss-function"]["name"]}): {loss:.8f}')

        # Save model every 10 epochs
        if epoch % 100 == 99:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'model_' + str(epoch + 1) + '.pth'))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda).')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--metadata_path', help='Path to metadata folder.')
    parser.add_argument('--data_path', help='Path to data (frames) folder.')
    parser.add_argument('--validation', type=bool, default=True, help='Test the model on validation set after every epoch.')
    parser.add_argument('--save_path', help='Path for saving model\'s state dict.')

    opt = parser.parse_args()

    config = load_config(path='./config.json')

    train(opt=opt, config=config)