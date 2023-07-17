from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import json

from pipeline.modules import TheModel
from pipeline.dataset import DFDCDataset
from pipeline.utils import get_ids, load_config

def count_accurate(pred, tar, thres=0.5):
    pred = pred.flatten().numpy()
    tar = tar.flatten().numpy()
    count = 0
    for i, pred_value in enumerate(pred):
        pred_label = 1. if pred_value >= thres else 0.
        if np.abs(pred_label - tar[i]) < 1e-05:
            count += 1
    return count

def train(opt=None, config=None, conf_stg=None):
    # Load model
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device available!')
    model = TheModel(config=config["model"]).to(device='cuda')

    # Initialize function
    if config["loss-function"]["name"] == "BCE":
        loss_func = torch.nn.BCELoss(reduction='sum')
    else:
        raise ValueError('Inapropriate loss function {}.'.format(config["loss-function"]["name"]))

    # Initialize optimizer
    if config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning-rate"])
    else:
        raise ValueError('Unavailable optimizer {}.'.format(config["optimizer"]["name"]))
    
    # Decision strategy
    thres = config["decision-strategy"]["threshold"]

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

    # Log dict
    log_dict = {}

    # Training loop
    for epoch in range(opt.num_epochs):
        model.train()

        # Iteration on training dataset
        for item in tqdm(training_dataloader, desc=f'Epoch {epoch + 1}/{opt.num_epochs}'):
        # for item in training_dataloader:
            # Unpack item
            x, y, _ = item
            x = x.to('cuda')
            y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

            # Forward, backward and optimize paramters
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        if opt.validation:
            with torch.no_grad():
                model.eval()
                
                val_loss = 0.0
                count_acc = 0

                for item in validation_dataloader:
                    x, y = item
                    x = x.to('cuda')
                    y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

                    pred = model(x)

                    # Accumulate loss
                    val_loss += loss_func(pred, y).item()
                    
                    # Accumulate accurate predict
                    count_acc += count_accurate(pred.to('cpu'), y.to('cpu'), thres)

                loss = val_loss / len(validation_dataset)
                acc = count_acc / len(validation_dataset)
                log_dict[epoch] = {'loss': loss, 'accuracy': acc}
                print(f'Validation loss ({config["loss-function"]["name"]}): {loss:.8f} ---- Accuracy: {acc:.2f}')

                with open(opt.log_path, 'a') as f:
                    f.write(f'Epoch {epoch + 1}: loss {loss} ---- accuracy {acc}' + '\n')

        # Save model every 10 epochs
        if epoch % 10 == 9:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'model_' + str(epoch + 1) + '.pth'))

    with open('./log_dict.json', 'w') as f:
        json.dump(log_dict, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--metadata_path', help='Path to metadata folder.')
    parser.add_argument('--data_path', help='Path to data (frames) folder.')
    parser.add_argument('--validation', type=bool, default=True, help='Test the model on validation set after every epoch.')
    parser.add_argument('--save_path', default='./state_dict/', help='Path for saving model\'s state dict.')
    parser.add_argument('--log_path', help='Path for saving logs.')

    opt = parser.parse_args()

    config = load_config(path='./config/train_config.json')

    train(opt=opt, config=config)