from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import json
from sklearn.model_selection import KFold

from pipeline.modules import TheModel
from pipeline.dataset import CustomDataset, CrossValidationDataset
from pipeline.utils import get_ids, load_config
from pipeline.log import Log, EpochLog

def train(opt=None, config=None, conf_stg=None):
    # Load model
    model = TheModel(config=config['model']).to(device='cuda')

    # Initialize function
    loss_func = torch.nn.BCELoss(reduction='sum')

    # Initialize optimizer & learning rate scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['optimizer']['learning-rate'])
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['optimizer']['learning-rate'], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(10, 101, 10)], gamma=0.2, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)

    # Training dataset and data loader
    training_dataset = CustomDataset(
        ids=get_ids(path=os.path.join(opt.data_path, 'training')),
        frames_path=os.path.join(opt.data_path, 'training'),
        labels_path=os.path.join(opt.metadata_path, 'training'),
        sampling=config['sampling'],
        img_size=config['input-size']
    )
    training_dataloader = DataLoader(training_dataset, batch_size=config['batch-size'], shuffle=True)

    dataset = CrossValidationDataset(
        frames_path=opt.data_path,
        labels_path=opt.metadata_path,
        sampling=config['sampling'],
        img_size=config['input-size']
    )
    k_fold = KFold(n_splits=opt.k_folds, shuffle=True)
    loader = DataLoader(dataset)

    # Validation dataset and data loader
    if opt.validation:
        validation_dataset = CustomDataset(
            ids=get_ids(path=os.path.join(opt.data_path, 'validation')),
            frames_path=os.path.join(opt.data_path, 'validation'),
            labels_path=os.path.join(opt.metadata_path, 'validation'),
            sampling=config['sampling'],
            img_size=config['input-size']
        )
        validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch-size'], shuffle=False)

    # Log dict
    log_dict = {}

    # Log
    train_log = Log(os.path.join(opt.log_path, 'train_log.txt'), threshold=0.5)
    valid_log = Log(os.path.join(opt.log_path, 'valid_log.txt'), threshold=0.5)

    # Training loop
    for kfold, (train_indices, valid_indices) in enumerate(k_fold.split(dataset)):
        print(len(train_indices), len(valid_indices))
    # for epoch in range(opt.num_epochs):
    #     # Put model to train mode
    #     model.train()

    #     # Start logging
    #     train_epoch_log = EpochLog(epoch)

    #     # Iteration on training dataset
    #     for item in tqdm(training_dataloader, desc=f'Epoch {epoch + 1}/{opt.num_epochs}'):
    #     # for item in training_dataloader:
    #         # Unpack item
    #         x, y, _ = item
    #         x = x.to('cuda')
    #         y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

    #         # Forward, backward and optimize paramters
    #         optimizer.zero_grad()
    #         pred = model(x)
    #         loss = loss_func(pred, y)
    #         loss.backward()
    #         optimizer.step()

    #         # Log batch output loss and accuracy
    #         train_epoch_log.log(pred.clone().detach().to('cpu'), y.clone().detach().to('cpu'))
            
    #     train_best = train_log.add_epoch(train_epoch_log.summary())

    #     # Validation
    #     if opt.validation:
    #         with torch.no_grad():
    #             # Put model to evaluation mode
    #             model.eval()

    #             # Start logging
    #             valid_epoch_log = EpochLog(epoch)
                
    #             # val_loss = 0.0
    #             # count_acc = 0

    #             for item in validation_dataloader:
    #                 # Unpack item
    #                 x, y, _ = item
    #                 x = x.to('cuda')
    #                 y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

    #                 # Feed through the frozen network
    #                 pred = model(x)

    #                 # Log
    #                 valid_epoch_log.log(pred.to('cpu'), y.to('cpu'))
                
    #             valid_best = valid_log.add_epoch(valid_epoch_log.summary())

    #                 # # Accumulate loss
    #                 # val_loss += loss_func(pred, y).item()
                    
    #                 # # Accumulate accurate predict
    #                 # count_acc += count_accurate(pred.to('cpu'), y.to('cpu'), thres)

    #             # loss = val_loss / len(validation_dataset)
    #             # acc = count_acc / len(validation_dataset)
    #             # log_dict[epoch] = {'loss': loss, 'accuracy': acc}
    #             # print(f'Validation loss ({config[\'loss-function\'][\'name\']}): {loss:.8f} ---- Accuracy: {acc:.2f}')

    #             # with open(opt.log_path, 'a') as f:
    #             #     f.write(f'Epoch {epoch + 1}: loss {loss} ---- accuracy {acc}' + '\n')

    #     scheduler.step()

    #     # Save model every 10 epochs
    #     if epoch % 10 == 9 or valid_best:
    #         torch.save(model.state_dict(), os.path.join(opt.save_path, 'model_' + str(epoch + 1) + '.pth'))

    # with open('./log_dict.json', 'w') as f:
    #     json.dump(log_dict, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--metadata_path', help='Path to metadata folder.')
    parser.add_argument('--data_path', help='Path to data (frames) folder.')
    parser.add_argument('--validation', type=bool, default=True, help='Test the model on validation set after every epoch.')
    parser.add_argument('--save_path', default='./state_dict/', help='Path for saving model\'s state dict.')
    parser.add_argument('--log_path', help='Path for saving logs.')
    parser.add_argument('--k_folds', type=int, help='k for k-folds cross validation.')

    opt = parser.parse_args()

    config = load_config(path='./config/train_config.json')

    train(opt=opt, config=config)