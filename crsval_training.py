from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import torch
from sklearn.model_selection import KFold

from pipeline.modules import TheModel
from pipeline.dataset import CustomDataset, CrossValidationDataset
from pipeline.utils import get_ids, load_config
from pipeline.log import Log, EpochLog, CrsVal_Log, FoldLog

def train(opt=None, config=None):
    # Load model
    model = TheModel(config=config['model']).to(device='cuda')

    # Initialize function
    loss_func = torch.nn.BCELoss(reduction='sum')

    # Initialize optimizer & learning rate scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['optimizer']['learning-rate'])
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['optimizer']['learning-rate'], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(10, 101, 10)], gamma=0.2, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)

    # Training dataset and k-folds split
    dataset = CrossValidationDataset(
        frames_path=opt.data_path,
        labels_path=opt.metadata_path,
        sampling=config['sampling'],
        img_size=config['input-size']
    )
    k_fold = KFold(n_splits=opt.k_folds, shuffle=True)

    # Test dataset and data loader
    test_dataset = CustomDataset(
        ids=get_ids(path=os.path.join(opt.data_path, 'test')),
        frames_path=os.path.join(opt.data_path, 'test'),
        labels_path=os.path.join(opt.metadata_path, 'test'),
        sampling=config['sampling'],
        img_size=config['input-size']
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch-size'], shuffle=False)

    # Log
    train_log = CrsVal_Log(os.path.join(opt.log_path, 'train_log.txt'), threshold=0.5)
    valid_log = CrsVal_Log(os.path.join(opt.log_path, 'valid_log.txt'), threshold=0.5)
    test_log = CrsVal_Log(os.path.join(opt.log_path, 'test_log.txt'), threshold=0.5)

    # Training loop
    for epoch in range(opt.num_epochs):
        # Put the model to train mode
        model.train()

        # # Start logging
        # train_epoch_log = EpochLog(epoch)
        # valid_epoch_log = EpochLog(epoch)
        # test_epoch_log = EpochLog(epoch)

        for fold, (train_indices, valid_indices) in enumerate(k_fold.split(dataset.get_ids())):
            # Data loaders
            train_subsampler = SubsetRandomSampler(train_indices)
            valid_subsampler = SubsetRandomSampler(valid_indices)
            train_loader = DataLoader(dataset, batch_size=config['batch-size'], shuffle=True, sampler=train_subsampler)
            valid_loader = DataLoader(dataset, batch_size=config['batch-size'], shuffle=False, sampler=valid_subsampler)

            # Fold log
            train_fold_log = FoldLog(epoch, fold)

            for item in tqdm(train_loader, desc=f'Epoch {epoch + 1}, fold {fold}'):
                x, y, _ = item
                x = x.to('cuda')
                y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

                optimizer.zero_grad()
                pred = model(x)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()

                # train_epoch_log.log(pred.clone().detach().to('cpu'), y.clone().detach().to('cpu'))
                train_fold_log.log(pred.clone().detach().to('cpu'), y.clone().deteach().to('cpu'))

            train_best = train_log.add_fold(train_fold_log.summary())

            # Validataion
            with torch.no_grad():
                model.eval()

                valid_fold_log = FoldLog(epoch, fold)

                for item in valid_loader:
                    x, y, _ = item
                    x = x.to('cuda')
                    y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

                    pred = model(x)

                    valid_fold_log.log(pred.to('cpu'), y.to('cpu'))

                    # valid_epoch_log.log(pred.to('cpu'), y.to('cpu'))

                valid_best = valid_log.add_fold(valid_fold_log.summary())


            # Run on test dataset
            with torch.no_grad():
                model.eval()

                test_fold_log = FoldLog(epoch, fold)

                for item in test_dataloader:
                    x, y, _ = item
                    x = x.to('cuda')
                    y = torch.unsqueeze(y, 1).to(torch.float32).to('cuda')

                    pred = model(x)

                    # test_epoch_log.log(pred.to('cpu'), y.to('cpu'))

                    test_fold_log.log(pred.to('cpu'), y.to('cpu'))

                test_best = test_log.add_fold(test_fold_log.summary())

            # training_best = train_log.add_epoch(train_epoch_log.summary())
            # valid_best = valid_log.add_epoch(valid_epoch_log.summary())
            # test_best = test_log.add_epoch(test_epoch_log.summary())
            
            # if epoch % 10 == 9 or test_best:
            if fold == opt.k_folds - 1 or test_best:
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'model_' + str(epoch + 1) + '.pth'))

        scheduler.step()







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