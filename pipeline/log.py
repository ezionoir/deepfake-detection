from torch import is_tensor
import torch
import numpy as np

class Log:
    def __init__(self, save_path=None, threshold=0.5):
        self.epochs = {}
        self.best = {
            'epoch': 0,
            'loss': 1.0e2
        }
        self.save_path = save_path
        self.threshold = threshold

    def add_epoch(self, epoch_log) -> bool:
        is_best = False

        epoch = epoch_log['epoch']
        loss = epoch_log['loss']
        accuracy = epoch_log['accuracy']

        if self.best['loss'] > loss:
            self.best['epoch'] = epoch
            self.best['loss'] = loss
            is_best = True

        self.epochs[epoch] = loss

        print(f'Epoch {epoch + 1}: loss = {loss:.8f}, accuracy = {accuracy:.2f}')

        if self.save_path is not None:
            self.write(loss, accuracy)

        return is_best
        
    def write(self, loss, accuracy):
        with open(self.save_path, 'a') as f:
            f.write(f'Epoch {len(self.epochs)}: loss = {loss}, accuracy = {accuracy}\n')


class EpochLog:
    def __init__(self, epoch, threshold=0.5):
        self.epoch = epoch
        self.threshold = threshold
        self.num_acc = 0
        self.total_loss = 0
        self.count = 0

    def log(self, pred, tar):
        with torch.no_grad():
            bce_loss = torch.nn.functional.binary_cross_entropy(pred, tar, reduction='sum')
            self.total_loss += bce_loss.item()

        if torch.is_tensor(pred):
            pred = pred.flatten().numpy()
        if torch.is_tensor(tar):
            tar = tar.flatten().numpy()

        self.count += len(pred)

        for i, p in enumerate(pred):
            pred_label = 1. if p >= self.threshold else 0.
            if np.abs(pred_label - tar[i]) < 1e-05:
                self.num_acc += 1

    def summary(self):
        return {
            'epoch': self.epoch,
            'loss': self.total_loss / self.count,
            'accuracy': self.num_acc / self.count
        }


class FoldLog:
    def __init__(self, epoch, fold, threshold=0.5):
        self.epoch = epoch
        self.fold = fold
        self.threshold = threshold
        self.num_acc = 0
        self.total_loss = 0
        self.count = 0

    def log(self, pred, tar):
        with torch.no_grad():
            bce_loss = torch.nn.functional.binary_cross_entropy(pred, tar, reduction='sum')
            self.total_loss += bce_loss.item()

        if torch.is_tensor(pred):
            pred = pred.flatten().numpy()
        if torch.is_tensor(tar):
            tar = tar.flatten().numpy()

        self.count += len(pred)

        for i, p in enumerate(pred):
            pred_label = 1. if p >= self.threshold else 0.
            if np.abs(pred_label - tar[i]) < 1e-05:
                self.num_acc += 1

    def summary(self):
        return {
            'epoch': self.epoch,
            'fold': self.fold,
            'loss': self.total_loss / self.count,
            'accuracy': self.num_acc / self.count
        }


class CrsVal_Log:
    def __init__(self, save_path=None, threshold=0.5):
        self.folds = {}
        self.best = {
            'epoch': 0,
            'fold': 0,
            'loss': 1.0e2
        }
        self.save_path = save_path
        self.threshold = threshold

    def add_fold(self, fold_log) -> bool:
        is_best = False

        epoch = fold_log['epoch']
        fold = fold_log['fold']
        loss = fold_log['loss']
        accuracy = fold_log['accuracy']

        if self.best['loss'] > loss:
            self.best['epoch'] = epoch
            self.best['fold'] = fold
            self.best['loss'] = loss
            is_best = True

        self.folds[(epoch, fold)] = loss

        print(f'Epoch {epoch + 1} - fold {fold + 1}: loss = {loss:.8f}, accuracy = {accuracy:.4f}')

        if self.save_path is not None:
            self.write(epoch, fold, loss, accuracy)

        return is_best
        
    def write(self, epoch, fold, loss, accuracy):
        with open(self.save_path, 'a') as f:
            f.write(f'Epoch {epoch + 1} - fold {fold + 1}: loss = {loss}, accuracy = {accuracy}\n')