from torch.utils.data import Dataset
import os
import json
import cv2
import numpy as np

class DFDCDataset(Dataset):
    def __init__(self, ids: list, frames_path, labels_path, transform=None, sampling=None):
        self.ids = ids
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.transfrom = transform
        self.sampling = sampling

        self.labels = self.get_labels(labels_path=self.labels_path)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]
        folder_path = os.path.join(self.frames_path, id)

        frame_names = self.get_frame_names(folder_path=folder_path, sampling=self.sampling)

        item = []
        for group in frame_names:
            volume = []
            for frame_name in group:
                image = cv2.imread(os.path.join(folder_path, frame_name))
                image = cv2.resize(image, (64, 64))
                image = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, axes=(2, 0, 1))
                volume.append(image)
            item.append(np.stack(volume))

        return np.stack(item), self.labels[id]

    def get_labels(self, labels_path):
        labels = {}
    
        for json_file in os.listdir(labels_path):
            with open(os.path.join(labels_path, json_file), 'r') as f:
                file_data = json.load(f)
                for name, details in file_data.items():
                    id = name.split('.')[0]
                    labels[id] = details["is_fake"]

        return labels
    
    def get_frame_names(self, folder_path, sampling=None):
        num_groups = sampling['num_groups']
        num_frames_per_group = sampling['group_size']

        faces = {}
        for file_name in os.listdir(folder_path):
            face = int(file_name.split('.')[0].split('_')[-1])

            if face not in faces:
                faces[face] = [file_name]
            else:
                faces[face].append(file_name)

        total_frames = len(faces[0])
        interval = (total_frames - num_frames_per_group) // num_groups

        frame_names = []
        for i in range(num_groups):
            group = []
            for j in range(num_frames_per_group):
                idx = i * interval + j
                group.append(str(str(idx) + '_0.png'))
            frame_names.append(group)

        return frame_names
    
class TestDataset(Dataset):
    def __init__(self, ids: list, frames_path, labels_path, transform=None, sampling=None):
        self.ids = ids
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.transfrom = transform
        self.sampling = sampling

        self.labels = self.get_labels(labels_path=self.labels_path)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]
        folder_path = os.path.join(self.frames_path, id)

        frame_names = self.get_frame_names(folder_path=folder_path, sampling=self.sampling)

        item = []
        for group in frame_names:
            volume = []
            for frame_name in group:
                image = cv2.imread(os.path.join(folder_path, frame_name))
                image = cv2.resize(image, (64, 64))
                image = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, axes=(2, 0, 1))
                volume.append(image)
            item.append(np.stack(volume))

        return np.stack(item), self.labels[id]

    def get_labels(self, labels_path):
        labels = {}
    
        for json_file in os.listdir(labels_path):
            with open(os.path.join(labels_path, json_file), 'r') as f:
                file_data = json.load(f)
                for name, details in file_data.items():
                    id = name.split('.')[0]
                    labels[id] = details["is_fake"]

        return labels
    
    def get_frame_names(self, folder_path, sampling=None):
        num_groups = sampling['num_groups']
        num_frames_per_group = sampling['num_frames_per_group']

        faces = {}
        for file_name in os.listdir(folder_path):
            face = int(file_name.split('.')[0].split('_')[-1])

            if face not in faces:
                faces[face] = [file_name]
            else:
                faces[face].append(file_name)

        total_frames = len(faces[0])
        interval = (total_frames - num_frames_per_group) // num_groups

        frame_names = []
        for i in range(num_groups):
            group = []
            for j in range(num_frames_per_group):
                idx = i * interval + j
                group.append(str(str(idx) + '_0.png'))
            frame_names.append(group)

        return frame_names