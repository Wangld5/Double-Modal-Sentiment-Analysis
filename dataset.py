import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Pred_Dataset(Dataset):
    def __init__(self, dataset_path, dataset_name, split_type):
        """
        split_type: type of the data, etc: train
        """
        super(Pred_Dataset, self).__init__()
        dataset_path = os.path.join(dataset_path, dataset_name)
        dataset = pickle.load(open(dataset_path, 'rb'))

        # seperate and get text, vision, audio, label part
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = torch.tensor(dataset[split_type]['audio'].astype(np.float32)).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        self.data = dataset_name
        self.n_modalities = 3

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return (self.text.shape[1], self.vision.shape[1],self.audio.shape[1])
    def get_dim(self):
        return (self.text.shape[2], self.vision.shape[2],self.audio.shape[2])
    def get_label_info(self):
        return (self.labels.shape[1], self.labels.shape[2])
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y