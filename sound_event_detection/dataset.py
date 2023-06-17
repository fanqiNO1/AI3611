import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import encode_label


def load_dict_from_csv(filename, colnames, sep='\t'):
    assert len(colnames) == 2, "colnames must be a tuple of 2 strings"
    data = pd.read_csv(filename, sep=sep)
    data = dict(zip(data[colnames[0]], data[colnames[1]]))
    return data


def load_dict_from_dataframe(data, colnames):
    assert len(colnames) == 2, "colnames must be a tuple of 2 strings"
    data = dict(zip(data[colnames[0]], data[colnames[1]]))
    return data


def pad(tensor_list, batch_first=True, padding_value=0):
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
    lens = [len(tensor) for tensor in tensor_list]
    return padded_seq, lens


def collate_fn_labelled(batches):
    audio_ids, features, targets = zip(*batches)
    features, lens = pad(features)
    return {
        "audio_ids": audio_ids,
        "features": features,
        "targets": torch.stack(targets),
    }
    

def collate_fn_to_be_labelled(batches):
    audio_ids, features = zip(*batches)
    features, lens = pad(features)
    return {
        "audio_ids": audio_ids,
        "features": features,
    }

    
def collate_fn_unlabelled(batches):
    audio_ids, features = zip(*batches)
    features = torch.stack(features)
    return {
        "audio_ids": audio_ids,
        "features": features,
    }
    
    
class LabelledDataset(Dataset):
    def __init__(self, audio_path, label, label_to_idx):
        super(LabelledDataset, self).__init__()
        self.audio_id_to_h5 = load_dict_from_csv(audio_path, ("audio_id", "hdf5_path"))
        self.cache = {}
        # self.audio_id_to_label = load_dict_from_dataframe(label, ("filename", "event_labels"))
        if isinstance(label, str):
            self.audio_id_to_label = load_dict_from_csv(label, ("filename", "event_labels"))
        elif isinstance(label, pd.DataFrame):
            self.audio_id_to_label = load_dict_from_dataframe(label, ("filename", "event_labels"))
        else:
            raise ValueError("label must be either a string or a pandas DataFrame")
        self.audio_ids = list(self.audio_id_to_label.keys())
        self.label_to_idx = label_to_idx
        with h5py.File(self.audio_id_to_h5[self.audio_ids[0]], 'r') as store:
            self.data_dim = store[self.audio_ids[0]].shape[-1]
        
    def __len__(self):
        return len(self.audio_ids)
    
    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        h5_file = self.audio_id_to_h5[audio_id]
        if h5_file not in self.cache:
            self.cache[h5_file] = h5py.File(h5_file, 'r', libver='latest')
        feature = self.cache[h5_file][audio_id][()]
        feature = torch.as_tensor(feature).float()
        label = self.audio_id_to_label[audio_id]
        target = encode_label(label, self.label_to_idx)
        target = torch.as_tensor(target).float()
        return audio_id, feature, target
    
    
class UnlabelledDataset(Dataset):
    def __init__(self, audio_path):
        super(UnlabelledDataset, self).__init__()
        self.audio_id_to_h5 = load_dict_from_csv(audio_path, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.audio_id_to_label = {}
        self.audio_ids = list(self.audio_id_to_h5.keys())
        with h5py.File(self.audio_id_to_h5[self.audio_ids[0]], 'r') as store:
            self.data_dim = store[self.audio_ids[0]].shape[-1]
        
    def __len__(self):
        return len(self.audio_ids)
    
    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        h5_file = self.audio_id_to_h5[audio_id]
        if h5_file not in self.cache:
            self.cache[h5_file] = h5py.File(h5_file, 'r', libver='latest')
        feature = self.cache[h5_file][audio_id][()]
        feature = torch.as_tensor(feature).float()
        return audio_id, feature
    
    
class ConcatDataLoader(DataLoader):
    def __init__(self, datasets, batch_size, shuffle, collate_fn):
        num_samples = sum([len(dataset) for dataset in datasets])
        self.batch_sizes = [batch_size * len(dataset) // num_samples for dataset in datasets]
        self.collate_fn = collate_fn
        self.dataloaders = [
            DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn
            ) for dataset, batch_size in zip(datasets, self.batch_sizes)
        ]
        
    def __len__(self):
        return len(self.dataloaders[0])
        
    def __iter__(self):
        for batches in zip(*self.dataloaders):
            has_target = all(["targets" in batch for batch in batches])
            data = {"audio_ids": [], "features": []}
            if has_target:
                data["targets"] = []
            for batch in batches:
                data["audio_ids"].extend(batch["audio_ids"])
                for i in range(len(batch["features"])):
                    data["features"].append(batch["features"][i])
                if has_target:
                    data["targets"].append(batch["targets"])
            # data["features"] = torch.cat(data["features"], dim=0)
            data["features"], _ = pad(data["features"])
            if has_target:
                data["targets"] = torch.cat(data["targets"], dim=0)
            yield data
    
class EMADataLoader(DataLoader):
    def __init__(self, datasets, batch_size, shuffle, collate_fns):
        num_samples = sum([len(dataset) for dataset in datasets])
        self.batch_sizes = [batch_size * len(dataset) // num_samples for dataset in datasets]
        self.dataloaders = [
            DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
            ) for dataset, batch_size, collate_fn in zip(datasets, self.batch_sizes, collate_fns)
        ]
        
    def __len__(self):
        return len(self.dataloaders[0])
    
    def __iter__(self):
        for batches in zip(*self.dataloaders):
            yield batches
            
    
if __name__ == "__main__":
    from utils import train_test_split
    
    train_label, test_label, label_to_idx = train_test_split("data/metadata/class_label_indices.txt", "data/dev/label_weak.csv")
    
    labelled_dataset = LabelledDataset("data/dev/feature_weak.csv", train_label, label_to_idx)
    print(labelled_dataset[0])
    dataloader = DataLoader(labelled_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_labelled)
    for batch in dataloader:
        print("audio_ids", len(batch["audio_ids"]))
        print("features", batch["features"].shape)
        print("targets", batch["targets"].shape)
        break