import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SceneEmbeddingDataset(Dataset):
    def __init__(self, audio_path, video_path, audio_transform=None, video_transform=None):
        super(SceneEmbeddingDataset, self).__init__()
        self.audio_path = audio_path
        self.video_path = video_path
        self.audio_transform = audio_transform
        self.video_transform = video_transform

        self.audio_hf = None
        self.video_hf = None

        self.files = []
        traverse = lambda name, x: self.files.append(name) if isinstance(x, h5py.Dataset) else None
        audio_hf = h5py.File(audio_path, 'r')
        audio_hf.visititems(traverse)
        audio_hf.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.audio_hf is None:
            self.audio_hf = h5py.File(self.audio_path, 'r')
        if self.video_hf is None:
            self.video_hf = h5py.File(self.video_path, 'r')

        audio_file = self.files[idx]
        audio_feature = self.audio_hf[audio_file][:96, :]
        if self.audio_transform is not None:
            audio_feature = self.audio_transform(audio_feature)

        video_file = audio_file.replace('audio', 'video')
        video_feature = self.video_hf[video_file][:96, :]
        if self.video_transform is not None:
            video_feature = self.video_transform(video_feature)

        target = int(audio_file.split('/')[0])
        
        audio_feature = torch.as_tensor(audio_feature).float()
        video_feature = torch.as_tensor(video_feature).float()
        target = torch.as_tensor(target).long()
        return {
            "file": audio_file.split('/')[-1],
            "audio_feature": audio_feature,
            "video_feature": video_feature,
            "target": target
        }


if __name__ == "__main__":
    audio_path = "../data/feature/audio_features_data/test.hdf5"
    video_path = "../data/feature/video_features_data/test.hdf5"
    dataset = SceneEmbeddingDataset(audio_path, video_path)
    print(len(dataset))
    file, audio_feature, video_feature, target = dataset[0].values()
    print(file, audio_feature.shape, video_feature.shape, target)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for batch in dataloader:
        print(type(batch["file"]), batch["audio_feature"].shape, batch["video_feature"].shape, batch["target"].shape)
        break
