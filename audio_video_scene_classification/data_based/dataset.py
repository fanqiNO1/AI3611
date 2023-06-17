import pickle

import torch
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    def __init__(self, data_dir, split, audio_transform=None, video_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.audio_transform = audio_transform
        self.image_transform = video_transform  
        file = f"{split}.pkl"
        with open(f"{data_dir}/{file}", "rb") as f:
            self.dataset = pickle.load(f)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.audio_transform is not None:
            audio_feature = self.audio_transform(torch.tensor(data["audio_feature"]))
        else:
            audio_feature = torch.tensor(data["audio_feature"])
        if self.image_transform is not None:
            video_feature = self.image_transform(torch.tensor(data["video_feature"]))
        else:
            video_feature = torch.tensor(data["video_feature"])
        return {
            "file": data["file"],
            "audio_feature": audio_feature,
            "video_feature": video_feature,
            "target": torch.tensor(data["target"]),
        }
        
        
if __name__ == "__main__":
    train_dataset=  SceneDataset(".", is_training=True)
    print(len(train_dataset))
    print(train_dataset[0]["file"])
    print(train_dataset[0]["audio_feature"].shape)
    print(train_dataset[0]["video_feature"].shape)
    print(train_dataset[0]["target"])
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for batch in train_loader:
        print(len(batch["file"]))
        print(batch["audio_feature"].shape)
        print(batch["video_feature"].shape)
        print(batch["target"].shape)
        break
    from torchvision.transforms import Compose, Lambda
    audio_transform = Compose([
        Lambda(lambda x: torch.cat([x, torch.zeros(512 - x.shape[0], x.shape[1])], dim=0))
    ])
    test_dataset = SceneDataset(".", is_training=False, audio_transform=audio_transform)
    print(len(test_dataset))
    print(test_dataset[0]["filename"])
    print(test_dataset[0]["audio_feature"].shape)
    print(test_dataset[0]["video_feature"].shape)
    print(test_dataset[0]["target"])
    