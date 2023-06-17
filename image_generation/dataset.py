from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ImageGenerationDataset(Dataset):
    def __init__(self, is_training=True, data_path="./data"):
        super(ImageGenerationDataset, self).__init__()
        self.is_training = is_training
        self.dataset = datasets.MNIST(root=data_path, train=is_training, transform=transforms.ToTensor(), download=False)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = ImageGenerationDataset(is_training=True, data_path="./data")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
    dataset = ImageGenerationDataset(is_training=False, data_path="./data")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
