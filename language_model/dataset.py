import torch
from torch.utils.data import DataLoader


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, data_path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(f"{data_path}/train.txt")
        self.valid = self.tokenize(f"{data_path}/valid.txt")
        self.test = self.tokenize(f"{data_path}/test.txt")

    def tokenize(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, "r", encoding="utf-8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            idss = torch.cat(idss)
        return idss


class CorpusDataLoader(DataLoader):
    def __init__(self, corpus, batch_size, bptt, dataset_type="train"):
        self.corpus = corpus
        self.batch_size = batch_size
        self.bptt = bptt
        if dataset_type == "train":
            self.data = corpus.train
        elif dataset_type == "valid":
            self.data = corpus.valid
        elif dataset_type == "test":
            self.data = corpus.test
        else:
            raise ValueError(f"dataset_type must be 'train', 'valid' or 'test'.")
        self.data = self.batchify(self.data, batch_size)

    def batchify(self, data, batch_size):
        num_batch = data.size(0) // batch_size
        data = data.narrow(0, 0, num_batch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data

    def __iter__(self):
        for i in range(0, self.data.size(0) - 1, self.bptt):
            seq_len = min(self.bptt, self.data.size(0) - 1 - i)
            data = self.data[i : i + seq_len]
            target = self.data[i + 1 : i + 1 + seq_len].view(-1)
            yield data, target

    def __len__(self):
        return (self.data.size(0) - 1) // self.bptt


if __name__ == "__main__":
    corpus = Corpus("data/gigaspeech")
    train_loader = CorpusDataLoader(corpus, 32, 35, "train")
    valid_loader = CorpusDataLoader(corpus, 32, 35, "valid")
    test_loader = CorpusDataLoader(corpus, 32, 35, "test")
    for data, target in train_loader:
        print(data.shape, target.shape)
        break
    print(len(test_loader))