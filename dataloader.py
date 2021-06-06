
from torch import functional
from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchtext.legacy import data, datasets

import numpy as np

import pytorch_lightning as pl

from configs import Configs

configs = Configs()

def tokenize(x): return x.split()

class TrainDataset(Dataset):

    def __init__(self, dataset):
        
        self.MAX_LEN = configs.MAX_LEN
        self.BOS_WORD = configs.BOS_WORD
        self.EOS_WORD = configs.EOS_WORD
        self.BLANK_WORD = configs.BLANK_WORD

        self.SHRD = data.Field(tokenize=tokenize, pad_token=self.BLANK_WORD,init_token=self.BOS_WORD,
                        eos_token=self.EOS_WORD,fix_length=self.MAX_LEN, lower=True)

        # SRC = data.Field(tokenize=tokenize, pad_token=BLANK_WORD,init_token=BOS_WORD,
        #                 eos_token=EOS_WORD,fix_length=MAX_LEN, lower=True)
        # TRG = data.Field(tokenize=tokenize, init_token=BOS_WORD,
                        # eos_token=EOS_WORD, pad_token=BLANK_WORD, fix_length=MAX_LEN, lower=True)

        fields = {'Input': ('src', self.SHRD), 'Output': ('trg', self.SHRD), }


        self.train_data, self.valid_data, self.test_data = data.TabularDataset.splits(path='./',
                                                                    train=configs.trainset, format='csv',
                                                                    validation=configs.valset, test=configs.testset,
                                                                    fields=fields)


        self.SHRD.build_vocab(self.train_data, min_freq=2)
        # TRG.build_vocab(train_data, min_freq=2)
        # TRG.build_vocab(train_data, min_freq=2)

        print(f"Unique tokens in vocabulary: {len(self.SHRD.vocab)}")
        # print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
        # print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")


    def __getitem__(self, index):

        images = []
        segs = []
        flip = False
        if index >= self.length / 2:
            flip = True
            index = index % (self.length // 2)

        for i in range(index, index + 6):
            image = self.transform(self.data[i]['front'])
            seg = self.segmentation(self.data[i]['road']).bool().float()

            if flip:
                image = torchvision.transforms.functional.hflip(image)
                seg = torchvision.transforms.functional.hflip(seg)
            images.append(image)
            segs.append(seg)

        seg = torch.stack(tuple(segs), dim=0)
        image = torch.stack(tuple(images), dim=0)
        real = self.nonorm(self.data[i]['front'])

        return {'input': image, 'target': seg, 'real': real}

    def __len__(self):
        return self.length


class lit_custom_data(pl.LightningDataModule):

    def setup(self, stage=None):

        self.configs = Configs()

        if stage is not None:
            self.configs.trainset = stage + self.configs.trainset
            self.configs.valset = stage + self.configs.valset
            self.configs.testset = stage + self.configs.testset

        self.cpu = 0
        self.pin = True
        print('Loading dataset')

    def train_dataloader(self):
       return self.train_data

    def val_dataloader(self):
        dataset = TrainDataset(torch.load(self.configs.valset))
        return DataLoader(dataset, batch_size=self.configs.batchSize,
                          num_workers=self.cpu, sampler=SubsetRandomSampler(self.valIndices), pin_memory=self.pin)

    def test_dataloader(self):
        dataset = TestDataset(torch.load(self.configs.testset))
        return DataLoader(dataset, batch_size=1,
                          num_workers=self.cpu, pin_memory=self.pin)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = "datasetsmall.pt"
    # cd = CustomDataset(configs)
    # print(cd[0]['input'].size())