import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl

from torchtext.legacy import data, datasets
from torchtext.legacy.data import Field, BucketIterator

import numpy as np

from configs import Configs
from model import mainModel

configs = Configs()
def tokenize(x): return x.split()


class LitModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()
        
        self.learning_rate = 0.001

        # Dataset
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


        train_data, valid_data, test_data = data.TabularDataset.splits(path='./',
                                                                    train=configs.trainset, format='csv',
                                                                    validation=configs.valset, test=configs.testset,
                                                                    fields=fields)

        self.SHRD.build_vocab(train_data, min_freq=1)
        # TRG.build_vocab(train_data, min_freq=2)
        # TRG.build_vocab(train_data, min_freq=2)

        print(f"Unique tokens in vocabulary: {len(self.SHRD.vocab)}")
        # print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
        # print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")
        print(self.SHRD.vocab.itos[0])
                        
        self.BATCH_SIZE = configs.batchSize

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.BATCH_SIZE,
            # device=device,
            sort = False)
        
        self.model = mainModel(len(self.SHRD.vocab))

    def train_dataloader(self):
        return self.train_iterator
    def val_dataloader(self):
        return self.valid_iterator

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, z,):        
        features = self.resnet18(z)['x5']
        out = self.deconv(features)
        return out

    def training_step(self, batch, index):
        src = batch.src
        trg = batch.trg
        out = self.model(src,trg)

        loss1 = F.cross_entropy(out["stage1"].permute((0,2,1)), trg)
        loss2 = F.cross_entropy(out["stage2"].permute((0,2,1)), src)
         
        self.log('valid_stage1_loss', loss1, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_stage2_loss', loss2, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        
        dic = {'loss': loss1+loss2}
        
        return dic


    def validation_step(self, batch, index):
        # training_step defines the train loop. It is independent of forward.
        src = batch.src
        
        trg = batch.trg

        out = self.model(src,trg)

        # b, c, h, w = image.size()
        # print(image.size(), self.vgg11(image)['x5'].size())
       

        loss1 = F.cross_entropy(out["stage1"].permute((0,2,1)), trg)
        loss2 = F.cross_entropy(out["stage2"].permute((0,2,1)), src)
         
        self.log('valid_stage1_loss', loss1, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_stage2_loss', loss2, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        
        dic = {'loss': loss1+loss2}
        return dic
