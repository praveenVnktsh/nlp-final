
import os
import json
import torch


class Configs():

    def __init__(self):

        self.device = torch.device('cuda')

        self.trainset = "dataset/algolisp/Train.csv"
        self.valset = "dataset/algolisp/Dev.csv"
        self.testset = "dataset/algolisp/Test.csv"
        self.MAX_LEN = 150
        self.BOS_WORD = '<sos>'
        self.EOS_WORD = '<eos>'
        self.BLANK_WORD = "<blank>"

        self.batchSize = 32
        self.valSplit = 0.9

    def dumpConfigs(self):
        dic = self.__dict__
        dic['device'] = 'cuda'
        return dic


if __name__ == "__main__":

    c = Configs()
    c.dumpConfigs()