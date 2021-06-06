import torch
import torch.nn as nn
import torch.nn.functional as F

class mainModel(nn.Module):
    def __init__(self,vocab):
        super(mainModel, self).__init__()
        self.dim = 512

        self.srcembd = nn.Embedding(vocab, self.dim)
        self.stage1 = torch.nn.Transformer(d_model=self.dim)
        self.stage2 = torch.nn.Transformer()
        self.trgembd = nn.Linear(self.dim,vocab)
        self.relu = nn.ReLU()

    def forward(self, x,y):
        a = self.srcembd(x)
        b = self.srcembd(y)
        out1 = self.stage1(a,b)
        out2 = self.stage2(out1,b)
        c = F.softmax(F.relu(self.trgembd(out1)),dim=2)
        d = F.softmax(F.relu(self.trgembd(out2)),dim=2)
        return {"stage1":c,"stage2":d}