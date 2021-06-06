import torch
import torch.nn as nn
import torch.nn.functional as F

class mainModel(nn.Module):
    def __init__(self,vocab):
        super(mainModel, self).__init__()
        self.dim = 512

        self.srcembd = nn.Embedding(vocab, self.dim)
        self.stage1 = nn.Transformer(d_model=self.dim)
        self.stage2 = nn.Transformer()
        self.trgembd = nn.Linear(self.dim,vocab)
        self.relu = nn.ReLU()
    
    def make_len_mask(self, inp):
            return (inp == 1).transpose(0, 1)
            # Assuming <blank> stoi to be 1 for now!

    def forward(self, x, y):
        # x-->input sentence
        # y-->output sentence
        
        a = self.srcembd(x)
        b = self.srcembd(y)
        src_pad_mask = self.make_len_mask(x)
        trg_pad_mask1 = self.make_len_mask(y)
        # print(src_pad_mask.shape)
        src_mask = self.stage1.generate_square_subsequent_mask(len(a)).to(x.device)
        tgt_mask = self.stage1.generate_square_subsequent_mask(len(b)).to(x.device)
        out1 = self.stage1(a, b, src_mask=src_mask, tgt_mask=tgt_mask, \
            src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask1, memory_key_padding_mask=src_pad_mask)
        
        src_mask = self.stage2.generate_square_subsequent_mask(len(out1)).to(y.device)
        tgt_mask = self.stage1.generate_square_subsequent_mask(len(a)).to(x.device)
        # src_pad_mask = self.make_len_mask(out1)
        # trg_pad_mask1 = self.make_len_mask(a)
        out2 = self.stage2(out1, a, src_mask=src_mask, tgt_mask=tgt_mask,\
            src_key_padding_mask=trg_pad_mask1, tgt_key_padding_mask=src_pad_mask, memory_key_padding_mask=trg_pad_mask1)
        
        c = F.softmax(F.relu(self.trgembd(out1)),dim=2)
        d = F.softmax(F.relu(self.trgembd(out2)),dim=2)
        return {"stage1":c,"stage2":d}