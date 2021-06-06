import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="2"

from litmodel import LitModel
import pytorch_lightning as pl



model = LitModel()
trainer = pl.Trainer(gpus=1, max_epochs=20)
# trainer.tune(model)
trainer.fit(model)

print("hello world")