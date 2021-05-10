# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import os, sys, random, gc
import numpy as np
import pandas as pd
import torch

sys.path.append('../')

from playdict_ocr.tokenization import Tokenizer
from datasets import PartitionedTrainDataset, TrainDataset, TestDataset

tokenizer = Tokenizer()


# %%
class CFG:
    max_dec_len=25
    size=(128, 32)
    epochs, batch_size = 3, 256
    max_grad_norm=4
    embed_dim, attention_dim = 160, 192
    encoder_dim, decoder_dim = 192, 192

# %% [markdown]
# # MODEL

# %%
from models import EncoderDecoderModel
import keras4torch as k4t
from easyocr_model.modules import VGG_FeatureExtractor, ResNet_FeatureExtractor
from repvgg import RepVGG

model = EncoderDecoderModel(
    #VGG_FeatureExtractor(1, CFG.encoder_dim),
    RepVGG(
        num_blocks=[2, 4, 6],
        width_multiplier=[0.75, 0.75, 0.75],
        use_se=False, in_channels=1, output_channels=CFG.encoder_dim),
    CFG, tokenizer)


# %%
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class CollateWrapper:
    # run on cpu
    def __call__(self, batch):
        src, tgt, tgt_lens = [], [], []
        for t in batch:
            src.append(t[0])
            tgt.append(torch.from_numpy(t[1]))
            tgt_lens.append(t[2])

        src = torch.stack(src)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
        tgt_lens = torch.tensor(tgt_lens, dtype=torch.int64)
        return src, tgt, tgt_lens, torch.tensor(0)


# %%
class MyLoopConfig(k4t.configs.TrainerLoopConfig):
    # run on gpu
    def process_batch(self, batch):
        src, tgt, tgt_lens, _ = batch
        if not self.training:
            return (src,), tgt

        tgt_lens, sort_idx = tgt_lens.sort(dim=0, descending=True)
        src, tgt = src[sort_idx], tgt[sort_idx]
        return (src, tgt, tgt_lens), tgt

    def prepare_for_optimizer_step(self, model):
        torch.nn.utils.clip_grad_norm_(model.model.encoder.parameters(), CFG.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.model.decoder.parameters(), CFG.max_grad_norm)


# %%
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torch.nn.functional as F
from torch_optimizer import AdaBelief

class CombinedOpt(torch.optim.Optimizer):
    def __init__(self, model):
        super().__init__(model.parameters(), {'lr': float('-inf')})
        self.encoder_opt = AdaBelief(
            model.encoder.parameters(), lr=2e-3)
        self.decoder_opt = torch.optim.Adam(
            model.decoder.parameters(), lr=1e-3)

    def step(self):
        self.encoder_opt.step()
        self.decoder_opt.step()

opt = CombinedOpt(model)

model = k4t.Model(model)

def ce_loss(y_pred, y_true):
    y_pred = y_pred.reshape(-1, tokenizer.vocab_size)
    y_true = y_true.reshape(-1)
    nonzero_indices = torch.nonzero(y_true).view(-1)
    return F.cross_entropy(y_pred[nonzero_indices], y_true[nonzero_indices])

def acc(y_pred, y_true):
    y_pred = y_pred.argmax(-1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    y_ = [(tokenizer.indices_to_string(i) == tokenizer.indices_to_string(j))
            for i,j in zip(y_pred, y_true)]

    return torch.tensor(y_, dtype=float).mean()

model.compile(optimizer=opt, loss=ce_loss, metrics=[acc], loop_config=MyLoopConfig(), disable_val_loss=True)


# %%
file_list = [f"../preprocessed/train_data_{i}.pkl" for i in range(4)]
cnt_list = [2000000] * 3 + [1224600]

val_data = pd.read_pickle("../preprocessed/val_data.pkl")

train_set = PartitionedTrainDataset(file_list, cnt_list, tokenizer, CFG.size)
val_set = TrainDataset(val_data, tokenizer, CFG.size)

# %% [markdown]
# # Train loop

# %%
from torch.utils.data import DataLoader
from keras4torch.callbacks import LRScheduler
import pickle
from torch.optim.lr_scheduler import MultiStepLR
from keras4torch.utils.data import RestrictedRandomSampler
from keras4torch.callbacks import ModelCheckpoint

torch.backends.cudnn.benchmark = True

scheduler_1 = LRScheduler(MultiStepLR(opt.encoder_opt, [1, 2], 0.3))
scheduler_2 = LRScheduler(MultiStepLR(opt.decoder_opt, [1, 2], 0.3))

model.fit(train_set,
            validation_data=val_set,
            epochs=CFG.epochs,
            batch_size=CFG.batch_size,
            validation_batch_size=CFG.batch_size*2,
            collate_fn=CollateWrapper(),
            sampler=RestrictedRandomSampler(cnt_list),
            callbacks=[scheduler_1, scheduler_2, ModelCheckpoint('saved_model/best.pt', monitor='val_acc')]
)


# %%
model.load_weights('saved_model/best.pt')

model.model.deploy()

_ = torch.onnx.export(model.model.cpu(),
    val_set[0][0].unsqueeze_(0), "saved_model/vgg_lstm.onnx", verbose=True, opset_version=11, input_names=['x'], output_names=['y'])


# %%



