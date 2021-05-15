# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import multiprocessing as mp
import os, sys
import time
from multiprocessing.context import Process

import cv2

from text_renderer.config import get_cfg, GeneratorCfg
from text_renderer.dataset import ImgDataset
from text_renderer.render import Render
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from playdict_ocr.datasets import RecognizationDataset


# %%
from text_renderer.corpus import Corpus
from playdict_ocr.word_generator import WordGenerator

class GameOcrCorpus(Corpus):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.wg = WordGenerator()

    def get_text(self):
        return self.wg.generate_word()


# %%
from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)


# %%
import pathlib

render_cfg = RenderCfg(
    bg_dir=pathlib.Path("bg"),
    height=32,
    perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
    corpus=GameOcrCorpus(
        RandCorpusCfg(
            font_dir=pathlib.Path("font"),
            font_size=(16, 30),
        ),
    ),
    gray=True,
    #layout_effects=Padding(p=1, w_ratio=(0.0, 0.5), h_ratio=(0.0, 0.5), center=False),
)

render = Render(render_cfg)


# %%
target_width = 144

def generate_img():
    img, label = render()

    # random flip
    if np.random.uniform(0, 1) > 0.5:
        img = 255 - img

    h, w = img.shape
    w_delta = w - target_width
    if w_delta < 0:
        img = np.pad(img, ((0,0), (-w_delta//2, -w_delta//2)), constant_values=0)

    img = cv2.resize(img, (target_width, 32))

    return img, label


# %%
def show_img(img, label=None):
    plt.imshow(img)
    plt.ylim(32, 0)
    if label is not None:
        plt.title(label)


# %%
show_img(*generate_img())


# %%
from tqdm import tqdm
import multiprocessing.dummy as mp

cnt = 2000000

def loop_body(_):
    img, label = generate_img()
    data.append(img)
    tgt.append(label)

for i in range(4):
    data, tgt = [], []

    with mp.Pool(8) as p:
        with tqdm(total=cnt) as pbar:
            for _ in p.imap_unordered(loop_body, range(cnt)):
                pbar.update()

    RecognizationDataset(data, tgt).to_pickle(f'synth_{i}.pkl')


# %%



