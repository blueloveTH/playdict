# recognizer（研究代码）

#### 文件结构

+ recognizer
    + playdict_ocr
    + mjsynth
        + 90kDICT32px
            + ...
    + preprocessed
    + train
        + ...
        + train.ipynb
    + preprocess.ipynb

#### Requirements
```txt
torch
torch_optimizer
keras4torch==1.2.2
opencv-python
tqdm
```

#### 运行说明
1. 安装所需依赖
2. 将mjsynth数据集解压到当前目录，删除中间的空层级，仅保留`90kDICT32px`
3. 运行`preprocess.ipynb`
4. 运行`train/train.ipynb`