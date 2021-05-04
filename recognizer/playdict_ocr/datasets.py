import pickle
from tqdm import tqdm
import pathlib
import cv2
from .tokenization import Tokenizer
import numpy as np

class RecognizationDataset:
    def __init__(self, data, tgt=None, **kwargs) -> None:
        self.data = data
        self.tgt = tgt
        if tgt is not None:
            assert len(data) == len(tgt)
        self.kwargs = kwargs

    def __getitem__(self, i):
        if self.tgt is None:
            return self.data[i]
        else:
            return self.data[i], self.tgt[i]

    def __len__(self):
        return len(self.data)

    def to_pickle(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f)


def convert_mjsynth_to_dataset(annotation_file, output_file, max_count=2000000, shuffle=None):
    assert output_file[-4:] == '.pkl'

    with open(annotation_file) as f:
        lines = f.readlines()

    if shuffle is None:
        shuffle = len(lines) > max_count
    if shuffle:
        np.random.seed(7)
        np.random.shuffle(lines)

    tokenizer = Tokenizer()

    root_path = pathlib.Path(annotation_file).parent

    data, tgt = [], []

    part_id = 0
    for l in tqdm(lines):
        path, _ = l.split(' ')
        label = path.split('_')[1]
        
        img_path = root_path.joinpath(path[2:])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        data.append(img)
        tgt.append(tokenizer.string_to_indices(label))

        if len(data) == max_count:
            filename = output_file[:-4] + f'_{part_id}.pkl'
            RecognizationDataset(data, tgt).to_pickle(filename)
            data, tgt = [], []
            part_id += 1

    if len(data) > 0:
        if part_id > 0:
            filename = output_file[:-4] + f'_{part_id}.pkl'
        else:
            filename = output_file
        RecognizationDataset(data, tgt).to_pickle(filename)

    