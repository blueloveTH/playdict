import pickle
from tqdm import tqdm
import pathlib
import cv2
import numpy as np
import multiprocessing.dummy as mp

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

class _MjsynthMultiprocessingConverter:
    def __init__(self, annotation_file, output_file, max_count, num_workers) -> None:
        assert output_file[-4:] == '.pkl'

        self.output_file = output_file
        self.max_count = max_count

        with open(annotation_file) as f:
            lines = f.readlines()

        np.random.seed(7)
        np.random.shuffle(lines)

        self.data, self.tgt = [], []
        self.root_path = pathlib.Path(annotation_file).parent
        self.part_id = 0

        self.lock = mp.Lock()

        # start convertion
        with mp.Pool(num_workers) as p:
            with tqdm(total=len(lines)) as pbar:
                for _ in p.imap_unordered(self.loop, lines):
                    pbar.update()
            

        # save the rest data
        if len(self.data) > 0:
            if self.part_id > 0:
                filename = self.output_file[:-4] + f'_{self.part_id}.pkl'
            else:
                filename = self.output_file
            RecognizationDataset(self.data, self.tgt).to_pickle(filename)

    def loop(self, l):
        path, _ = l.split(' ')
        label = path.split('_')[1]
        
        img_path = self.root_path.joinpath(path[2:])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        self.lock.acquire()

        self.data.append(img)
        self.tgt.append(label)

        if len(self.data) == self.max_count:
            filename = self.output_file[:-4] + f'_{self.part_id}.pkl'
            RecognizationDataset(self.data, self.tgt).to_pickle(filename)
            self.data, self.tgt = [], []
            self.part_id += 1
        self.lock.release()


def convert_mjsynth_to_dataset(annotation_file, output_file, max_count=2000000, num_workers=8):
    _MjsynthMultiprocessingConverter(annotation_file, output_file, max_count, num_workers)

__all__ = ['convert_mjsynth_to_dataset', 'RecognizationDataset']




    