import numpy as np

class Tokenizer:
    def __init__(self):
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'
        self.i2w = ['<PAD>', '<START>', '<END>'] + list(characters)
        self.w2i = {self.i2w[i]: i for i in range(len(self.i2w))}

        #print(self.w2i)

    @property
    def vocab_size(self):
        return len(self.i2w)

    def string_to_indices(self, s, dtype='uint8'):
        indices = []
        indices.append(self.w2i['<START>'])
        indices.extend(map(self.w2i.__getitem__, s))
        indices.append(self.w2i['<END>'])
        return np.array(indices, dtype=dtype)

    def indices_to_string(self, idx):
        idx = idx[np.where(idx>1)[0]]     # remove <PAD> and <START>
        end_idx = np.where(idx==2)[0]
        if len(end_idx) > 0:
            idx = idx[:end_idx[0]]
        return ''.join(map(self.i2w.__getitem__, idx))


class TokenizerNAT:
    def __init__(self):
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'
        self.i2w = ['<PAD>', '<PAD_1>', '<PAD_2>'] + list(characters)
        self.w2i = {self.i2w[i]: i for i in range(len(self.i2w))}

    @property
    def vocab_size(self):
        return len(self.i2w)

    def string_to_indices(self, s, dtype='uint8'):
        indices = list(map(self.w2i.__getitem__, s))
        return np.array(indices, dtype=dtype)

    def indices_to_string(self, idx):
        idx = idx[idx>2]     # remove <PAD>
        return ''.join(map(self.i2w.__getitem__, idx))

    def indices_to_string_ctc(self, idx):
        """See reference: https://zhuanlan.zhihu.com/p/42719047"""
        curr_i = -1
        result = []
        for i in idx:
            if i != curr_i:
                result.append(i)
                curr_i = i

        return self.indices_to_string(np.array(result))