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

    def string_to_indices(self, s):
        indices = []
        indices.append(self.w2i['<START>'])
        indices.extend([self.w2i[c] for c in s])
        indices.append(self.w2i['<END>'])
        return np.array(indices, dtype='uint8')

    def indices_to_string(self, idx):
        idx = idx[np.where(idx>1)[0]]     # remove <PAD> and <START>
        end_idx = np.where(idx==2)[0]
        if len(end_idx) > 0:
            idx = idx[:end_idx[0]]
        return ''.join([self.i2w[i] for i in idx])