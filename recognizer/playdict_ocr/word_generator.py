import numpy as np
from random import choice, choices, random

choice_one = lambda x: [choice(x)]

class WordGenerator:
    def __init__(self, max_word_len=20) -> None:
        self.max_word_len = max_word_len

    non_word_chars = ['~', '$', '%', '&', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*']
    uppercase_word_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    lowercase_word_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    all_word_chars = uppercase_word_chars + lowercase_word_chars + non_word_chars
    space_or_hyphen_chars = [' ', '-']
    edge_only_chars = ['(', '[', ')', ']', '"', '!', ',', '.', ':', ';', '?']


    @staticmethod
    def get_all_characters():
        results = WordGenerator.all_word_chars + WordGenerator.space_or_hyphen_chars + WordGenerator.edge_only_chars
        results = {k:None for k in results}
        return list(results.keys())

    @staticmethod
    def random_insert(src, target, margin):
        insert_idx = np.random.randint(margin, len(src)-margin+1)
        src[insert_idx:insert_idx] = target
        return src

    def generate_word(self):
        word_length = int(np.random.beta(3, 4.5) * (self.max_word_len-1) + 1)

        proba = ['uppercase_only'] + ['lowercase_only'] + ['uppercase_first']
        idx = np.random.randint(0, len(proba))
        if proba[idx] == 'uppercase_only':
            word = choices(self.uppercase_word_chars, k=word_length)
        if proba[idx] == 'lowercase_only':
            word = choices(self.lowercase_word_chars, k=word_length)
        if proba[idx] == 'uppercase_first':
            first_char = choice_one(self.uppercase_word_chars)
            word = first_char + choices(self.lowercase_word_chars, k=word_length-1)

        if word_length >= 5 and np.random.uniform(0, 1) < 0.2:
            space_or_hyphen = choice_one(self.space_or_hyphen_chars)
            word = self.random_insert(word, space_or_hyphen, 2)

        actv_proba, side_proba = np.random.uniform(0, 1, size=[2])
        if actv_proba < 0.08:
            cnt = np.random.randint(1, 5)
            if side_proba < 0.5:
                word = word + choices(self.non_word_chars, k=cnt)
            else:
                word = choices(self.non_word_chars, k=cnt) + word

        first_proba, last_proba = np.random.uniform(0, 1, size=[2])

        if last_proba < 0.06:
            word = word + choice_one(self.edge_only_chars)
        if first_proba < 0.10:
            word = choice_one(self.edge_only_chars) + word
        
        if len(word) > self.max_word_len:
            if np.random.uniform(0, 1) < 0.5:
                word = word[:self.max_word_len]
            else:
                word = word[-self.max_word_len:]
        return ''.join(word).strip(''.join(self.space_or_hyphen_chars))