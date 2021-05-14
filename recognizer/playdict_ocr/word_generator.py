import numpy as np
from random import choice, choices

choice_one = lambda x: [choice(x)]

class WordGenerator:
    def __init__(self, max_word_len=20) -> None:
        self.max_word_len = max_word_len

    non_word_chars = ['~', '#', '$', '%', '&', '+', '<', '=', '>', '?', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '|']
    uppercase_word_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    lowercase_word_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    all_word_chars = uppercase_word_chars + lowercase_word_chars
    space_or_hyphen_chars = [' ', '-']
    last_only_chars = ['!', ',', '.', ':', ';', '?']
    pair_only_chars = [('"', '"'), ("'", "'"), ('*', '*'), ('(', ')'), ('[', ']'), ('{', '}')]


    @staticmethod
    def get_all_characters():
        results = WordGenerator.non_word_chars + WordGenerator.all_word_chars + WordGenerator.space_or_hyphen_chars + WordGenerator.last_only_chars
        for p1, p2 in WordGenerator.pair_only_chars:
            results.append(p1)
            results.append(p2)
        results = {k:None for k in results}
        return list(results.keys())

    @staticmethod
    def random_insert(src, target, margin):
        insert_idx = np.random.randint(margin, len(src)-margin+1)
        src[insert_idx:insert_idx] = target
        return src

    def generate_word(self):
        word_length = int(np.random.beta(3, 3) * (self.max_word_len-1) + 1)

        proba = ['uppercase_only'] * 2 + ['lowercase_only'] * 4 + ['uppercase_first'] * 2 + ['random'] * 1
        idx = np.random.randint(0, len(proba))
        if proba[idx] == 'uppercase_only':
            word = choices(self.uppercase_word_chars, k=word_length)
        if proba[idx] == 'lowercase_only':
            word = choices(self.lowercase_word_chars, k=word_length)
        if proba[idx] == 'uppercase_first':
            first_char = choice_one(self.uppercase_word_chars)
            word = first_char + choices(self.lowercase_word_chars, k=word_length-1)
        if proba[idx] == 'random':
            word = choices(self.all_word_chars, k=word_length)

        if word_length >= 5 and np.random.uniform(0, 1) < 0.3:
            space_or_hyphen = choice_one(self.space_or_hyphen_chars)
            word = self.random_insert(word, space_or_hyphen, 2)

        if np.random.uniform(0, 1) < 0.15:
            non_word_count = np.random.randint(1, 4)
            non_word_target = choices(self.non_word_chars, k=non_word_count)
            word = self.random_insert(word, non_word_target, 0)

        pair_proba, last_proba = np.random.uniform(0, 1, size=[2])
        if pair_proba < 0.15:
            p_first, p_second = choice_one(self.pair_only_chars)[0]
            sub_length = np.random.randint(0, len(word)) + 1
            min_idx = len(word) - sub_length
            min_idx = np.random.randint(0, min_idx+1)
            word[min_idx:min_idx] = [p_first]
            min_idx = min_idx + sub_length + 1
            word[min_idx:min_idx] = [p_second]

        if last_proba < 0.15:
            word += choice_one(self.last_only_chars)
        
        if len(word) > self.max_word_len:
            if np.random.uniform(0, 1) < 0.5:
                word = word[:self.max_word_len]
            else:
                word = word[-self.max_word_len:]
        elif len(word) > 8:
            final_length = len(word) - np.random.randint(1, 3)
            p1, p2 = np.random.uniform(0, 1, size=[2])
            if p1 < 0.1:
                word = word[:final_length]
            if p2 < 0.1:
                word = word[-final_length:]
        word = ''.join(word)
        return word.strip(''.join(self.space_or_hyphen_chars))