from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, List
import numpy as np
import re
from collections import Counter
import itertools
from tqdm import tqdm
class TextTranslationDataset(Dataset):

    START = '<START>'
    END = '<END>'
    UNKNOWN = '<UNKNOWN>'
    PADDING = '<PADDING>'
    def __init__(self,
                 lang_src_file:  str,
                 lang_tgt_file: str,
                 max_vocab: int,
                 max_seq_len: int,
                 sen_len_cutoff: int=None
                 ):

        with open(lang_src_file, 'r') as corpus:
            src_corpus = corpus.read().lower()

        with open(lang_tgt_file, 'r') as corpus:
            tgt_corpus = corpus.read().lower()

        # Tokenize
        src_lang = src_corpus.split('\n')[:sen_len_cutoff]
        tgt_lang = tgt_corpus.split('\n')[:sen_len_cutoff]
        assert len(src_lang) == len(tgt_lang), f"Number of sentences between both languages is not equal! {len(src_lang):,} vs {len(tgt_lang):,}"
        self.num_sentences = len(src_lang)

        # Split words and characters
        special_chars = ',?;.:/*!+-()[]{}"\'&'
        # Splits all words and special characters, and collects the vocabulary.
        src_lang = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in src_lang ]
        src_lang = [[w for w in s if len(w)] for s in src_lang]

        # We convert the Counter into a dictionary to extract the words into a list
        # The itertools.chain.from_iterable() joins the sub-lists. similar to sum(list.extend, l)
        self.src_vocab = list(
            dict(
                Counter(
                    list(
                        itertools.chain.from_iterable(src_lang)
                    )
                ).most_common(max_vocab - 4)).keys()
        )
        tgt_lang = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in tgt_lang]
        tgt_lang = [[w for w in s if len(w)] for s in tgt_lang]
        if not len(tgt_lang[-1]) and not len(src_lang[-1]):
            tgt_lang.pop(-1)
            src_lang.pop(-1)
            self.num_sentences -= 1
        # We convert the Counter into a dictionary to extract the words into a list
        self.tgt_vocab = list(
            dict(
                Counter(
                    list(
                        itertools.chain.from_iterable(tgt_lang)
                    )
                ).most_common(max_vocab - 4)
            ).keys()
        )

        # Cut off sentences, add padding, and add <START>, <UNKWN> and <END> tokens.
        # First, lets add the known tokens <START>, <END>, <PADDING>, and <UNKWN>
        self.src_vocab.extend([self.UNKNOWN, self.END, self.START, self.PADDING])
        self.src_vocab.reverse() # Makes padding index 0, and so on.
        self.tgt_vocab.extend([self.UNKNOWN, self.END, self.START, self.PADDING])
        self.tgt_vocab.reverse() # Makes padding index 0, and so on.

        # Prepare temporary lists for numerical tokenization
        self.src = []
        self.tgt = []
        for i, (src_sen, tgt_sen) in tqdm(enumerate(zip(src_lang, tgt_lang)), total=len(src_lang)):
            # clip sentences, leaving space for <START> and <END> tokens to be added
            src_lang[i] = src_sen[:max_seq_len - 2]
            src_lang[i].insert(0, self.START)
            src_lang[i].append(self.END)

            tgt_lang[i] = tgt_sen[:max_seq_len - 2]
            tgt_lang[i].insert(0, self.START)
            tgt_lang[i].append(self.END)

            # Pad sentences if needed
            if len(src_lang[i]) < max_seq_len:
                # We need padding after adding <START> and <END>
                padding_length = max_seq_len - len(src_lang[i])
                src_lang[i].extend([self.PADDING for _ in range(padding_length)])

            if len(tgt_lang[i]) < max_seq_len:
                # We need padding after adding <START> and <END>
                padding_length = max_seq_len - len(tgt_lang[i])
                tgt_lang[i].extend([self.PADDING for _ in range(padding_length)])

            # Replace unknown words and create numerical tokens
            self.src.append([])
            self.tgt.append([])
            for j, (src_word, tgt_word) in enumerate(zip(src_lang[i], tgt_lang[i])):
                if src_word not in self.src_vocab:
                    src_lang[i][j] = self.UNKNOWN
                if tgt_word not in self.tgt_vocab:
                    tgt_lang[i][j] = self.UNKNOWN
                # Add numerical token
                self.src[i].append(self.src_vocab.index(src_lang[i][j]))
                self.tgt[i].append(self.tgt_vocab.index(tgt_lang[i][j]))

        # Actual text
        self.src_lang = np.array(src_lang)
        self.tgt_lang = np.array(tgt_lang)
        # Nominal tokens
        self.src = np.array(self.src)
        self.tgt = np.array(self.src)

        # Create look-forward padding


    def __getitem__(self, item):
        return self.src[item], self.tgt[item]

    def __len__(self):
        return self.num_sentences

