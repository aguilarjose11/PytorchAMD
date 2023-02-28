#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:53:09 2023

@author: joseaguilar
"""

from collections import Counter
import re


# Parameters
# Maximum vocabulary size
n_vocab = 30_000


en_pth = 'dataset/Europarl.en-es.en'
en_sentences = open(en_pth).read().lower().split('\n')
es_pth = 'dataset/Europarl.en-es.es'
es_sentences = open(es_pth).read().lower().split('\n')


special_chars = ',?;.:/*!+-(){}[]"\'&'
en_sentences = [re.sub(f'[{re.escape(special_chars)}]', '\g<0> ', s) for s in en_sentences]
es_sentences = [re.sub(f'[{re.escape(special_chars)}]', '\g<0> ', s) for s in es_sentences]

# Create vocabulary
words_en = [w for s in en_sentences for w in s]
vocab_en = Counter(words_en).most_common(n_vocab)
vocab_en = [w[0] for w in vocab_en]

words_es = [w for s in es_sentences for w in s]
words_es = Counter(words_es).most_common(n_vocab)
words_es = [w[0] for w in words_es]