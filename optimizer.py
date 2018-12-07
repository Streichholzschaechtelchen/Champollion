import numpy as np
from math import sqrt

from tools import *

WINDOW_SIZE = 5
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 5

def context_vectors(words, words_index, text, window_size=WINDOW_SIZE, min_freq=MIN_FREQ_SOURCE):
    K = len(words)
    S = len(text)
    tf = np.zeros([K, K])
    f = np.zeros(K)
    a, b, c = 0, window_size, 0
    while c < S:
        word = text[c]
        k = words_index[word]
        f[k] += 1
        for word2 in text[a:b]:
            k2 = words_index[word2]
            tf[k][k2] += 1
            tf[k2][k] += 1
        a, b, c = _move_window(a, b, c, S, window_size)
    maxn = max(idf)
    to_delete = []
    for i in range(K):
        if f[i] < min_freq:
            to_delete.append(i)
    tf = np.delete(tf, to_delete, 1)
    tf = np.delete(tf, to_delete)
    words = [w for i, w in enumerate(words) if i not in to_delete]
    for k in range(K):
        for k2 in range(K):
            tf[k][k2] /= np.log(maxn ** 2 / (f[k] * f[k2]))
    return words, tf


def f(v, english_size, french_size, english_cv, french_cv):
    p = np.reshape(v, [english_size, french_size])
    pt = np.transpose(p)
    return np.linalg.norm(english_cv - np.matmul(p, np.matmul(french_cv, pt))) ** 2
    

def translate(english_text, french_text, lexicon):
    english_seed = list(set(lexicon.keys()))
    french_seed = list(set([e for v in lexicon.values() for e in v]))
    english_seed_index = _build_index(english_seed)
    french_seed_index = _build_index(french_seed)
    english_text = _get_text(english_text)
    french_text = _get_text(french_text)
    other_english_words = list(w for w in set(english_text) if w not in english_seed)
    other_french_words = list(w for w in set(french_text) if w not in french_seed)
    english_words = english_seed + other_english_words
    french_words = french_seed + other_french_words
    english_words_index = _build_index(english_words)
    french_words_index = _build_index(french_words)
