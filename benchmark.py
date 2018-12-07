import numpy as np
import heapq
import re

from tools import *

WINDOW_SIZE = 5
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 5

def context_vectors(words, words_index, text, seed, seed_index, window_size=WINDOW_SIZE, min_freq=MIN_FREQ_SOURCE):
    N = len(seed)
    K = len(words)
    S = len(text)
    tf = np.zeros([N, K])
    idf = np.zeros(N)
    f = np.zeros(K)
    a, b, c = 0, window_size, 0
    while c < S:
        word = text[c]
        try:
            n = seed_index[word]
            idf[n] += 1
            for word2 in text[a:b]:
                if word2 == word:
                    continue
                try:
                    k2 = words_index[word2]
                except KeyError:
                    continue
                else:
                    tf[n][k2] += 1
        except KeyError:
            k = words_index[word]
            f[k] += 1
        finally:
            a, b, c = _move_window(a, b, c, S, window_size)
    maxn = max(idf)
    to_delete = []
    for i in range(K):
        if f[i] < min_freq:
            to_delete.append(i)
    tf = np.delete(tf, to_delete, 1)
    words = [w for i,w in enumerate(words) if i not in to_delete]
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    return words, np.dot(np.diag(idf), tf)
   
def projection_matrix(english_seed, french_seed, lexicon):
    E = len(english_seed)
    F = len(french_seed)
    m = np.zeros([F, E])
    for english_word, french_words in lexicon.items():
        v = 1. / np.sqrt(len(french_words))
        for french_word in french_words:
            m[french_seed.index(french_word)][english_seed.index(english_word)] = 1
    return m

def find_nearest(vec, mat, nb_translations=NB_TRANSLATIONS):
    K = len(mat[0])
    heap = []
    heapq.heapify(heap)
    if np.linalg.norm(vec) == 0:
        return []
    for k in range(K):
        veck = mat[:,k]
        if np.linalg.norm(veck) == 0:
            continue
        score = np.dot(veck, vec) / (np.linalg.norm(veck) * np.linalg.norm(vec))
        heapq.heappush(heap, (-score, k))
    l = []
    for _ in range(nb_translations):
        score, k = heapq.heappop(heap)
        l.append( (k, -score) )
    return l

def translate(english_text, french_text, lexicon):
    english_seed = list(set(lexicon.keys()))
    french_seed = list(set([e for v in lexicon.values() for e in v]))
    english_seed_index = _build_index(english_seed)
    french_seed_index = _build_index(french_seed)
    english_text = _get_text(english_text)
    french_text = _get_text(french_text)
    english_words = list(set(english_text))
    french_words = list(set(french_text))
    english_words_index = _build_index(english_words)
    french_words_index = _build_index(french_words)
    mat = projection_matrix(english_seed, french_seed, lexicon)
    english_words, english_cv = context_vectors(english_words, english_words_index, english_text, english_seed, english_seed_index)
    french_words, french_cv = context_vectors(french_words, french_words_index, french_text, french_seed, french_seed_index, min_freq=MIN_FREQ_DEST)
    proj_english_cv = np.dot(mat, english_cv)
    translations = {}
    for e, english_word in enumerate(english_words):
        f = find_nearest(proj_english_cv[:,e], french_cv)
        translations[english_word] = [(french_words[w[0]], w[1]) for w in f]
    for english_word, french_words in translations.items():
        print('Translation for "{0}":'.format(english_word))
        for fw in french_words:
            print('- {0} (score: {1})'.format(*fw))
    return translations
