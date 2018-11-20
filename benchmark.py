import numpy as np
import heapq

WINDOW_SIZE = 8
MIN_FREQ = 5
NB_TRANSLATIONS = 5

def _move_window(a, b, c, max_, window_size):
    if b < max_:
        b += 1
    if c - a >= window_size:
        a += 1
    return a, b, c + 1

def _map_array_in_place(f, a):
    for i, b in enumerate(a):
        a[i] = f(b)

def context_vectors(words, text, seed, window_size=WINDOW_SIZE, min_freq=MIN_FREQ):
    N = len(seed)
    K = len(words)
    S = len(text)
    tf = np.zeros([N, K])
    idf = np.zeros(K)
    a, b, c = 0, window_size, 0
    while c < S:
        word = text[c]
        #try:
        #    k = words.index(word)
        #except ValueError:
        #    pass
        #else:
        #    idf[k] += 1
        try:
            n = seed.index(word)
            for word2 in text[a:b]:
                if word2 == word:
                    continue
                try:
                    k2 = words.index(word2)
                except ValueError:
                    continue
                else:
                    tf[n][k2] += 1
        except ValueError:
            try:
                k = words.index(word)
            except ValueError:
                pass
            else:
                idf[k] += 1
        finally:
            a, b, c = _move_window(a, b, c, S, window_size)
    maxn = max(idf)
    i = 0
    deletions = 0
    while i + deletions < K:
        if idf[i] < MIN_FREQ:
            idf = np.delete(idf, i)
            tf = np.delete(tf, i, 1)
            del words[i]
            deletions += 1
        else:
            i += 1
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    return tf * idf

def projection_matrix(english_seed, french_seed, lexicon):
    E = len(english_seed)
    F = len(french_seed)
    m = np.zeros([F, E])
    for english_word, french_words in lexicon.items():
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
    english_text = english_text.split(' ')
    french_text = french_text.split(' ')
    english_words = list(set(english_text))
    french_words = list(set(french_text))
    mat = projection_matrix(english_seed, french_seed, lexicon)
    english_cv = context_vectors(english_words, english_text, english_seed)
    french_cv = context_vectors(french_words, french_text, french_seed)
    print(english_cv.shape,french_cv.shape, mat.shape,len(english_words),len(french_words))
    proj_english_cv = np.dot(mat, english_cv)
    translations = {}
    for e, english_word in enumerate(english_words):
        f = find_nearest(proj_english_cv[:,e], french_cv)
        translations[english_word] = [(french_words[w[0]], w[1]) for w in f]
    print(translations)
    return translations
