import numpy as np
from math import sqrt
from numpy.random import random_sample
from numpy.linalg import norm

WINDOW_SIZE = 5
#MIN_FREQ_SOURCE = 150
#MIN_FREQ_DEST = 50
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 5
F = 5

def context_vectors(words, words_index, text, f, window_size=WINDOW_SIZE, min_freq=MIN_FREQ_SOURCE):

    from tools import _move_window, _map_array_in_place
    
    K = len(words)
    S = len(text)

    #Count words
    idf = [0 for _ in range(K)]
    for c, word in enumerate(text):
        k = words_index[word]
        idf[k] += 1

    #Delete words with insufficient frequency
    i = 0
    while i < K:
        word = words[i]
        if idf[i] < min_freq:
            del words[i]
            del idf[i]
            del words_index[word]
            K -= 1
        else:
            words_index[word] = i
            i += 1

    #Create co-occurence matrix
    #One column is added to allow for the "-1" trick
    tf = np.zeros([K, K])

    #Compute co-occurence matrix
    a, b, c = 0, window_size, 0
    while c < S:
        word = text[c]
        if word in words_index:
            k = words_index[word]
            for word2 in text[a:b]:
                if word2 == word:
                    continue
                if word2 not in words_index:
                    continue    
                k2 = words_index[word2]
                tf[k][k2] += 1
                tf[k2][k] += 1
        a, b, c = _move_window(a, b, c, S, window_size)

    #Replace low co-occurence scores by zeros
    for k in range(K):
        for k2 in range(K):
            if tf[k][k2] < f * idf[k] * idf[k2] * window_size / S:
                tf[k][k2] = 0

    #Compute TF-IDF matrix from co-occurence and frequency
    maxn = max(idf)
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    tf = np.dot(np.diag(idf), tf)
    tfidf = np.zeros([K+1,K+1])
    tfidf[:K,:K] = tf
    return words, words_index, tfidf

def min_argmin_admissible(mat, n, R):
    
    argmini = None
    argminj = None
    min_ = float('+inf')
    for i in R:
        for j in range(n):
            if mat[i,j] < min_:
                min_ = mat[i,j]
                argmini, argminj = i, j
    return argmini, argminj, min_

def initialize(delta, P, english_cv, french_cv, english_size, french_size, english_neighs, english_co_neighs):

    #Initialize E
    #for i in range(english_size):
    #    for j in range(english_size):
    #        E[i,j] = english_cv[i,j] - french_cv[P[i],P[j]]

    #Compute P without -1s
    Pp = {}
    for i in range(english_size):
        if P[i] != -1:
            Pp[i] = P[i]

    print(len(Pp))
    
    #Initialize delta
    for beta in range(french_size):
        v = sum(french_cv[beta,Pp[j]] ** 2 + french_cv[Pp[j],beta] ** 2 for j in Pp)
        for alpha in range(english_size):
            delta[alpha,beta] += v

    for alpha in range(english_size):
        V = english_neighs[alpha]
        coV = english_co_neighs[alpha]
        for beta in range(french_size):
            for j in V:
                if j in Pp:
                    delta[alpha,beta] -= 2 * english_cv[alpha,j] * french_cv[beta,Pp[j]]
            for j in coV:
                if j in Pp:
                    delta[alpha,beta] -= 2 * english_cv[j,alpha] * french_cv[Pp[j],beta]

    return len(Pp)

def neighbors(M, n):

    neighs = [[] for _ in range(n)]
    co_neighs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i,j]:
                neighs[i].append(j)
                co_neighs[j].append(i)
    return neighs, co_neighs

def update(delta, R, english_cv, french_cv, english_neighs, english_co_neighs, french_neighs, french_co_neighs, w, wp, english_words, french_words):

    fV = french_neighs[wp]
    fcoV = french_co_neighs[wp]
    eV = english_neighs[w]
    ecoV = english_co_neighs[w]
    for b in fcoV:
        for a in ecoV:
            if a in R:
                delta[a,b] += (french_cv[b,wp] - 2 * english_cv[a,w]) * french_cv[b,wp]
    for b in fV:
        for a in eV: 
            if a in R:
                 delta[a,b] += (french_cv[wp,b] - 2 * english_cv[w,a]) * french_cv[wp,b]
    
def translate(english_text, french_text, lexicon, f):

    f = f or F
    
    from tools import _build_index, _get_text

    #Re-format input
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
    english_words, english_words_index, english_cv = context_vectors(english_words, english_words_index, english_text, f)
    french_words, french_words_index, french_cv = context_vectors(french_words, french_words_index, french_text, f, min_freq=MIN_FREQ_DEST)
    english_words_index = _build_index(english_words)
    french_words_index = _build_index(french_words)
    english_size = len(english_words)
    french_size = len(french_words)
    
    print(english_size, french_size)

    #Create tables
    #E = np.zeros([english_size, english_size])
    delta = np.zeros([english_size, french_size])
    P = [-1] * english_size
    R = set(range(english_size))

    #Compute neighbors
    french_neighs, french_co_neighs = neighbors(french_cv, french_size)
    english_neighs, english_co_neighs = neighbors(english_cv, english_size)
    
    #Initialize P with seed lexicon
    for wa, wbs in lexicon.items():
        for wb in wbs:
            #Temporary hack
            if wa in english_words_index and wb in french_words_index:
                P[english_words_index[wa]] = french_words_index[wb]
                break

    #matching_size = initialize(E, delta, P, english_cv, french_cv, english_size, french_size)
    matching_size = initialize(delta, P, english_cv, french_cv, english_size, french_size, english_neighs, english_co_neighs)

    print(delta)
    
    while matching_size < english_size:
        w, wp, DeltaE = min_argmin_admissible(delta, french_size, R)
        #for testing only
        tests=[]
        for j in range(french_size):
            if delta[w, j] < 0:
                tests.append((delta[w, j], french_words[j]))
        tests = sorted(tests)[:10]
        print([(t[0]/ tests[0][0],t[1]) for t in tests])
        #end for testing only
        print(w,wp,DeltaE)
        if DeltaE >= 0:
            break
        P[w] = wp
        R.remove(w)
        matching_size += 1
        print('add matching {}->{}, d={}'.format(english_words[w], french_words[wp], DeltaE))
        update(delta, R, english_cv, french_cv, english_neighs, english_co_neighs, french_neighs, french_co_neighs, w, wp, english_words, french_words)

    #Compute and show best translations
    translations = {}
    for i, english_word in enumerate(english_words):
        print('Translation for "{0}":'.format(english_word))
        french_translations = sorted([(french_words[j], delta[i,j])
                                      for j in range(french_size)],
                                     key=lambda x: x[1])[:NB_TRANSLATIONS]
        translations[english_word] = french_translations
        for fw in french_translations:
            print('- {0} (score: {1})'.format(*fw))
    return translations

#english_text = 'i am a potato and i love potato and i have potato blood in my veins'
#french_text  = 'je suis une patate et j aime la patate et j ai du sang de patate dans mes veines'
#lexicon = {'i': ['je'], 'blood': ['sang']}
#translate(english_text, french_text, lexicon)
