import numpy as np
from math import sqrt, log
#from scipy.optimize import minimize, Bounds
from scipy.sparse import lil_matrix
from numpy.random import random_sample
from numpy.linalg import norm

WINDOW_SIZE = 5
#MIN_FREQ_SOURCE = 150
#MIN_FREQ_DEST = 50
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 5
F = 5

#temp
NB_STOPWORDS = 30

#def stopwords(tf, f, K, words):

#    entropy = np.zeros(K)
#    for i in range(K):
#        for j in range(K):
#            p = tf[i,j] / f[i]
#            if p > 1:
#                print(tf[i,j],f[i])
#                print(words[i],words[j])
#            if p:
#                entropy[i] -= log(p) * p

#    stopwords = list(zip(*sorted([(v, words[j]) for (j, v) in enumerate(entropy)],
#                                 reverse=True)))[1][:NB_STOPWORDS]
#    print(stopwords)
#    return stopwords
        

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
    tf = np.zeros([K, K])

    #Compute co-occurence matrix
    a, b, c = 0, window_size, 0
    while c < S:
        word = text[c]    
        if word in words_index:
            k = words_index[word]
            #for word2 in text[a:b]:
            for word2 in set(text[a:b]):
                if word2 == word:
                    continue
                if word2 not in words_index:
                    continue    
                k2 = words_index[word2]
                tf[k][k2] += 1
                tf[k2][k] += 1
        a, b, c = _move_window(a, b, c, S, window_size)

    #Compute stopwords
    #stopwords_ = stopwords(tf, idf, K, words)

    #Remove stopwords
    #i = 0
    #while i < K:
    #    word = words[i]
    #    if word in stopwords_:
    #        del words[i]
    #        del idf[i]
    #        del words_index[word]
    #        tf = np.delete(tf, i, 0)
    #        tf = np.delete(tf, i, 1)
    #        K -= 1
    #    else:
    #        words_index[word] = i
    #        i += 1

    #TODO: missing translations for delete stopwords

    #Replace low co-occurence scores by zeros
    for k in range(K):
        for k2 in range(K):
            if tf[k][k2] < f * idf[k] * idf[k2] * window_size / S:
                tf[k][k2] = 0

    #temp
    #if "january" in words_index:
    #    ij = words_index["january"]
    #    for k in range(K):
    #        if tf[ij,k]:
    #            print(words[k],tf[ij,k])


    #Compute TF-IDF matrix from co-occurence and frequency
    maxn = max(idf)
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    tf = np.dot(np.diag(idf), tf)
    #tfidf = np.zeros([K+1,K+1])
    #tfidf[:K,:K] = tf
    return words, words_index, tf

def min_argmin_admissible(mat, n, R):
    
    argmini = None
    argminj = None
    min_ = float('+inf')
    for i in R:
        min1 = float('+inf')
        argminj1 = None
        min2 = float('+inf')
        for j in range(n):
            if mat[i,j] < min1:
                min2 = min1
                min1 = mat[i,j]
                argminj1 = j
            elif mat[i,j] < min2:
                min2 = mat[i,j]
        if min1 - min2 < min_:
            min_ = min1 - min2
            argmini = i
            argminj = argminj1
    return argmini, argminj, min_

#def min_argmin_admissible(mat, n, R):
    
#    argmini = None
#    argminj = None
#    min_ = float('+inf')
#    for i in R:
#        for j in range(n):
#            if mat[i,j] < min_:
#                min_ = mat[i,j]
#                argmini, argminj = i, j
#    return argmini, argminj, min_

def initialize(delta, E, P, english_cv, french_cv, english_size, french_size, lexicon_indices):

    s=len(lexicon_indices)
    print(s)
    t=0
    Attila = np.ones([english_size, french_size])
    PD = P.dot(french_cv)
    PDt = P.dot(french_cv.transpose())
    M1 = np.diag(np.diag(PD.transpose().dot(PD)))
    M2 = np.diag(np.diag(PDt.transpose().dot(PDt)))

    E[:,:] = english_cv - P.dot(np.dot(french_size, P.transpose()))
    
    delta[:,:] = 2 * PD * PDt + np.dot(Attila, M1) + np.dot(Attila, M2) \
                 - 2 * np.dot(english_cv.transpose(), PD) - 2 * np.dot(english_cv, PDt)
    
def neighbors(M, n):

    neighs = [[] for _ in range(n)]
    co_neighs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if M[i,j]:
                neighs[i].append(j)
                co_neighs[j].append(i)
    return neighs, co_neighs

def set_(delta, P, w, french_size):

    best = []
    for j in range(french_size):
        if delta[w, j] < 0:
            best.append((delta[w, j], j))
    best = sorted(best)[:NB_TRANSLATIONS]
    T = sum(map(lambda x:log(-x[0]), best))
    for b in best:
        P[w,b[1]] = log(-b[0]) / T
    
def update(delta, E, english_cv, french_cv, french_size, english_size, w, P):

    PD = P.dot(french_cv)
    PDt = P.dot(french_cv.transpose())
    PDw = np.repeat([PD[w,:]], english_size, axis=0)
    PDtw = np.repeat([PDt[w,:]], english_size, axis=0)
    Ew = np.repeat(np.vstack(E[w,:]), french_size, axis=1)
    Etw = np.repeat(np.vstack(E[:,w]), french_size, axis=1)
    
    delta[:,:] += PDtw ** 2 + PDw ** 2 - 2 * Etw * PDtw - 2 * Ew * PDw
       
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
    E = np.zeros([english_size, english_size])
    P = lil_matrix((english_size, french_size))
    R = set(range(english_size))

    #Initialize P with seed lexicon
    lexicon_indices = set()
    for wa, wbs in lexicon.items():
        if wa not in english_words_index:
            continue
        i = english_words_index[wa]
        lexicon_indices.add(i)
        indices = []
        for wb in wbs:
            if wb in french_words_index:
                indices.append(french_words_index[wb])
        L = len(indices)
        R.remove(i)
        if L:
            c = 1 / L
            for j in indices:
                P[i,j] = c
    
    initialize(delta, E, P, english_cv, french_cv, english_size, french_size, lexicon_indices)
    
    while R:
        w, wp, DeltaE = min_argmin_admissible(delta, french_size, R)
        print(w,wp,DeltaE)
        if DeltaE >= 0:
            break
        set_(delta, P, w, french_size)
        #for testing only
        tests=[]
        for j in P[w,:].nonzero()[1]:
            tests.append((P[w, j], french_words[j]))
        tests = sorted(tests, reverse=True)
        print(tests)
        #end for testing only
        R.remove(w)
        print('add matching {}->{}, d={}'.format(english_words[w], french_words[wp], DeltaE))
        print(len(R))
        update(delta, E, english_cv, french_cv, french_size, english_size, w, P)

    #Compute and show best translations
    translations = {}
    for i, english_word in enumerate(english_words):
        #print('Translation for "{0}":'.format(english_word))
        french_translations = sorted([(french_words[j], P[i,j])
                                      for j in P[i,:].nonzero()[1]],
                                     key=lambda x: x[1],
                                     reverse=True)[:NB_TRANSLATIONS]
        translations[english_word] = french_translations
        #for fw in french_translations:
        #    print('- {0} (score: {1})'.format(*fw))
    return translations

#english_text = 'i am a potato and i love potato and i have potato blood in my veins'
#french_text  = 'je suis une patate et j aime la patate et j ai du sang de patate dans mes veines'
#lexicon = {'i': ['je'], 'blood': ['sang']}
#translate(english_text, french_text, lexicon)
