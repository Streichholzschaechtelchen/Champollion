import numpy as np
from math import sqrt
from scipy.optimize import minimize, Bounds
from numpy.random import random_sample

WINDOW_SIZE = 5
#MIN_FREQ_SOURCE = 150
#MIN_FREQ_DEST = 50
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 20

def context_vectors(words, words_index, text, window_size=WINDOW_SIZE, min_freq=MIN_FREQ_SOURCE):

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
    F = 5
    for k in range(K):
        for k2 in range(K):
            if tf[k][k2] < F * idf[k] * idf[k2] * window_size / S:
                tf[k][k2] = 0

    #Compute TF-IDF matrix from co-occurence and frequency
    maxn = max(idf)
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    tf = np.dot(np.diag(idf), tf)
    tfidf = np.zeros([K+1,K+1])
    tfidf[:K,:K] = tf
    return words, words_index, tfidf

def min_argmin_admissible(mat, m, n, P, Q):#Pt, Q):
    
    argmini = None
    argminj = None
    min_ = float('+inf')
    for i in range(m):
        for j in range(n):
            #if ((P[i] == -1 and Pt[j] == -1) or P[i] == j) and (Q[i] == -1):
            if (P[i] == -1 or P[i] == j) and (Q[i] == -1):
                if mat[i,j] < min_:
                    min_ = mat[i,j]
                    argmini, argminj = i, j
    return argmini, argminj, min_

def initialize(E, delta, P, english_cv, french_cv, english_size, french_size):

    #Initialize E
    for i in range(english_size):
        for j in range(english_size):
            E[i,j] = english_cv[i,j] - french_cv[P[i],P[j]]

    #Compute P without -1s
    Pp = {}
    for i in range(english_size):
        if P[i] != -1:
            Pp[i] = P[i]

    print(len(Pp))
    
    #Initialize delta
    for alpha in range(english_size):
        print(alpha)
        for beta in range(french_size):
            e = 1 if P[alpha] == beta else -1
            delta[alpha,beta] = sum(french_cv[beta,Pp[j]] ** 2 + e * 2 * E[alpha,j] * french_cv[beta,Pp[j]] for j in Pp) \
                                + sum(french_cv[Pp[i],beta] ** 2 + e * 2 * E[i,alpha] * french_cv[Pp[i],beta] for i in Pp) \
                                + 2 * french_cv[P[alpha],beta] * french_cv[beta,P[alpha]]

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

def update(E, delta, P, english_cv, french_cv, neighs, co_neighs, w, wp, old_wp):

    e = 1 if wp == -1 else -1

    #Compute neighbors of wp
    V = neighs[old_wp]
    coV = co_neighs[old_wp]
    for v in V:
        #If v is matched
        if v in P:
            j = P.index(v)
            #Update E
            E[w,j] += e * french_cv[old_wp,v]
            for y in co_neighs[v]:
                if y == old_wp:
                    continue
                #Update delta
                f = 1 if P[w] == -1 else -1
                delta[w,y] -= e * f * 2 * french_cv[old_wp,v] * french_cv[y,v]
    for cov in coV:
        if cov in P:
            i = P.index(cov)
            E[i,w] += e * french_cv[cov,old_wp]
            for y in neighs[cov]:
                if y == old_wp:
                    continue
                f = 1 if P[w] == -1 else -1
                delta[w,y] -= e * f * 2 * french_cv[cov,old_wp] * french_cv[cov,y]
    delta[w,old_wp] = -delta[w,old_wp]

def translate(english_text, french_text, lexicon):
    
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
    english_words, english_words_index, english_cv = context_vectors(english_words, english_words_index, english_text)
    french_words, french_words_index, french_cv = context_vectors(french_words, french_words_index, french_text, min_freq=MIN_FREQ_DEST)
    english_words_index = _build_index(english_words)
    french_words_index = _build_index(french_words)
    english_size = len(english_words)
    french_size = len(french_words)
    
    print(english_size, french_size)

    #Create tables
    E = np.zeros([english_size, english_size])
    delta = np.zeros([english_size, french_size])
    P = [-1] * english_size
    #Pt = [-1] * french_size
    Q = [-1] * english_size

    #Compute neighbors
    neighs, co_neighs = neighbors(french_cv, french_size)

    #Initialize P with seed lexicon
    for wa, wbs in lexicon.items():
        for wb in wbs:
            #Temporary hack
            if wa in english_words_index and wb in french_words_index:
                P[english_words_index[wa]] = french_words_index[wb]
                #Pt[french_words_index[wb]] = english_words_index[wa]
                Q[english_words_index[wa]] = french_words_index[wb]
                break

    matching_size = initialize(E, delta, P, english_cv, french_cv, english_size, french_size)
    
    while True:#matching_size < english_size:
        #TODO vérifier min_argmin_linewise
        w, wp, DeltaE = min_argmin_admissible(delta, english_size, french_size, P, Q)#Pt, Q)
        print(w,wp,DeltaE)
        if DeltaE >= 0:
            break
        P[w] = wp if P[w] == -1 else -1
        #Pt[wp] = w if Pt[wp] == -1 else -1
        matching_size += -1 if P[w] == -1 else 1
        print('{0} matching {1}->{2}, d='.format('del' if P[w] == -1 else 'add',
                                                 english_words[w], french_words[wp]), DeltaE)
        print(delta[w,wp])
        update(E, delta, P, english_cv, french_cv, neighs, co_neighs, w, P[w], wp)
        print(delta[w,wp])

    #Compute and show best translations
    translations = {}
    for i, english_word in enumerate(english_words):
        print('Translation for "{0}":'.format(english_word))
        translations[english_word] = [ french_words[P[i]] ]
        #french_translations = sorted([(french_words[j], argmin[i,j])
                                     # for j in range(french_size)],
                                     # key=lambda x: x[1],
                                     # reverse=True)[:NB_TRANSLATIONS]
        #translations[english_word] = french_translations
        
        #for fw in french_translations:
            #print('- {0} (score: {1})'.format(*fw))
        if P[i] == -1:
            print(' NO TRANSLATION')
        else:
            print('- ' + french_words[P[i]])
    return translations

#english_text = 'i am a potato and i love potato and i have potato blood in my veins'
#french_text  = 'je suis une patate et j aime la patate et j ai du sang de patate dans mes veines'
#lexicon = {'i': ['je'], 'blood': ['sang']}
#translate(english_text, french_text, lexicon)
