import numpy as np
from math import sqrt
from scipy.optimize import minimize, Bounds
from numpy.random import random_sample

from tools import _move_window, _build_index, _get_text

WINDOW_SIZE = 5
#MIN_FREQ_SOURCE = 75
#MIN_FREQ_DEST = 25
NB_TRANSLATIONS = 5
MIN_FREQ_SOURCE = 1
MIN_FREQ_DEST = 1

#Ce n'est pas une très bonne solution
EPSILON = 0.01
PENALTY = 2

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
            if word2 == word:
                continue
            k2 = words_index[word2]
            tf[k][k2] += 1
            tf[k2][k] += 1
        a, b, c = _move_window(a, b, c, S, window_size)
    maxn = max(f)
    to_delete = []
    for i in range(K):
        if f[i] < min_freq:
            to_delete.append(i)
    tf = np.delete(tf, to_delete, 1)
    tf = np.delete(tf, to_delete, 0)
    words = [w for i, w in enumerate(words) if i not in to_delete]
    for k in range(K):
        for k2 in range(K):
            #Comment résoudre ce problème ?
            #Epsilon n'est pas une très bonne solution
            tf[k][k2] /= np.log(EPSILON + maxn ** 2 / (f[k] * f[k2]))
    penalty_c = tf.max() * PENALTY
    for k in range(K):
        for k2 in range(K):
            tf[k][k2] = tf[k][k2] or penalty_c
    return words, tf

def _p(v, english_size, french_size):
    return np.reshape(v, [english_size, french_size])

def _q(m, english_size, french_size):
    return np.reshape(m, [english_size * french_size])

def vec_to_mat(func, english_cv, french_cv, output_vector=True):
    english_size = english_cv.shape[0]
    french_size = french_cv.shape[0]
    def wrapper(v):
        p = _p(v, english_size, french_size)
        pt = np.transpose(p)
        if output_vector:
            return _q(func(p, pt, english_cv, french_cv),
                      english_size, french_size)
        else:
            return func(p, pt, english_cv, french_cv)
    return wrapper

def f(p, pt, english_cv, french_cv):
    return np.linalg.norm(english_cv - np.matmul(p, np.matmul(french_cv, pt))) ** 2

def g(p, pt, english_cv, french_cv):
    return 4 * np.matmul(np.matmul(p, np.matmul(french_cv, pt)) - english_cv, np.matmul(p, french_cv))

def Ev(i, j, english_size, french_size):
    m = np.zeros(english_size * french_size)
    m[i * french_size + j] = 1
    return m

def Rv(i, english_size, french_size):
    m = np.zeros(english_size * french_size)
    for k in range(french_size):
        m[i * french_size + k] = 1
    return m

def translate(english_text, french_text, lexicon):
    #Pourquoi est-ce nécessaire ?
    from tools import _build_index, _get_text
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
    english_words, english_cv = context_vectors(english_words, english_words_index, english_text)
    french_words, french_cv = context_vectors(french_words, french_words_index, french_text)
    english_words_index = _build_index(english_words)
    french_words_index = _build_index(french_words)
    english_size = len(english_words)
    french_size = len(french_words)
    x0 = np.zeros([english_size * french_size])
    constraints = []
    for english_word in english_words:
        if english_word in english_seed:
            french_translations = [w for w in lexicon[english_word]
                                  if w in french_words]
            english_index = english_words_index[english_word]
            if french_translations:
                c = 1 / len(french_translations)
                for french_translation in french_translations:
                   french_index = french_words_index[french_translation]
                   fun = lambda x: x[english_index * french_size \
                                     + french_index] - c
                   jac = lambda x: Ev(english_index, french_index,
                                      english_size, french_size)
                   constraints.append({'type': 'eq',
                                       'fun': fun,
                                       'jac': jac})
                   x0[english_index * french_size + french_index] = c
    for i in range(english_size):
        fun = lambda x: sum([x[i * french_size + j]
                             for j in range(french_size)]) - 1
        jac = lambda x: Rv(i, english_size, french_size)
        constraints.append({'type': 'eq',
                            'fun': fun,
                            'jac': jac})
        if english_words[i] not in english_seed:
            x0[i * french_size:(i+1) * french_size] = random_sample(french_size)
            s = sum(x0[i * french_size:(i+1) * french_size])
            x0[i * french_size:(i+1) * french_size] /= s
    print(np.reshape(x0,[english_size, french_size]))
    #Comment traiter les coefficients "infinis" (qui pour l'instant valent 0...)
    argmin = minimize(vec_to_mat(f, english_cv, french_cv, False),
                      x0,
                      options={'maxiter': 1000,'disp':True},
                      method='SLSQP',
                      constraints=constraints,
                      bounds=Bounds(0, 1),
                      jac=vec_to_mat(g, english_cv, french_cv)).x
    argmin = np.reshape(argmin, [english_size, french_size])
    translations = {}
    for i, english_word in enumerate(english_words):
        print('Translation for "{0}":'.format(english_word))
        french_translations = sorted([(french_words[j], argmin[i,j])
                                      for j in range(french_size)],
                                     key=lambda x: x[1],
                                     reverse=True)[:NB_TRANSLATIONS]
        translations[english_word] = french_translations
        for fw in french_translations:
            print('- {0} (score: {1})'.format(*fw))
    return translations

english_text = 'i am a potato and i love potato and i have potato blood in my veins'
french_text  = 'je suis une patate et j aime la patate et j ai du sang de patate dans mes veines'
lexicon = {'i': ['je'], 'blood': ['sang']}
print(translate(english_text, french_text, lexicon))
