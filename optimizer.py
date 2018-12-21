import numpy as np
from math import sqrt
from scipy.optimize import minimize, Bounds
from numpy.random import random_sample

WINDOW_SIZE = 5
#MIN_FREQ_SOURCE = 75
#MIN_FREQ_DEST = 25
MIN_FREQ_SOURCE = 75
MIN_FREQ_DEST = 50
NB_TRANSLATIONS = 20

#Ce n'est pas une très bonne solution
EPSILON = 0.0001
PENALTY = 100

def context_vectors(words, words_index, text, window_size=WINDOW_SIZE, min_freq=MIN_FREQ_SOURCE):

    from tools import _move_window
    
    K = len(words)
    S = len(text)

    f = [0 for _ in range(K)]
    for c, word in enumerate(text):
        k = words_index[word]
        f[k] += 1

    i = 0
    while i < K:
        word = words[i]
        if f[i] < min_freq:
            del words[i]
            del f[i]
            del words_index[word]
            K -= 1
        else:
            words_index[word] = i
            i += 1

    tf = np.zeros([K, K])
    
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
        
    maxn = max(f)
    for k in range(K):
        for k2 in range(K):
            #Comment résoudre ce problème ?
            #La formule ci-dessous fait planter l'optimisation
            #tf[k][k2] /= np.log(EPSILON + maxn ** 2 / (f[k] * f[k2]))
            #Il faut travailler sur cette métrique
            if f[k] * f[k2] == 0:
                tf[k][k2] = 0
            else:
                tf[k][k2] /= f[k] * f[k2]
    penalty_c = tf.max() * PENALTY
    for k in range(K):
        for k2 in range(K):
            tf[k][k2] = tf[k][k2] or penalty_c

    return words, words_index, tf

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
        a = np.ones([french_size, french_size])
        if output_vector:
            return _q(func(p, pt, a, english_cv, french_cv),
                      english_size, french_size)
        else:
            return func(p, pt, a, english_cv, french_cv)
    return wrapper

def f(p, pt, a, english_cv, french_cv):
    pdpt = np.matmul(p, np.matmul(french_cv, pt))
    papt = np.matmul(p, np.matmul(a, pt))
    return np.linalg.norm(english_cv - pdpt / papt) ** 2

def g(p, pt, a, english_cv, french_cv):
    pdpt = np.matmul(p, np.matmul(french_cv, pt))
    papt = np.matmul(p, np.matmul(a, pt))
    vp = (pdpt / papt - english_cv) / papt
    m1 = np.matmul(vp, np.matmul(p, french_cv))
    m2 = np.matmul(vp * pdpt / papt, np.matmul(p, a))
    return 4 * (m1 - m2)

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

    #Prepare optimization
    #Initial guess    
    x0 = np.zeros([english_size * french_size])
    #Lower and upper bounds
    lower = np.full([english_size * french_size], EPSILON)
    upper = np.full([english_size * french_size], 1)

    print('Initialize...')
    
    #Update x0 and add constraints corresponding to seed lexicon
    for english_word in english_words:
        if english_word in english_seed:
            french_translations = [w for w in lexicon[english_word]
                                   if w in french_words]
            english_index = english_words_index[english_word]
            if french_translations:
                for french_translation in french_translations:
                    french_index = french_words_index[french_translation]
                    i = english_index * french_size + french_index
                    lower[i] = upper[i] = x0[i] = 1
                for j in range(french_size):
                    if french_words[j] not in french_seed:
                        i = english_index * french_size + j
                        lower[i] = upper[i] = x0[i] = 0

    #Update x0 to make it a stochastic matrix
    for i in range(english_size):
        #Random guess for whatever word not in seed
        if english_words[i] not in english_seed:
            x0[i * french_size:(i+1) * french_size] = 1

    #Print p
    #print(english_cv)
    #print(french_cv)
    #Callback function (for debugging purposes)
    k = 0
    def callback(xk):
        nonlocal k
        k += 1
        print('#{0}'.format(k))
        #print('CALLBACK:')
        #print('x=')
        #print(np.reshape(xk, [english_size, french_size]))
        #print('fun(x)=')
        #print(fun(xk))
        #print('err(x)=')
        #print(vec_to_mat(f,english_cv,french_cv,False)(xk))
        #print('grad(x)=')
        #print(vec_to_mat(g,english_cv,french_cv)(xk))
    #callback(x0)

    print('Optimize...')
    
    #Call optimization module
    argmin = minimize(vec_to_mat(f, english_cv, french_cv, False),
                      x0,
                      options={'maxiter': 400, 'disp':True},
                      method='L-BFGS-B',
                      bounds=Bounds(lower, upper, keep_feasible=True),
                      jac=vec_to_mat(g, english_cv, french_cv),
                      callback=callback
                     ).x

    #Put output into matricial form
    argmin = np.reshape(argmin, [english_size, french_size])

    #Compute and show best translations
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

#english_text = 'i am a potato and i love potato and i have potato blood in my veins'
#french_text  = 'je suis une patate et j aime la patate et j ai du sang de patate dans mes veines'
#lexicon = {'i': ['je'], 'blood': ['sang']}
#translate(english_text, french_text, lexicon)
