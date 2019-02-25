import re
from numpy import exp

def _move_window(a, b, c, max_, window_size):
    if b < max_:
        b += 1
    if c - a >= window_size:
        a += 1
    return a, b, c + 1

def _map_array_in_place(f, a):
    for i, b in enumerate(a):
        a[i] = f(b)

def _build_index(words):
    return {word: i for i, word in enumerate(words)}

def _get_text(words):
    newwords = words.lower().split(' ')
    correctword = re.compile('^[^0-9\s]+$')
    return [word for word in newwords if correctword.match(word)]

def _merge(dict1, dict2, n_translations):

    mdict = {}
    for english_word in dict1:
        scores = {}
        sum1   = sum([v[1] for v in dict1[english_word]])
        for (french_word, score) in dict1[english_word]:
            scores[french_word] = 0.5 * score / sum1
        if english_word in dict2:
            sum2 = sum([v[1] for v in dict2[english_word]])
            for (french_word, score) in dict2[english_word]:
                if french_word in scores:
                    scores[french_word] += 0.5 * score / sum2
                else:
                    scores[french_word] = 0.5 * score / sum2
        scores = sorted(list(scores.items()), key=lambda x:x[1], reverse=True)[:n_translations]
        mdict[english_word] = scores
    return mdict
