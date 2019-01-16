import re

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
