import json

def evaluate_translations(translations_file):
    with open(translations_file, 'r') as f:
        translations = json.load(f)
    i, l = 1, len(translations)
    scores = {0: 0}
    for french_word, english_words in translations.items():
        print('{0}/{1} Which of the following means {2}?' \
              .format(i, l, french_word))
        for j, english_word_ in enumerate(english_words):
            if type(english_word_) in [tuple, list, dict]:
                english_word, _ = english_word_
            else:
                english_word = english_word_
            print('- [{0}] {1}'.format(j + 1, english_word))
        i += 1
        m = len(english_words)
        answers = [str(i+1) for i in range(m)] + ['0']
        answer = None
        while answer not in answers:
            if answer is not None:
                print('Invalid answer.')
            answer = input('Answer 0 (none of the above) or 1-{0}: '.format(m))
        answer = int(answer)
        if answer in scores:
            scores[answer] += 1
        else:
            scores[answer] = 1
    print(' ---------')
    print('| RESULTS |')
    print(' ---------')
    for item in scores.items():
        print('|{0: > 2}|{1: > 6}|'.format(*item))
    print(' ---------')
    print('|P=|{0: >6}|'.format((l - scores[0]) / l))
    print(' ---------')
    return scores
        
