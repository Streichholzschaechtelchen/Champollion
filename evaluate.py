import json

def evaluate_translations(translations_file):
    with open(translations_file, 'r') as f:
        translations = json.load(f)
    i, l = 1, len(translations)
    scores = {0: 0}
    for french_word, english_words in translations.items():
        print('{0}/{1} Which of the following means {2}' \
              .format(i, l, french_word))
        for j, english_word_ in enumerate(english_words):
            english_word, _ = english_word_
            print('- [{0}] {1}'.format(j + 1, english_word))
        i += 1
        m = len(english_words)
        answers = [str(i) for i in range(m)] + ['0']
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
    print('RESULTS:')
    for item in scores.items():
        print('{0}: {1}'.format(*item))
    print('OVERALL PRECISION: {0}'.format((l - scores[0]) / l))
    return score
        
def auto_evaluate_translations(translations_file, solution_file, lexicon_file):
    with open(translations_file, 'r') as f:
        translations = json.load(f)
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    with open(lexicon_file, 'r') as f:
        lexicon = json.load(f)    
    l = len(translations)
    scores = {0: 0}
    in_lex, out = 0, 0
    for french_word, english_words in translations.items():
        if french_word in lexicon:
            in_lex += 1
            continue
        if french_word not in solution:
            print('Missing translations for {0}!'.format(french_word))
            out += 1
            continue
        for j, english_word in enumerate(english_words):
            if english_word[0] in solution[french_word]:
                print(french_word, english_word[0])
                if j + 1 in scores:
                    scores[j + 1] += 1
                else:
                    scores[j + 1] = 1
                break
        else:
            scores[0] += 1
    print('RESULTS:')
    for item in scores.items():
        print('{0}: {1}'.format(*item))
    print('In lexicon: {0}'.format(in_lex))
    print('Missing: {0}'.format(out))
    ll = l - in_lex - out
    print('OVERALL PRECISION: {0}'.format((ll - scores[0]) / ll))
    return scores
