from urllib import request
from bs4 import BeautifulSoup
import json
from math import ceil

FREQUENCY_LIST_URL = 'https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/1-10000'
TRANSLATION_URL = 'https://en.wiktionary.org/w/api.php?action=query&prop=iwlinks&titles={0}&iwprop=url&iwprefix={1}&format=json'

def generate_lexicon(target_lang,
                     lexicon_file='lexicon.json',
                     errors_file='errors.json'):
    frequency_list = request.urlopen(FREQUENCY_LIST_URL).read()
    lexicon = {}
    errors = []
    soup = BeautifulSoup(frequency_list, 'lxml')
    first_1000 = soup.find('table')
    lines = soup.findAll('tr')[1:1001]
    for line in lines:
        n = line.findAll('td')[0].text
        english_word = line.findAll('td')[1].find('a').text
        try:
            french_json = request.urlopen(TRANSLATION_URL.format(english_word, target_lang)).read()
            french_json = json.loads(french_json)
            french_pages_json = french_json['query']['pages']
            french_links_json = french_pages_json[list(french_pages_json)[0]]['iwlinks']
            french_words = [link['*'] for link in french_links_json]
            lexicon[english_word] = french_words
            print('{0}: Ok on {1}'.format(n, english_word))
        except:
            errors.append(english_word)
            print('{0}: Error on {1}'.format(n, english_word))
    with open(lexicon_file, 'w') as f:
        json.dump(lexicon, f, ensure_ascii=False)
    with open(errors_file, 'w') as f:
        json.dump(errors, f, ensure_ascii=False)


def edit_lexicon(lexicon_file='lexicon.json', stopwords_file=None):
    ANSWERS = ['y', 'Y', 'n', 'N', 'nta']
    print(('You will edit the base lexicon.\n'
           'For each word offering several translation alternatives, '
           'you will be offered to choose to:\n'
           '[y]   accept the current translation\n'
           '[Y]   accept the current translation and reject all other ones\n'
           '[n]   reject the current translation\n'
           '[N]   reject the current translation and all other ones\n'
           '[nta] delete the word from the lexicon\n'))
    with open(lexicon_file, 'r') as f:
        lexicon_json = json.load(f)
    if stopwords_file:
        with open(stopwords_file, 'r') as f:
            stopwords = json.load(f)
    else:
        stopwords = []
    j = 0
    new_lexicon = {}
    no_to_all = False
    for english_word, french_words in lexicon_json.items():
        j += 1
        if english_word in stopwords:
            continue
        elif len(french_words) == 1:
            if '_' in french_words[0]:
                new_lexicon[english_word] = []
            else:
                new_lexicon[english_word] = french_words
        else:
            new_french_words = []
            for french_word in french_words:
                if '_' in french_word:
                    continue
                yn = None
                while yn not in ANSWERS:
                    if yn is not None:
                        print('{0}: Please answer {1}.'\
                              .format(j, ' or '.join(ANSWERS)))
                    yn = input('{0}: Keep "{1}" as a translation of "{2}"? ({3}) '\
                               .format(j, french_word, english_word, '/'.join(ANSWERS)))
                if yn == 'y':
                    new_french_words.append(french_word)
                elif yn == 'Y':
                    new_french_words.append(french_word)
                    break
                elif yn == 'N':
                    break
                elif yn == 'nta':
                    no_to_all = True
                    break
            if not no_to_all:
                new_lexicon[english_word] = new_french_words
            else:
                no_to_all = False
    with open(lexicon_file, 'w') as f:
        json.dump(new_lexicon, f, ensure_ascii=False)

        
def fix_lexicon(lexicon_file='lexicon.json', errors_file='errors.json'):
    with open(errors_file, 'r') as f:
        errors = json.load(f)
    print(('You will fix the base lexicon.\n'
           'For each word with no translation, '
           'you will have to enter new translations, separated by commas.\n'))
    new_lexicon = {}
    j = 0

    def get_input(j, error, new_lexicon):
        input_ = None
        while not input_:
            if input_ is not None:
                print('Please enter new translations.')
            input_ = input('{0}: Enter new translations or nta for {1}. '.\
                           format(j, error))
        if input_ != 'nta':
            new_lexicon[error] = input_.split(',')
        
    for error in errors:
        j += 1
        get_input(j, error, new_lexicon)
        
    with open(lexicon_file, 'r') as f:
        lexicon = json.load(f)
        for english_word, french_words in lexicon.items():
            if len(french_words) == 0:
                j += 1
                get_input(j, english_word, new_lexicon)
            else:
                new_lexicon[english_word] = french_words

    with open(lexicon_file, 'w') as f:
        json.dump(new_lexicon, f, ensure_ascii=False)

        
def to_csv(lexicon_file='lexicon.json', lexicon_csv_file='lexicon.csv'):
    
    with open(lexicon_file, 'r') as f:
        lexicon = json.load(f)
    with open(lexicon_csv_file, 'w') as g:
        for k, v in lexicon.items():
            g.write('{0},{1}\n'.format(k, ','.join(v)))


def from_csv(lexicon_file='lexicon.json', lexicon_csv_file='lexicon.csv'):

    with open(lexicon_csv_file, 'r') as f:
        lexicon = {c[0]: [d if d[-1] != '\n' else d[:-1] for d in c[1:] if d not in ['', '\n']] for c in [line.split(',') for line in f.readlines()]}
    with open(lexicon_file, 'w') as g:
        json.dump(lexicon, g, ensure_ascii=False)
        
