import argparse
import wget
import re
import json
from os import mkdir, remove, walk
from os.path import exists, join
from shutil import rmtree
import subprocess
from bs4 import BeautifulSoup #also install lxml
import sys
from heapq import heapify, heappushpop
import networkx as nx
import pylab
import numpy as np
import matplotlib.pyplot as plt

from wikiextractor.WikiExtractor import process_dump

import benchmark as bm
import evaluate as ev
import optimizer as oz
import greedy as gy
import greedy2 as gy2

WORD_REGEXP = r'\w+'
INDEX_FORMAT = '{0}/index'
BIG_INDEX_FORMAT = '{0}/big_index'
DOWNLOAD_FORMAT = 'http://download.wikimedia.org/{0}wiki/latest/{0}wiki-latest-pages-articles.xml.bz2'
FOUND_FORMAT = '{1} ({0} words)'

def erase_line():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")

def words_from_text(text):
    return ' '.join(re.findall(WORD_REGEXP, text))

def get_prefix(s):
   return ''.join(c for c in s if c.isalnum())[:3]

def draw(graph_of_words):

    #graph = nx.Graph()
    #for edge in graph_of_words:
    #    graph.add_edges_from({edge})
    #pos = nx.spring_layout(graph)
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=graph_of_words)
    #nx.draw(graph, pos, with_labels=True)
    #pylab.show()
    pass

def hist(graph_of_words):
    
    words, graph_of_words = graph_of_words
    
    #Histogram of values in adjacency matrix
    fig1 = plt.figure(1)
    x, y = graph_of_words.shape
    plt.yscale('log', nonposy='clip')
    n, bins, patches = plt.hist(graph_of_words.reshape([x * y]), bins=25)
    fig1.show()
    
    #Histogram of number of neighbors of each word
    fig2 = plt.figure(2)
    neighbors = np.zeros([x])
    for i in range(x):
        for j in range(y):
            if i != j and graph_of_words[i][j] > 0:
                neighbors[j] += 1
    n2, bins2, patches2 = plt.hist(neighbors, bins=range(int(np.ceil(max(neighbors)))))
    print('Number of vertices: {0}'.format(x))
    print('Number of non-isolated vertices: {0}'.format(np.count_nonzero(neighbors)))
    print('Average number of neighbors: {0}'.format(np.average(neighbors)))
    fig2.show()

    while True:
        my_neighbors = []
        word = input('Print neighbors of word ')
        if word in words:
            index = words.index(word)
            for j in range(y):
                if graph_of_words[j][index] > 0:
                    my_neighbors.append((graph_of_words[j][index], words[j]))
            my_neighbors.sort(reverse=True)
            for e in my_neighbors:
                print('-{1}: {0}'.format(*e))
        else:
            print('Word {0} cannot be found in the text'.format(word))

def graph_of_words(args, text):

    from tools import _build_index, _get_text, _move_window, _map_array_in_place
    
    if not text:
         print('connot compute graph of words')
         return

    window = args.w or 5
    min_freq = args.m or 25
    f = args.f or 1

    text = _get_text(text)
    words = list(set(text))
    index = _build_index(words)

    K = len(words)
    idf = [0] * K
    for c, word in enumerate(text):
        k = index[word]
        idf[k] += 1

    i = 0
    while i < K:
        word = words[i]
        if idf[i] < min_freq:
            del words[i]
            del idf[i]
            del index[word]
            K -= 1
        else:
            index[word] = i
            i += 1

    S = len(text)
    tf = np.zeros([K, K])
    idf = np.array(idf, dtype=np.float)
    a, b, c = 0, window, 0
    while c < S:
        word = text[c]
        if word in index:
            n = index[word]
            idf[n] += 1
            for word2 in text[a:b]:
                if word2 == word or word2 not in index:
                    continue
                k = index[word2]
                tf[n][k] += 1
        a, b, c = _move_window(a, b, c, S, window)

    for i in range(K):
        for j in range(K):
            if tf[i][j] < f * idf[i] * idf[j] * window / S:
                tf[i][j] = 0
            
    maxn = max(idf)
    to_delete = []
    _map_array_in_place(lambda x: 0. if x == 0 else np.log(maxn / x) + 1, idf)
    
    return words, np.dot(np.diag(idf), tf)

def words(args, print_=True):

    if not args.a:
        print('no article title given')
        return

    if args.b:
        index = BIG_INDEX_FORMAT.format(args.wiki)
    else:
        index = INDEX_FORMAT.format(args.wiki)

    if not exists(index):
        print("index for {0} does not exist".format(args.wiki))

    prefix = get_prefix(args.a)

    index_file = join(index, prefix + '.xml')

    if not exists(index_file):
        print('Article {0} not found.'.format(args.a))
        return None
        
    with open(index_file, 'r') as i:
        tree = BeautifulSoup(i, "lxml")
        for doc in tree.find_all('doc'):
            if doc.string.replace("\n","") == args.a:
                with open(doc['href'], 'r') as i2:
                    tree2 = BeautifulSoup(i2, "lxml")
                    for doc in tree2.find_all('doc'):
                        if doc['title'] == args.a:
                            if print_:
                                print(doc.string)
                                print('({0} words)'.format(doc.string.count(' ') + 1))
                            return doc.string
               
    print('Article {0} not found.'.format(args.a))
    return None

def grep(args):

    if not args.r:
        print('no regular expression given')
        return

    regexp = re.compile(args.r)
    
    index = INDEX_FORMAT.format(args.wiki)

    if not exists(index):
        print("index for {0} does not exist".format(args.wiki))

    print("Following articles match your regexp:")

    flag = True

    for _, _, files in walk(index):
        for index_file in files:
            index_abs = join(index, index_file)
            with open(index_abs, 'r') as i:
                tree = BeautifulSoup(i, "lxml")
                for doc in tree.find_all('doc'):
                    doc.string = doc.string.replace("\n","")
                    if regexp.search(doc.string):
                        print(FOUND_FORMAT.format(doc['size'], doc.string))
                        flag = False

    if flag:
        print("No article found.")

def list_big(args):

    print("List of longest {0} articles in {1}:".format(args.n, args.wiki))
    
    big_index = BIG_INDEX_FORMAT.format(args.wiki)
    if not exists(big_index):
        print('index for {0} does not exist'.format(args.wiki))
        return

    sol = []
    len_sol = 0
    
    for _, _, files in walk(big_index):
        for index_file in files:
           index_abs = join(big_index, index_file) 
           with open(index_abs, 'r') as i:
               tree = BeautifulSoup(i, "lxml")
               for doc in tree.find_all('doc'):
                   if len_sol < args.n:
                       sol.append((int(doc['size']), doc.string))
                       len_sol += 1
                       if len_sol == args.n:
                           heapify(sol)
                   else:
                       heappushpop(sol, (int(doc['size']), doc.string))
                       
    sol.sort(reverse=True)
    for s in sol:
        print(FOUND_FORMAT.format(*s))

def add_to_index(doc, xml_file, index):
    
    prefix = get_prefix(doc['title'])
    index_file = join(index, prefix + '.xml')
    in_ = BeautifulSoup(features='lxml')
    in_entry = in_.new_tag('doc',
                           href=xml_file,
                           size=len(doc.string))
    in_entry.string = doc['title']
    with open(index_file, 'a+') as o:
        o.write(str(in_entry))

def create_index(args):

    if not exists(args.wiki):
        print('folder {0} does not exist'.format(args.wiki))
        return

    index = INDEX_FORMAT.format(args.wiki)
    big_index = BIG_INDEX_FORMAT.format(args.wiki)

    if exists(index):
        if args.ow:
            print('-ow flag set: overwriting existing index {0}'.format(index))
            rmtree(index)
        else:
            print('cannot overwrite existing index {0}: -ow flag not set'.format(index))
            return
    
    mkdir(index)

    if exists(big_index):
        if args.ow:
            print('-ow flag set: overwriting existing index {0}'.format(index))
            rmtree(big_index)
        else:
            print('cannot overwrite existing index {0}: -ow flag not set'.format(index))
            return

    mkdir(big_index)

    print('')
    
    for _, dirs, _ in walk(args.wiki):
        n_dirs = len(dirs)
        for (i_, dir_) in enumerate(dirs):
            if dir_ in ['index', 'big_index']:
                continue
            dir_abs = join(args.wiki, dir_)
            for _, _, files in walk(dir_abs):
                n_files = len(files)
                for (j, file_) in enumerate(files):
                    erase_line()
                    print('Processing file {0}/{1} in folder {2}/{3}' \
                          .format(j + 1, n_files, i_ + 1, n_dirs))
                    file_abs = join(dir_abs, file_)
                    xml_file = file_abs + '.xml'
                    with open(file_abs, 'r') as i:
                        with open(xml_file, 'w') as o:
                            tree = BeautifulSoup(i, "lxml")
                            for doc in tree.find_all('doc'):
                                del doc['id']
                                del doc['url']
                                raw_text = doc.string
                                try:
                                     doc.string = words_from_text(raw_text)
                                except:
                                     continue
                                o.write(doc.prettify(formatter="xml"))
                                add_to_index(doc, xml_file, index)
                                if len(doc.string) >= 10000:
                                    add_to_index(doc, xml_file, big_index)
                    remove(file_abs)

    erase_line()
    print('Processed all folders and created index.'.format(n_dirs))

def download(args):

    if exists(args.wiki):
        if args.ow:
            print('-ow flag set: overwriting existing folder {0}'.format(args.wiki))
            rmtree(args.wiki)
        else:
            print('cannot overwrite existing folder {0}: -ow flag not set'.format(args.wiki))
            return False
        
    url = DOWNLOAD_FORMAT.format(args.wiki)
    filename = wget.download(url)
    mkdir(args.wiki)
    subprocess.call(['python3',
                     './wikiextractor/WikiExtractor.py',
                     '-b', '250K',
                     '-o', args.wiki,
                     filename])
    remove(filename)
    
    return True

def delete(args):

    if exists(args.wiki):
        rmtree(args.wiki)
    else:
        print('folder {0} does not exist'.format(args.wiki))

def two_texts(f, args):

    english_text = f(args, print_=False)
    wiki = args.wiki
    a = args.a
    al = args.al
    args.wiki = args.wiki2
    args.a = args.a2
    args.al = args.al2
    french_text = f(args, print_=False)
    args.wiki = wiki
    args.a = a
    args.al = al
    return english_text, french_text
        
def benchmark(args, f=words):

    english_text, french_text = two_texts(f, args)
    with open(args.l, 'r') as f:
        lexicon = json.load(f)
    translations = bm.translate(english_text, french_text, lexicon)
    if args.o:
        with open(args.o, 'w') as f:
            json.dump(translations, f, ensure_ascii=False)
    return translations

def optimizer(args, f=words):

    english_text, french_text = two_texts(f, args)
    with open(args.l, 'r') as f:
        lexicon = json.load(f)
    translations = oz.translate(english_text, french_text, lexicon)
    if args.o:
        with open(args.o, 'w') as f:
            json.dump(translations, f, ensure_ascii=False)
    return translations

def greedy(args, f=words, m=gy):

    english_text, french_text = two_texts(f, args)
    with open(args.l, 'r') as f:
        lexicon = json.load(f)
    if args.f:
        f = args.f
    else:
        f = None
    translations = m.translate(english_text, french_text, lexicon, f)
    if args.o:
        with open(args.o, 'w') as f:
            json.dump(translations, f, ensure_ascii=False)
    return translations

def multiwords(args, print_=True):

    if args.a:
        articles = args.a.split(';')
    elif args.al:
        with open(args.al, 'r') as f:
            articles = json.load(f)
    texts = []
    lengths = {}
    missing = []
    for article in articles:
        args.a = article
        text = words(args, print_=False)
        if text:
            texts.append(text)
            lengths[article] = text.count(' ')
            if print_:
                print('Found {0}.'.format(article))
        else:
            missing.append(article)
            if print_:
                print('Could not find {0}.'.format(article))
    multiwords = ' '.join(texts)
    hline = '{:-^25}'.format('')
    pattern = '|{0: >6}|{1: >16}|'
    if print_:
        print('\nSummary of corpus:')
        print(hline)
        print(pattern.format('#words', 'Title'))
        print(hline)
        for k, v in lengths.items():
            print(pattern.format(v, k[:16]))
        print(hline)
        for article in missing:
            print(pattern.format('?', article[:16]))
        print(hline)
        print(pattern.format(sum(lengths.values()),
                             '{0} articles'.format(len(lengths))))
        print(hline)
        print('\n')
    return multiwords
    
#Integrate lexicon extraction to champollion.py

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command', metavar='cmd', type=str, help='a command')
    parser.add_argument('wiki', metavar='xx', type=str, help='a wiki code')
    parser.add_argument('wiki2', metavar='yy', type=str, nargs='?', help='another wiki code')
    parser.add_argument('-ow', action='store_true', help='overwrite existing files')
    parser.add_argument('-n', action='store', type=int, default=25, help='number of articles to return')
    parser.add_argument('-r', action='store', type=str, help='a regular expression')
    parser.add_argument('-a', action='store', type=str, help='an article title')
    parser.add_argument('-a2', action='store', type=str, help='another article title')
    parser.add_argument('-b', action='store_true', help='restrict search to big articles')
    parser.add_argument('-w', action='store', type=int, help='size of the window')
    parser.add_argument('-m', action='store', type=int, help='count threshold')
    parser.add_argument('-f', action='store', type=float, help='a factor')
    parser.add_argument('-l', action='store', type=str, help='a lexicon file')
    parser.add_argument('-al', action='store', type=str, help='a list of articles')
    parser.add_argument('-al2', action='store', type=str, help='another list of articles')
    parser.add_argument('-o', action='store', type=str, help='an output file')
    args = parser.parse_args()
    o = None
    if args.command == 'download':
        if download(args):
            create_index(args)
    elif args.command == 'create_index':
        create_index(args)
    elif args.command == 'delete':
        delete(args)
    elif args.command == 'list_big':
        list_big(args)
    elif args.command == 'grep':
        grep(args)
    elif args.command == 'words':
        o = words(args)
    elif args.command == 'multiwords':
        o = multiwords(args)
    elif args.command == 'draw':
        draw(graph_of_words(args, words(args, print_=False)))
    elif args.command == 'hist':
        hist(graph_of_words(args, words(args, print_=False)))
    elif args.command == 'multihist':
        hist(graph_of_words(args, multiwords(args, print_=False)))
    elif args.command == 'benchmark':
        benchmark(args)
    elif args.command == 'multibenchmark':
        benchmark(args, f=multiwords)
    elif args.command == 'evaluate':
        ev.evaluate_translations(args.o)
    elif args.command == 'optimizer':
        optimizer(args)
    elif args.command == 'multioptimizer':
        optimizer(args, f=multiwords)
    elif args.command == 'greedy':
        greedy(args)
    elif args.command == 'multigreedy':
        greedy(args, f=multiwords)
    elif args.command == 'greedy2':
        greedy(args, m=gy2)
    elif args.command == 'multigreedy2':
        greedy(args, f=multiwords, m=gy2)
    else:
        print('unknown command {0}'.format(args.command))
    if o and args.o:
        with open(args.o, 'w') as f:
            f.write(o)
        print('Wrote output in file {0}.'.format(args.o))
 
