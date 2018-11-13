import argparse
import wget
import re
from os import mkdir, remove, walk
from os.path import exists, join
from shutil import rmtree
import subprocess
from bs4 import BeautifulSoup #also install lxml
import sys
from heapq import heapify, heappushpop
import networkx as nx
import pylab

from wikiextractor.WikiExtractor import process_dump

WORD_REGEXP = r'\w+'
INDEX_FORMAT = '{0}/index{1}.xml'
BIG_INDEX_FORMAT = '{0}/big_index.xml'
DOWNLOAD_FORMAT = 'http://download.wikimedia.org/{0}wiki/latest/{0}wiki-latest-pages-articles.xml.bz2'
FOUND_FORMAT = '{1} ({0} words)'

def erase_line():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")

def words_from_text(text):
    return ' '.join(re.findall(WORD_REGEXP, text))

def draw(graph_of_words):

    graph = nx.Graph()
    for edge in graph_of_words:
        graph.add_edges_from({edge})
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=graph_of_words)
    nx.draw(graph, pos, with_labels=True)
    pylab.show()

def graph_of_words(args, text):
    
    if not text:
        print('connot compute graph of words')
        return

    window = args.w or 5
    min_freq = args.m or 5
    f = args.f or 10000
    
    words = text.lower().split(' ')
    words_freq = {}
    co_words_freq = {}
    l = len(words)
    neighbors = words[1:window+1]
    for i,word in enumerate(words):
        if word in words_freq:
            words_freq[word] += 1
        else:
            words_freq[word] = 1
            co_words_freq[word] = {}
        for neighbor in neighbors:
            if neighbor in co_words_freq:
                if word in co_words_freq[neighbor]:
                    co_words_freq[neighbor][word] += 1
                else:
                    co_words_freq[neighbor][word] = 1
            else:
                if neighbor in co_words_freq[word]:
                    co_words_freq[word][neighbor] += 1
                else:
                    co_words_freq[word][neighbor] = 1
        neighbors = neighbors[1:]
        if i + 1 < l - window:
            neighbors.append(words[i + window + 1])
    edges = {}
    for word in co_words_freq:
        for neighbor in co_words_freq[word]:
            if word != neighbor:
                if words_freq[word] >= min_freq and words_freq[neighbor] >= min_freq:
                    val = co_words_freq[word][neighbor] ** 2 / (words_freq[word] * words_freq[neighbor])
                    if val >= f * words_freq[word] * words_freq[neighbor] / l ** 2:
                        edges[(word, neighbor)] = val
    return edges

def words(args, print_=True):

    if not args.a:
        print('no article title given')
        return

    if args.b:
        index = BIG_INDEX_FORMAT.format(args.wiki)
    else:
        index = INDEX_FORMAT.format(args.wiki, '')

    in_c = 0

    regexp = re.compile('^\s*{0}\s*$'.format(args.a))

    while exists(index):
        with open(index, 'r') as i:
            tree = BeautifulSoup(i, "lxml")
            for doc in tree.find_all('doc'):
                if regexp.match(doc.string):
                    with open(doc['href'], 'r') as i2:
                        tree2 = BeautifulSoup(i2, "lxml")
                        for doc in tree2.find_all('doc'):
                            if regexp.match(doc['title']):
                                if print_:
                                    print(doc.string)
                                return doc.string
        if args.b:
            break
        in_c += 1
        index = INDEX_FORMAT.format(args.wiki, in_c)

    print('Article {0} not found.'.format(args.a))

def grep(args):

    if not args.r:
        print('no regular expression given')
        return

    regexp = re.compile(args.r)
    
    index = INDEX_FORMAT.format(args.wiki, '')
    in_c = 0

    print("Following articles match your regexp:")

    flag = True

    while exists(index):
        with open(index, 'r') as i:
            tree = BeautifulSoup(i, "lxml")
            for doc in tree.find_all('doc'):
                doc.string = doc.string.replace("\n","")
                if regexp.search(doc.string):
                    print(FOUND_FORMAT.format(doc['size'], doc.string))
                    flag = False
        in_c += 1
        index = INDEX_FORMAT.format(args.wiki, in_c)

    if flag:
        print("No article found.")

def list_big(args):

    print("List of longest {0} articles in {1}:".format(args.n, args.wiki))
    
    big_index = BIG_INDEX_FORMAT.format(args.wiki)
    if not exists(big_index):
        print('index for {0} does not exist'.format(args.wiki))
        return

    with open(big_index, 'r') as i:
        tree = BeautifulSoup(i, "lxml")
        sol = []
        len_sol = 0
        flag = False
        for doc in tree.find_all('doc'):
            doc.string = doc.string.replace("\n","")
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
    
def create_index(args):

    if not exists(args.wiki):
        print('folder {0} does not exist'.format(args.wiki))
        return

    index = INDEX_FORMAT.format(args.wiki, '')

    if exists(index):
        if args.ow:
            print('-ow flag set: overwriting existing index {0}'.format(index))
            in_c = 0
            while exists(index):
                remove(index)
                in_c += 1
                index = INDEX_FORMAT.format(args.wiki, in_c)
        else:
            print('cannot overwrite existing index {0}: -ow flag not set'.format(index))
            return

    index = INDEX_FORMAT.format(args.wiki, '')
    big_index = BIG_INDEX_FORMAT.format(args.wiki)

    if exists(big_index):
        if args.ow:
            print('-ow flag set: overwriting existing index {0}'.format(index))
            remove(big_index)
        else:
            print('cannot overwrite existing index {0}: -ow flag not set'.format(index))
            return

    in_ = BeautifulSoup(features='xml')
    big_in = BeautifulSoup(features='lxml')

    in_c = 0

    print('')
    
    for _, dirs, _ in walk(args.wiki):
        n_dirs = len(dirs)
        for (i_, dir_) in enumerate(dirs):
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
                                     words = words_from_text(raw_text)
                                     doc.string = words
                                     size = len(words)
                                except:
                                     doc.string = ''
                                     size = 0
                                o.write(doc.prettify(formatter="xml"))
                                in_entry = in_.new_tag('doc',
                                                       href=xml_file,
                                                       size=len(words))
                                in_entry.string = doc['title']
                                in_.append(in_entry)
                                if size >= 10000:
                                    big_in.append(in_entry)
                    remove(file_abs)
            if i_ > 0 and i_ % 25 == 0:
                with open(index, 'w') as in_file:
                    in_file.write(in_.prettify(formatter="xml"))
                in_ = BeautifulSoup(features="lxml")
                in_c += 1
                index = INDEX_FORMAT.format(args.wiki, in_c)

    with open(index, 'w') as in_file:
        in_file.write(in_.prettify(formatter="xml"))

    with open(big_index, 'w') as big_in_file:
        big_in_file.write(big_in.prettify(formatter="xml"))

def download(args):

    if exists(args.wiki):
        if args.ow:
            print('-ow flag set: overwriting existing folder {0}'.format(args.wiki))
            rmtree(args.wiki)
        else:
            print('cannot overwrite existing folder {0}: -ow flag not set'.format(args.wiki))
            return
        
    url = DOWNLOAD_FORMAT.format(args.wiki)
    filename = wget.download(url)
    mkdir(args.wiki)
    subprocess.call(['python3',
                     './wikiextractor/WikiExtractor.py',
                     '-b', '250K',
                     '-o', args.wiki,
                     filename])
    remove(filename)

def delete(args):

    if exists(args.wiki):
        rmtree(args.wiki)
    else:
        print('folder {0} does not exist'.format(args.wiki))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command', metavar='cmd', type=str, help='a command')
    parser.add_argument('wiki', metavar='xx', type=str, help='a wiki code')
    parser.add_argument('-ow', action='store_true', help='overwrite existing files')
    parser.add_argument('-n', action='store', type=int, default=25, help='number of articles to return')
    parser.add_argument('-r', action='store', type=str, help='a regular expression')
    parser.add_argument('-a', action='store', type=str, help='an article title')
    parser.add_argument('-b', action='store_true', help='restrict search to big articles')
    parser.add_argument('-w', action='store', type=int, help='size of the window')
    parser.add_argument('-m', action='store', type=int, help='count threshold')
    parser.add_argument('-f', action='store', type=float, help='a factor')
    args = parser.parse_args()
    if args.command == 'download':
        download(args)
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
        words(args)
    elif args.command == 'draw':
        draw(graph_of_words(args, words(args, print_=False)))
