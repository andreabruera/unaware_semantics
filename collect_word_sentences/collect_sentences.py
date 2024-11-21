import argparse
import multiprocessing
import numpy
import os
import pickle
import random
import re

from tqdm import tqdm

from readers import tagged_gutenberg_reader, tagged_leipzig_reader, cc100_original_reader, tagged_wiki_reader, wiki_reader, wac_reader, bnc_reader, opensubs_reader, paths_loader

def multiprocessing_counter(ins):
    file_path = ins[0]
    cased = ins[1]

    if args.corpus == 'tagged_wiki':
        all_sentences = tagged_wiki_reader(args, file_path)
    if args.corpus == 'wiki':
        all_sentences = wiki_reader(args, file_path)
    if args.corpus == 'wac':
        all_sentences = wac_reader(args, file_path, pos=True)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(args, file_path, pos=True)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(args, file_path)
    if args.corpus == 'cc100':
        all_sentences = cc100_original_reader(args, file_path)
    if args.corpus == 'tagged_leipzig':
        all_sentences = tagged_leipzig_reader(args, file_path)
    if args.corpus == 'tagged_gutenberg':
        all_sentences = tagged_gutenberg_reader(args, file_path)

    with tqdm() as counter:
        for sentence in all_sentences:
            #print(sentence)
            for w, p in zip(sentence['word'], sentence['pos']):
                if w in cased.keys() and p == 'NOUN':
                    try:
                        cased[w].add(tuple([word for word in sentence['word']]))
                    except KeyError:
                        cased[w] = set(tuple([word for word in sentence['word']]))
                    counter.update(1)
    return cased

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--language', 
                    choices=[
                             'it', 
                             'en', 
                             ],
                    required=True,
                    )
parser.add_argument(
                    '--corpus', 
                    choices=[
                             'tagged_wiki', 
                             'wiki', 
                             'wac', 
                             'bnc', 
                             'opensubs',
                             'cc100',
                             'tagged_leipzig',
                             'tagged_gutenberg',
                             ],
                    required=True,
                    )
global args
args = parser.parse_args()

pkls = os.path.join(
                    'sentences',
                    args.language, 
                    args.corpus, 
                    )
os.makedirs(pkls, exist_ok=True)

### reading all words for a certain language
w_path = os.path.join('..', 'data', 'chosen_words.txt')
cased = dict()
with open(w_path) as i:
    if args.language == 'it':
        idx = 0
    else:
        idx = 1
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        w = l.strip().split('\t')[idx].strip()
        cased[w] = set()
print('collecting sentences for {} words...'.format(len(cased.keys())))

paths = [[p, cased] for p in paths_loader(args)]

### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/4)) as pool:
   results = pool.map(multiprocessing_counter, paths)
   pool.terminate()
   pool.join()

all_cased = dict()
### Reorganizing results
print('now reorganizing multiprocessing results...')
for cased in tqdm(results):
    for k, v in cased.items():
        if len(v) == 0:
            continue
        try:
            all_cased[k] = all_cased[k].union(v)
        except KeyError:
            all_cased[k] = v

for k, v in all_cased.items():
    if len(v) > 1000:
        v = random.sample(list(v), k=1000)
    with open(os.path.join(pkls, '{}.tsv'.format(k)), 'w') as o:
        for s in v:
            s = [w for w in s if w not in ['\n', ' ', '\t'] and '\t' not in w and '\n' not in w]
            for w in s:
                o.write('{}\t'.format(w))
            o.write('\n')
