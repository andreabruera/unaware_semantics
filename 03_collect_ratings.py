import numpy
import os
import pickle
import scipy

from nltk.corpus import wordnet
from tqdm import tqdm

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

### reading categories
cat_mapper = {
          'animal' : -1,
          'object' : 1,
          }
cats = dict()
en_to_it = dict()
with open(os.path.join('data', 'chosen_words.txt')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        cats[line[0]] = cat_mapper[line[-1]]
        en_to_it[line[1]] = line[0]

#with open(os.path.join('data', 'unawaresemantics.txt'), 'w') as o:
#    o.write('word\tvalid_pos_tags\n')
#    for w in cats.keys():
#        o.write('{}\tNOUN\n'.format(w))

ratings = {
           'aoa' : {w : list() for w in en_to_it.values()},
           'concreteness' : {w : list() for w in en_to_it.values()},
           'vision' : {w : list() for w in en_to_it.values()},
           'smell' : {w : list() for w in en_to_it.values()},
           'touch' : {w : list() for w in en_to_it.values()},
           'hearing' : {w : list() for w in en_to_it.values()},
           'taste' : {w : list() for w in en_to_it.values()},
           'valence' : {w : list() for w in en_to_it.values()},
           'arousal' : {w : list() for w in en_to_it.values()},
           'dominance' : {w : list() for w in en_to_it.values()},
           }

### reading dataset conc
with open(os.path.join('data', 'brysbaert_conc.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        idx = header.index('Conc.M')
        ratings['concreteness'][en_to_it[word]].append(float(line[idx]))


### reading dataset aoa
with open(os.path.join('data', 'kuperman_aoa.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        idx = header.index('Rating.Mean')
        ratings['aoa'][en_to_it[word]].append(float(line[idx]))

### reading dataset #1
mapper = {
          'Visual.mean' : 'vision',
          'Olfactory.mean' : 'smell',
          'Haptic.mean' : 'touch',
          'Gustatory.mean' : 'taste',
          'Auditory.mean' : 'hearing',
          }
with open(os.path.join('data', 'Lancaster_sensorimotor_norms_for_39707_words.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        for k, dest in mapper.items():
            idx = header.index(k)
            ratings[dest][en_to_it[word]].append(float(line[idx]))

### reading dataset #1
mapper = {
          'V.Mean.Sum' : 'valence',
          'A.Mean.Sum' : 'arousal',
          'D.Mean.Sum' : 'dominance',
          }
with open(os.path.join('data', 'BRM-emot-submit.csv')) as i:
    counter = 0
    for l in i:
        line = l.strip().split(',')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[1].lower()
        if word not in en_to_it.keys():
            continue
        for k, dest in mapper.items():
            idx = header.index(k)
            ratings[dest][en_to_it[word]].append(float(line[idx]))

### reading frequencies
for corpus in ['opensubs', 'wac', 'wiki']:
    freqs = pickle.load(open(os.path.join('..', 'psychorpus', 'pickles', 'it', 'it_{}_word_freqs.pkl'.format(corpus)), 'rb'))
    ratings['{}_raw_frequency'.format(corpus)] = dict()
    ratings['{}_log10_frequency'.format(corpus)] = dict()
    for w in cats.keys():
        freq = freqs[w]
        ### also considering uppercase
        if w.capitalize() in freqs.keys():
            freq += freqs[w.capitalize()]
        if w == 'passero':
            alt_w = 'passerotto'
            if alt_w in freqs.keys():
                freq += freqs[w.capitalize()]
            if alt_w.capitalize() in freqs.keys():
                freq += freqs[w.capitalize()]
        ratings['{}_raw_frequency'.format(corpus)][w] = [freq]
        ratings['{}_log10_frequency'.format(corpus)][w] = [numpy.log10(freq)]
### computing correlation between corpora
ws = list(ratings['{}_log10_frequency'.format(corpus)].keys())
oss = [ratings['wiki_raw_frequency'][w][0] for w in ws]
wac = [ratings['wac_raw_frequency'][w][0] for w in ws]
corr = scipy.stats.pearsonr(oss, wac).statistic
print('correlation between corpora: {}'.format(round(corr, 4)))

ratings['joint_corpora_raw_frequency'] = dict()
ratings['joint_corpora_log10_frequency'] = dict()
for w in cats.keys():
    ratings['joint_corpora_raw_frequency'][w] = 0
    for corpus in ['opensubs', 'wac']:
        freq = ratings['{}_raw_frequency'.format(corpus)][w][0]
        ratings['joint_corpora_raw_frequency'][w] += freq
    ratings['joint_corpora_log10_frequency'][w] = [numpy.log10(ratings['joint_corpora_raw_frequency'][w])]
    ratings['joint_corpora_raw_frequency'][w] = [ratings['joint_corpora_raw_frequency'][w]]

### computing OLD20
lemma_freqs = pickle.load(open(os.path.join('..', 'psychorpus', 'pickles', 'it', 'it_wac_lemma_freqs.pkl'), 'rb'))
max_n = 35502
other_words = [w[0] for w in sorted(lemma_freqs.items(), key=lambda item : item[1], reverse=True)][:max_n]

ratings['OLD20'] = dict()
ratings['coltheart_N'] = dict()
for w in tqdm(cats.keys()):
    lev_vals = [levenshtein(w, other_w) for other_w in other_words if other_w!=w]
    score = numpy.average(sorted(lev_vals)[:20])
    ratings['OLD20'][w] = [score]
    score = sum([1 if val==1 else 0 for val in lev_vals])
    ratings['coltheart_N'][w] = [score]
print(ratings['OLD20'])

for k, v in ratings.items():
    for w, w_v in v.items():
        assert len(w_v) >= 1
        if len(w_v) > 1:
            print(w_v)
cat_list = [
        'coltheart_N', 
        'OLD20', 
        'wiki_raw_frequency', 
        'wiki_log10_frequency', 
        'wac_raw_frequency', 
        'wac_log10_frequency', 
        'opensubs_raw_frequency', 
        'opensubs_log10_frequency',
        'joint_corpora_raw_frequency', 
        'joint_corpora_log10_frequency',
        'aoa', 
        'concreteness',
        'vision', 
        'smell', 
        'touch', 
        'taste', 
        'hearing',
        'valence', 
        'arousal', 
        'dominance',
        ]

### writing to file
with open(os.path.join('data', 'word_norms.tsv'), 'w') as o:
    o.write('word\tword_length\tsemantic_category\t')
    for cat in cat_list:
        o.write('{}\t'.format(cat))
    o.write('\n')
    for w, c in cats.items():
        o.write('{}\t{}\t{}\t'.format(w, len(w), c))
        for cat in cat_list:
            o.write('{}\t'.format(ratings[cat][w][0]))
        o.write('\n')
