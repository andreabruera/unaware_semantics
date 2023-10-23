import collections
import fasttext
import gensim
import itertools
import os
import numpy
import scipy
import sklearn

from gensim.models import Word2Vec
from matplotlib import image, pyplot
from nltk.corpus import wordnet
from scipy import spatial, stats
from skimage import metrics
from sklearn import metrics
from tqdm import tqdm

word_dict = dict()

words = list()
with open(os.path.join('data', 'chosen_words.txt')) as stimuli_file:
    for i, l in enumerate(stimuli_file):
        if i > 0: 
            l = l.strip().split('\t')
            words.append(l[0])

out_folder = 'similarities'
os.makedirs(out_folder, exist_ok=True)
vec_folder = 'vectors'
os.makedirs(vec_folder, exist_ok=True)

###fasttext
ft = fasttext.load_model(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', "cc.it.300.bin"))
aligned_ft_file = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', "wiki.it.align.vec")
aligned_ft = dict()
with open(aligned_ft_file) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(' ')
        aligned_ft[line[0]] = numpy.array(line[1:], dtype=numpy.float64)

w2v = Word2Vec.load(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', 'word2vec_it_opensubs+wac_param-mandera2017', 'word2vec_it_opensubs+wac_param-mandera2017.model'))
baroni_w2v = Word2Vec.load(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', 'word2vec_it_opensubs+wac_param-baroni2014_min-count-50', 'word2vec_it_opensubs+wac_param-baroni2014_min-count-50.model'))

for model_name, model in [
                          ['fasttext', ft], 
                          ['fasttext-aligned', aligned_ft], 
                          ['w2v', w2v],
                          ['w2v-baroni', baroni_w2v]
                          ]:

    print('Now computing pairwise similarities...')
    combs = [tuple(sorted(k)) for k in itertools.combinations(words, 2)]
    sims = dict()
    for c in tqdm(combs):
        if 'fasttext' in model_name:
            vec_one = model[c[0]]
            vec_two = model[c[1]]
        elif 'w2v' in model_name:
            vec_one = model.wv[c[0]]
            vec_two = model.wv[c[1]]
        sims[c] = [scipy.stats.pearsonr(vec_one, vec_two)[0]]
        sims[c].append(1-scipy.spatial.distance.cosine(vec_one, vec_two))

    with open(os.path.join(out_folder, '{}_similarities.tsv'.format(model_name)), 'w') as o:
        o.write('Word 1\tWord 2\tpearson_correlation\tcosine_similarity\n')
        for c, res in sims.items():
            o.write('{}\t{}\t{}\t{}\n'.format(c[0], c[1], res[0], res[1]))
    with open(os.path.join(vec_folder, '{}_vectors.tsv'.format(model_name)), 'w') as o:
        o.write('word\tvector\n')
        for w in words:
            o.write('{}\t'.format(w))
            if 'fasttext' in model_name:
                vec = model[w]
            elif 'w2v' in model_name:
                vec = model.wv[w]
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')

### wordnet
### reading vectors
wn_vecs = dict()
with open(os.path.join('data', 'wn2vec.txt')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(' ')
        wn_vecs[line[0]] = numpy.array(line[1:], dtype=numpy.float64)

en_to_it = dict()
en_to_wn = dict()
with open(os.path.join('data', 'wordnet_words.txt')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        if line[1] not in wn_vecs.keys():
            print(line[1])
        en_to_it[line[1]] = line[0]
        en_to_wn[line[1]] = [syn.strip() for syn in line[2].split(',')]
path_wn_sims = dict()
wup_wn_sims = dict()
vecs_wn_sims = dict()
for en, it in en_to_it.items():
    for en_two, it_two in en_to_it.items():
        if en == en_two:
            continue
        words = list()
        path_tup_sims = list()
        wup_tup_sims = list()
        vecs_sim = scipy.spatial.distance.cosine(wn_vecs[en], wn_vecs[en_two])
        vecs_wn_sims[tuple(sorted([it, it_two]))] = vecs_sim
        for w in [en, en_two]:
            words.append(wordnet.synsets(w))
        combs = list(itertools.product(words[0], words[1]))
        for c in combs:
            path_sim = wordnet.path_similarity(c[0], c[1])
            wup_sim = wordnet.wup_similarity(c[0], c[1])
            path_tup_sims.append(path_sim)
            wup_tup_sims.append(wup_sim)
        path_avg_sims = numpy.median(path_tup_sims)
        wup_avg_sims = numpy.median(wup_tup_sims)
        path_wn_sims[tuple(sorted([it, it_two]))] = path_avg_sims
        wup_wn_sims[tuple(sorted([it, it_two]))] = wup_avg_sims

with open(os.path.join(out_folder, 'wordnet_similarities.tsv'), 'w') as o:
    o.write('Word 1\tWord 2\tpath_similarity\twup_similarity\tvector_similarity\n')
    for c, path in path_wn_sims.items():
        o.write('{}\t{}\t{}\t{}\n'.format(c[0], c[1], path, wup_wn_sims[c], vecs_wn_sims[c]))

with open(os.path.join(vec_folder, 'wordnet_vectors.tsv'), 'w') as o:
    o.write('word\tvector\n')
    for en, w in en_to_it.items():
        o.write('{}\t'.format(w))
        vec = wn_vecs[en]
        for dim in vec:
            o.write('{}\t'.format(float(dim)))
        o.write('{}\n')
