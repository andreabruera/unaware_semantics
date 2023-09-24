import collections
import fasttext
import gensim
import itertools
import os
import numpy
import scipy

from gensim.models import Word2Vec
from matplotlib import image, pyplot
from scipy import spatial, stats
from skimage import metrics
from tqdm import tqdm

word_dict = dict()

words = list()
with open(os.path.join('data', 'chosen_words.txt')) as stimuli_file:
    for i, l in enumerate(stimuli_file):
        if i > 0: 
            l = l.strip().split('\t')
            words.append(l[0])

###fasttext
#ft = fasttext.load_model(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', "cc.it.300.bin"))
w2v = Word2Vec.load(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', 'word2vec_it_opensubs+wac_param-mandera2017', 'word2vec_it_opensubs+wac_param-mandera2017.model'))

for model_name, model in [
                          #['fasttext', ft], 
                          ['w2v', w2v]
                          ]:

    print('Now computing pairwise similarities...')
    combs = [tuple(sorted(k)) for k in itertools.combinations(words, 2)]
    sims = dict()
    for c in tqdm(combs):
        if model_name == 'fasttext':
            vec_one = model[c[0]]
            vec_two = model[c[1]]
        elif model_name == 'w2v':
            vec_one = model.wv[c[0]]
            vec_two = model.wv[c[1]]
        sims[c] = [scipy.stats.pearsonr(vec_one, vec_two)[0]]
        sims[c].append(1-scipy.spatial.distance.cosine(vec_one, vec_two))
    out_folder = 'similarities'
    os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(out_folder, '{}_similarities.tsv'.format(model_name)), 'w') as o:
        o.write('Word 1\tWord 2\tpearson_correlation\tcosine_similarity\n')
        for c, res in sims.items():
            o.write('{}\t{}\t{}\t{}\n'.format(c[0], c[1], res[0], res[1]))
