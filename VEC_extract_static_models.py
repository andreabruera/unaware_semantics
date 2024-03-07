import collections
import fasttext
import gensim
import itertools
import os
import numpy
import pickle
import scipy
import sklearn

from gensim.models import Word2Vec
from matplotlib import image, pyplot
from nltk.corpus import wordnet, wordnet_ic
from scipy import spatial, stats
from skimage import metrics
from sklearn import metrics
from tqdm import tqdm

word_dict = dict()

words = list()
en_words = list()
with open(os.path.join('data', 'chosen_words.txt')) as stimuli_file:
    for i, l in enumerate(stimuli_file):
        if i > 0: 
            l = l.strip().split('\t')
            words.append(l[0])
            en_words.append(l[1])

out_folder = 'similarities'
os.makedirs(out_folder, exist_ok=True)
vec_folder = 'vectors'
os.makedirs(vec_folder, exist_ok=True)

'''
concept_net = dict()
concept_net_en = dict()
with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'numberbatch-19.08.txt')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(' ')
        lang_word = line[0].split('/')
        lang = lang_word[-2]
        word = lang_word[-1]
        if lang == 'it' and word in words:
            vec = numpy.array(line[1:], dtype=numpy.float64)
            concept_net[word] = vec
        if lang == 'en' and word in en_words:
            vec = numpy.array(line[1:], dtype=numpy.float64)
            concept_net_en[words[en_words.index(word)]] = vec
for w in words:
    assert w in concept_net.keys()
    assert w in concept_net_en.keys()
'''
### transe
transe_en = dict()
with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'en', 'transe_vectors.pkl'), 'rb') as i:
    all_vecs = pickle.load(i)
import pdb; pdb.set_trace()
for word, en_word in zip(words, en_words):
    transe_en[word] = all_vecs[en_word]
'''

###fasttext
ft = fasttext.load_model(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', "cc.it.300.bin"))
en_ft = fasttext.load_model(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'en', "cc.en.300.bin"))

w2v = Word2Vec.load(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', 'word2vec_it_opensubs+wac_param-mandera2017', 'word2vec_it_opensubs+wac_param-mandera2017.model'))
baroni_w2v = Word2Vec.load(os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors', 'it', 'word2vec_it_opensubs+wac_param-baroni2014_min-count-50', 'word2vec_it_opensubs+wac_param-baroni2014_min-count-50.model'))
'''

for model_name, model in [
                          #['fasttext', ft], 
                          #['fasttext-en', en_ft], 
                          #['w2v', w2v],
                          #['conceptnet', concept_net],
                          #['conceptnet-en', concept_net_en],
                          ['transe-en', transe_en]
                          ]:

    print('Now computing pairwise similarities...')
    combs = [tuple(sorted(k)) for k in itertools.combinations(words, 2)]
    sims = dict()
    for c in tqdm(combs):
        if 'w2v' in model_name:
            vec_one = model.wv[c[0]]
            vec_two = model.wv[c[1]]
        #if 'fasttext' in model_name or 'zscored' in model_name or 'concept' in model_name:
        else:
            vec_one = model[c[0]]
            vec_two = model[c[1]]
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
            if model_name == 'fasttext-en':
                vec = model[en_words[words.index(w)]]
            elif 'fasttext' in model_name:
                vec = model[w]
            elif 'w2v' in model_name:
                vec = model.wv[w]
            for dim in vec:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')

'''

### wordnet

ic = wordnet_ic.ic('ic-semcor.dat')
en_to_it = dict()
en_to_wn = dict()
with open(os.path.join('data', 'wordnet_words.txt')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        en_to_it[line[1]] = line[0]
        en_to_wn[line[1]] = [syn.strip() for syn in line[2].split(',')]
path_wn_sims = dict()
wup_wn_sims = dict()
res_wn_sims = dict()
jcn_wn_sims = dict()
lin_wn_sims = dict()
med_path_wn_sims = dict()
med_wup_wn_sims = dict()
med_res_wn_sims = dict()
med_jcn_wn_sims = dict()
med_lin_wn_sims = dict()
for en, it in en_to_it.items():
    for en_two, it_two in en_to_it.items():
        if en == en_two:
            continue
        words = list()
        path_tup_sims = list()
        wup_tup_sims = list()
        res_tup_sims = list()
        jcn_tup_sims = list()
        lin_tup_sims = list()
        for w in [en, en_two]:
            words.append(wordnet.synsets(w))
        combs = list(itertools.product(words[0], words[1]))
        for c in combs:
            ### basic measures
            path_sim = wordnet.path_similarity(c[0], c[1])
            wup_sim = wordnet.wup_similarity(c[0], c[1])
            path_tup_sims.append(path_sim)
            wup_tup_sims.append(wup_sim)
            ### other measures
            if c[1].pos() == c[0].pos():
                res_sim = c[0].res_similarity(c[1], ic)
                jcn_sim = c[0].jcn_similarity(c[1], ic)
                lin_sim = c[0].lin_similarity(c[1], ic)
                res_tup_sims.append(res_sim)
                jcn_tup_sims.append(jcn_sim)
                lin_tup_sims.append(lin_sim)
        ### averages
        path_avg_sims = numpy.average(path_tup_sims)
        wup_avg_sims = numpy.average(wup_tup_sims)
        res_avg_sims = numpy.average(res_tup_sims)
        jcn_avg_sims = numpy.average(jcn_tup_sims)
        lin_avg_sims = numpy.average(lin_tup_sims)
        ###
        path_wn_sims[tuple(sorted([it, it_two]))] = path_avg_sims
        wup_wn_sims[tuple(sorted([it, it_two]))] = wup_avg_sims
        res_wn_sims[tuple(sorted([it, it_two]))] = res_avg_sims
        jcn_wn_sims[tuple(sorted([it, it_two]))] = jcn_avg_sims
        lin_wn_sims[tuple(sorted([it, it_two]))] = lin_avg_sims
        ### medians
        path_median_sims = numpy.median(path_tup_sims)
        wup_median_sims = numpy.median(wup_tup_sims)
        res_median_sims = numpy.median(res_tup_sims)
        jcn_median_sims = numpy.median(jcn_tup_sims)
        lin_median_sims = numpy.median(lin_tup_sims)
        ### 
        med_path_wn_sims[tuple(sorted([it, it_two]))] = path_median_sims
        med_wup_wn_sims[tuple(sorted([it, it_two]))] = wup_median_sims
        med_res_wn_sims[tuple(sorted([it, it_two]))] = res_median_sims
        med_jcn_wn_sims[tuple(sorted([it, it_two]))] = jcn_median_sims
        med_lin_wn_sims[tuple(sorted([it, it_two]))] = lin_median_sims

with open(os.path.join(out_folder, 'wordnet_similarities.tsv'), 'w') as o:
    o.write('Word 1\tWord 2\t')
    o.write('avg_path_similarity\tmedian_path_similarity\t')
    o.write('avg_wup_similarity\tmedian_wup_similarity\t')
    o.write('avg_res_similarity\tmedian_res_similarity\t')
    o.write('avg_jcn_similarity\tmedian_jcn_similarity\t')
    o.write('avg_lin_similarity\tmedian_lin_similarity\n')
    for c, path in path_wn_sims.items():
        o.write('{}\t{}\t'.format(c[0], c[1]))
        o.write('{}\t{}\t'.format(path, med_path_wn_sims[c]))
        o.write('{}\t{}\t'.format(wup_wn_sims[c], med_wup_wn_sims[c]))
        o.write('{}\t{}\t'.format(res_wn_sims[c], med_res_wn_sims[c]))
        o.write('{}\t{}\t'.format(jcn_wn_sims[c], med_jcn_wn_sims[c]))
        o.write('{}\t{}\n'.format(lin_wn_sims[c], med_lin_wn_sims[c]))
'''
