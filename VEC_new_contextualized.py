import argparse
import numpy
import os
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--lang', required=True, choices=['en', 'it'])
args = parser.parse_args()

if args.lang == 'en':
    corpus = 'opensubs'
    mapper = dict()
    w_path = os.path.join('data', 'chosen_words.txt')
    with open(w_path) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            mapper[line[1].strip()] = line[0].strip()
else:
    corpus = 'wac'

base = os.path.join('collect_word_sentences', 'llm_vectors', args.lang, corpus)
for model in os.listdir(base):
    vecs = dict()
    for f in os.listdir(os.path.join(base, model)):
        if 'tsv' not in f:
            continue
        with open(os.path.join(base, model, f)) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                w = line[0]
                if args.lang == 'en':
                    w = mapper[w]
                vec = numpy.array(line[2:], dtype=numpy.float64)
                try:
                    vecs[w].append(vec)
                except KeyError:
                    vecs[w] = [vec] 
    layers = len(vecs[w])
    for case in [
                 #'mid-low-4', 
                 #'mid-low-6', 
                 #'mid-hi-4', 
                 #'mid-hi-6', 
                 'top-4',
                 #'top-6',
                 ]:
        if case == 'mid-low-4':
            m = int(layers*0.25)
            idx_layers = list(range(m-1, m+3))
        if case == 'mid-low-6':
            m = int(layers*0.25)
            idx_layers = list(range(m-2, m+4))
        elif case == 'mid-hi-4':
            m = int(layers*0.75)
            idx_layers = list(range(m-1, m+3))
        elif case == 'mid-hi-6':
            m = int(layers*0.75)
            idx_layers = list(range(m-2, m+4))
        elif case == 'top-4':
            idx_layers = list(range(layers-4, layers))
        elif case == 'top-6':
            idx_layers = list(range(layers-6, layers))
        print(idx_layers)
        assert len(idx_layers) in [4, 6]
        with open(os.path.join('similarities', '{}-{}-{}_similarities.tsv'.format(model, case, args.lang)), 'w') as o:
            o.write('word_one\tword_two\tspearman_correlation\tcosine_similarity\n')
            for w_one_i, w_one in enumerate(sorted(vecs.keys())):
                for w_two_i, w_two in enumerate(sorted(vecs.keys())):
                    if w_two_i <= w_one_i:
                        continue
                    v_one = numpy.average([vecs[w_one][l] for l in idx_layers], axis=0)
                    v_two = numpy.average([vecs[w_two][l] for l in idx_layers], axis=0)
                    cos = 1-scipy.spatial.distance.cosine(v_one, v_two)
                    sp = scipy.stats.spearmanr(v_one, v_two).statistic
                    ws = sorted([w_one, w_two])
                    o.write('{}\t{}\t'.format(ws[0], ws[1]))
                    o.write('{}\t{}\n'.format(sp, cos))
