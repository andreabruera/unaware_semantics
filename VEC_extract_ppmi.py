import argparse
import numpy
import os
import pickle
import random
import scipy

from scipy import spatial, stats
from tqdm import tqdm

def check_present_words(args, rows, vocab):
    present_words = list()
    for w in rows:
        ### for fasttext in german we only use uppercase!
        if w[0].isupper() == False and args.lang=='de':
            if args.model=='fasttext':
                #or 'lm' in args.model or 'llama' in args.model:
                continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    return present_words

def load_count_coocs(args):
    print(args.model)
    if args.lang == 'en':
        if 'bnc' in args.model:
            min_count = 10
        elif 'cc100' in args.model:
            min_count = 500
        else:
            min_count = 10
    else:
        if 'cc100' in args.model:
            if args.lang == 'it':
                min_count = 10
            else:
                min_count = 100
        else:
            min_count = 10
    #print(min_count)
    f = args.model.split('-')[0]
    base_folder = os.path.join(
                            '/',
                            'data',
                            'tu_bruera',
                            'counts',
                           args.lang, 
                           f,
                           )
    with open(os.path.join(
                            base_folder,
                           '{}_{}_uncased_word_freqs.pkl'.format(
                                                                 args.lang, 
                                                                 f
                                                                 ),
                           ), 'rb') as i:
        freqs = pickle.load(i)
    vocab_file = os.path.join(
                            base_folder,
                           '{}_{}_uncased_vocab_min_{}.pkl'.format(
                                                                   args.lang, 
                                                                   #args.model, 
                                                                   f,
                                                                   min_count
                                                                   ),
                           )
    if 'tagged_' in args.model:
        vocab_file = vocab_file.replace('.pkl', '_no-entities.pkl')
    with open(vocab_file, 'rb') as i:
        vocab = pickle.load(i)
    print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
    print('total size of the vocabulary: {:,} words'.format(max(vocab.values())))
    if 'fwd' not in args.model and 'surprisal' not in args.model:
        coocs_file = os.path.join(base_folder,
                      '{}_{}_coocs_uncased_min_{}_win_20.pkl'.format(
                                                                         args.lang,
                                                                         #args.model, 
                                                                         f,
                                                                         min_count
                                                                         ),
                           )
    else:
        coocs_file = os.path.join(base_folder,
                      '{}_{}_forward-coocs_uncased_min_{}_win_20.pkl'.format(
                                                                         args.lang,
                                                                         #args.model, 
                                                                         f,
                                                                         min_count
                                                                         )
                      )
    if 'tagged_' in args.model:
        coocs_file = coocs_file.replace('.pkl', '_no-entities.pkl')
    with open(coocs_file, 'rb') as i:
        coocs = pickle.load(i)
    return vocab, coocs, freqs

def build_ppmi_vecs(coocs, vocab, row_words, col_words, smoothing=False, power=1.):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    if power != 1.:
        pmi_mtrx = numpy.power(pmi_mtrx, power)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    #axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    axis_one_mtrx = numpy.array([1/val if val!=0 else val for val in axis_one_sum]).reshape(-1, 1)
    assert True not in numpy.isnan(axis_one_mtrx)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    #axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    axis_zero_mtrx = numpy.array([1/val if val!=0 else val for val in axis_zero_sum]).reshape(1, -1)
    assert True not in numpy.isnan(axis_one_mtrx)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    if smoothing:
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
        total_sum = pmi_mtrx.sum()
    #trans_pmi_mtrx = numpy.multiply(numpy.multiply(numpy.multiply(pmi_mtrx,1/pmi_mtrx.sum(axis=1).reshape(-1, 1)), 1/pmi_mtrx.sum(axis=0).reshape(1, -1)), pmi_mtrx.sum())
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    assert True not in numpy.isnan(trans_pmi_mtrx.flatten())
    ### checking for nans
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        assert True not in numpy.isnan(v)

    return trans_pmi_vecs

parser = argparse.ArgumentParser()
parser.add_argument('--lang', required=True, choices=['en', 'it'])
parser.add_argument('--model', required=True, choices=['opensubs-ppmi', 'wac-ppmi', 'cc100-ppmi', 'tagged_wiki-ppmi'])
args = parser.parse_args()

assert args.model.split('-')[-1] == 'ppmi'

### reading all words for a certain language
w_path = os.path.join('data', 'chosen_words.txt')
words = list()
trans = dict()
with open(w_path) as i:
    if args.lang == 'it':
        idx = 0
    else:
        idx = 1
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        w = line[idx].strip()
        trans[line[1].strip()] = line[0].strip()
       
        words.append(w)

vocab, coocs, freqs = load_count_coocs(args)
### keeping row words that are actually available
row_words = [w for w in words if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
assert len(row_words) == len(words)
present_words = check_present_words(args, row_words, list(vocab.keys()))
filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
for freq in tqdm([
                  200000,
                  #500,
                  ]):
    #if freq > max(vocab.values()):
    #    print('too many words requested, skipping!')
    #    #continue
    #    freq = 

    for row_mode in [
                     '_', 
                     #'_rowincol',
                     ]:
        for selection_mode in [
                               'top', 
                               #'random',
                               ]: 
            key = 'ppmi_{}_abs_freq_{}{}_{}_words'.format(args.model, selection_mode, row_mode, freq)
            if selection_mode == 'top':
                if row_mode == 'rowincol':
                    ctx_words = set([w for w in sorted_freqs[:freq]]+row_words)
                else:
                    ctx_words = [w for w in sorted_freqs[:freq]]
            else:
                random.seed(12)
                idxs = random.sample(range(len(sorted_freqs)), k=min(freq, len(sorted_freqs)))
                if row_mode == 'rowincol':
                    ctx_words = set([sorted_freqs[i] for i in idxs]+row_words)
                else:
                    ctx_words = [sorted_freqs[i] for i in idxs]
            ### using the basic required vocab for all tests as a basis set of words
            trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
            model = {k : v for k, v in trans_pmi_vecs.items()}
            with open(os.path.join('vectors', '{}-{}_vectors.tsv'.format(args.model, args.lang)), 'w') as o:
                o.write('word\tvector\n')
                for k, v in model.items():
                    if args.lang == 'en':
                        k = trans[k]
                    o.write('{}\t'.format(k))
                    for dim in v:
                        o.write('{}\t'.format(dim))
                    o.write('\n')
            with open(os.path.join('similarities', '{}-{}_similarities.tsv'.format(args.model, args.lang)), 'w') as o:
                o.write('word_one\tword_two\tspearman_correlation\tcosine_similarity\n')
                for w_one_i, w_one in enumerate(words):
                    for w_two_i, w_two in enumerate(words):
                        if w_two_i <= w_one_i:
                            continue
                        cos = 1-scipy.spatial.distance.cosine(model[w_one], model[w_two])
                        sp = scipy.stats.spearmanr(model[w_one], model[w_two]).statistic
                        if args.lang == 'en':
                            ws = sorted([trans[w_one], trans[w_two]])
                        else:
                            ws = sorted([w_one, w_two])
                        o.write('{}\t{}\t'.format(ws[0], ws[1]))
                        o.write('{}\t{}\n'.format(sp, cos))
            with open(os.path.join('similarities', '{}-{}_similarities.tsv'.format(args.model.replace('ppmi', 'coocs'), args.lang)), 'w') as o:
                o.write('word_one\tword_two\traw_cooc\tlog10_cooc\n')
                for w_one_i, w_one in enumerate(words):
                    for w_two_i, w_two in enumerate(words):
                        if w_two_i <= w_one_i:
                            continue
                        try:
                            c = coocs[vocab[w_one]][vocab[w_two]]
                            log_c = numpy.log10(c)
                        except KeyError:
                            c = 0
                            log_c = 0
                        if args.lang == 'en':
                            ws = sorted([trans[w_one], trans[w_two]])
                        else:
                            ws = sorted([w_one, w_two])
                        o.write('{}\t{}\t'.format(ws[0], ws[1]))
                        o.write('{}\t{}\n'.format(c, log_c))

