import argparse
import itertools
import multiprocessing
import mne
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import spatial, stats
from tqdm import tqdm

def process_subjects():
    erp_dict = {k : dict() for k in colors.keys()}
    for s in tqdm(range(1, 45+1)):

        eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s,))
        raw_f = mne.read_epochs(
                                eeg_f,
                                verbose=False,
                                preload=True,
                                )
        s_data = raw_f.get_data(picks='eeg')
        s_data_unscaled = raw_f.get_data(picks='eeg')
        ### Scaling 
        s_data = mne.decoding.Scaler(raw_f.info, \
                    scalings='mean'\
                    ).fit_transform(s_data_unscaled)
        xs = raw_f.times
        events = raw_f.events
        ### initializing ERPs
        #if s == 1:
        #    #erp_dict = {k : numpy.zeros(shape=s_data.shape[-2:]) for k in colors.keys()}

        events_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_events.tsv'.format(s,))
        with open(events_f) as i:
            all_lines = [l.strip().split('\t') for l in i.readlines()]
            header = all_lines[0]
            #rel_keys = [header.index(idx) for idx in ['PAS_score', 'accuracy']]
            rel_keys = [header.index(idx) for idx in ['PAS_score',]]
            lines = all_lines[1:]
        assert len(lines) == len(events)
        for erp, line in zip(s_data, lines):
            word = line[header.index('trial_type')]
            if word in ['_', '']:
                continue
            for key in rel_keys:
                if line[key] == '2':
                    continue
                case = mapper[line[key]]
                if word not in erp_dict[case].keys():
                    erp_dict[case][word] = list()
                erp_dict[case][word].append(erp)
                #if s > 1:
                #    erp_dict[mapper[line[key]]] = erp_dict[mapper[line[key]]] / 2
                ### just checking all is fine
                #assert erp_dict[mapper[line[key]]].shape == s_data.shape[-2:]
    current_erp_dict = {k : {word : numpy.mean(word_v, axis=0) for word, word_v in v.items()} for k, v in erp_dict.items()}
    for k, v in current_erp_dict.items():
        for word, word_v in v.items():
            assert word_v.shape[-1] == xs.shape[0]
    all_sectors = dict()

    ### analyses are across subjects
    for sector, elecs in tqdm(zones.items()):
        sector_data = {c : list() for c in mapper.values()}
        sector_name = zone_names[sector]

        ### for each condition, computing RSA
        for case, whole_erp_data in current_erp_dict.items():
            if len(whole_erp_data.keys()) < 8:
                continue
            #avg_data = {k : numpy.average(v, axis=0) for k, v in erp_data.items()}
            current_words = sorted(whole_erp_data.keys())
            erp_data = {k : v[elecs, :] for k, v in whole_erp_data.items()}
            if args.evaluation == 'rsa':
                baseline = 0.
                combs = list(itertools.combinations(current_words, r=2))
                rsa_model = [similarities[model][tuple(sorted(c))] for c in combs]

                #results = map(compute_rsa, range(erp.shape[-1]))
                results = list()
                for t in range(erp.shape[-1]):
                    results.append(compute_rsa(t, erp_data, combs,rsa_model))
                #if args.debugging:
                #    results = map(compute_rsa, tqdm(range(erp.shape[-1])))
                #else:
                #    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as pool:
                #        results = pool.map(compute_rsa, range(erp.shape[1]))
                #        pool.terminate()
                #        pool.join()
            if args.evaluation == 'correlation':
                baseline = 0.
                #combs = list(itertools.combinations(current_words, r=2))
                #rsa_model = [distances[model][tuple(sorted(c))] for c in combs]

                results = list()
                for t in range(erp.shape[-1]):
                    results.append(compute_correlation(t, erp_data))
                #results = map(compute_rsa, range(erp.shape[-1]))
                #if args.debugging:
                #    results = map(compute_correlation, range(erp.shape[-1]))
                #else:
                #    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as pool:
                #        results = pool.map(compute_correlation, range(erp.shape[1]))
                #        pool.terminate()
                #        pool.join()
            elif args.evaluation == 'pairwise':
                baseline = .5
                combs = list(itertools.combinations(current_words, r=2))
                if model in norms.keys():
                    if type(norms[model][word]) in [float]:
                        combs = [c for c in combs if norms[model][c[0]]!=norms[model][c[1]]]
                results = list()
                for t in range(erp.shape[-1]):
                    results.append(compute_pairwise(t, erp_data, combs))
                #if args.debugging:
                #    results = map(compute_pairwise, range(erp.shape[-1]))
                #else:
                #    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as pool:
                #        results = pool.map(compute_pairwise, range(erp.shape[1]))
                #        pool.terminate()
                #        pool.join()
            elif args.evaluation == 'ranking':
                baseline = .5
                results = list()
                for t in range(erp.shape[-1]):
                    results.append(compute_ranking(t, erp_data))
                #if args.debugging:
                #    results = map(compute_ranking, range(erp.shape[-1]))
                #else:
                #    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as pool:
                #        results = pool.map(compute_ranking, range(erp.shape[1]))
                #        pool.terminate()
                #        pool.join()
            corr_vec = [v[1] for v in sorted(results, key=lambda item : item[0])]

            corr_vec = numpy.array(corr_vec)
            sector_data[case] = corr_vec
        all_sectors[sector_name] = sector_data
    return all_sectors


def rsa_encoding(current_data, test_items):
    predicted_vectors = list()
    for t in test_items:
        num = numpy.sum([current_data[w]*similarities[model][tuple(sorted([w, t]))] for w in current_data.keys() if w not in test_items], axis=0)
        #den = numpy.sum([abs(similarities[model][tuple(sorted([w, t]))]) for w in current_data.keys() if w not in test_items])
        #pred = num / den
        pred = num.copy()
        predicted_vectors.append(pred)

    return predicted_vectors

def z_score(data, test_items):
    avg_data = numpy.average([v for k, v in data.items() if k not in test_items], axis=0)
    std_data = numpy.std([v for k, v in data.items() if k not in test_items], axis=0)
    current_data = {k : (v-avg_data)/std_data for k, v in data.items()}
    return current_data

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

def compute_rsa(t, erp_data, combs,rsa_model):
    #print([sector_name, elecs])
    t_data = {k : v[:, t] for k, v in erp_data.items()}
    ### z-scoring all items
    #current_data = z_score(t_data, [])
    current_data = t_data.copy()
    t_corrs = [scipy.stats.pearsonr(current_data[w_one], current_data[w_two])[0] for w_one, w_two in combs]
    corr = scipy.stats.pearsonr(rsa_model, t_corrs)[0]
    return (t, corr)

def compute_correlation(t, erp_data):
    t_data = {k : v[:, t] for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        current_data = z_score(t_data, [test_item])
        #current_data = t_data.copy()
        predictions = rsa_encoding(current_data, [test_item])
        predicted_vector = predictions[0]

        score = scipy.stats.pearsonr(predicted_vector, t_data[test_item])[0]
        #score = scipy.stats.spearmanr(predicted_vector, t_data[test_item])[0]

        accuracies.append(score)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def compute_ranking(t, erp_data):
    t_data = {k : v[:, t] for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        #current_data = z_score(t_data, [test_item])
        current_data = t_data.copy()
        predictions = rsa_encoding(current_data, [test_item])
        predicted_vector = predictions[0]

        scores = {w : scipy.stats.pearsonr(erp, predicted_vector)[0] for w, erp in t_data.items()}
        ### sorting and looking at ranking
        sorted_w = [v[0] for v in sorted(scores.items(), key=lambda item : item[1], reverse=True)]
        rank = 1 - (sorted_w.index(test_item) / len(sorted_w))
        accuracies.append(rank)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def compute_pairwise(t, erp_data, combs):
    #print([sector_name, elecs])
    t_data = {k : v[:, t] for k, v in erp_data.items()}
    ### these are the test sets
    accuracies = list()
    for w_one, w_two in combs:
        #current_data = z_score(t_data, [w_one, w_two])
        current_data = t_data.copy()
        predictions = rsa_encoding(current_data, [w_one, w_two])
        pred_one = predictions[0]
        pred_two = predictions[1]
        ### match
        match = 0.
        match += scipy.stats.pearsonr(pred_one, t_data[w_one])[0]
        match += scipy.stats.pearsonr(pred_two, t_data[w_two])[0]
        ### mismatch
        mismatch = 0.
        mismatch += scipy.stats.pearsonr(pred_one, t_data[w_two])[0]
        mismatch += scipy.stats.pearsonr(pred_two, t_data[w_one])[0]
        if match > mismatch:
            accuracies.append(1.)
        else:
            accuracies.append(0.)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def norm_minus_one_one(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [2*((x-min(labels))/(max(labels)-min(labels)))-1 for x in labels]
    assert min(norm_labels) == -1
    assert max(norm_labels) == 1
    vectors = {n : l for n, l in zip(names, norm_labels)}
    return vectors

def one_two_norm(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [((x-min(labels))/(max(labels)-min(labels)))+1 for x in labels]
    assert min(norm_labels) == 1
    assert max(norm_labels) == 2
    vectors = {n : l for n, l in zip(names, norm_labels)}
    return vectors

def invert_and_norm_minus_one_one(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [-(2*((x-min(labels))/(max(labels)-min(labels)))-1) for x in labels]
    assert min(norm_labels) == -1
    assert max(norm_labels) == 1
    vectors = {n : l for n, l in zip(names, norm_labels)}
    return vectors

def zero_one_norm(vectors):
    labels = [k[1] for k in vectors]
    names = [k[0] for k in vectors]
    norm_labels = [(x-min(labels))/(max(labels)-min(labels)) for x in labels]
    assert min(norm_labels) == 0
    assert max(norm_labels) == 1
    vectors = {n : l for n, l in zip(names, norm_labels)}
    return vectors

global args
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--debugging', action='store_true',)
parser.add_argument('--evaluation', choices=['rsa', 'correlation', 'pairwise', 'ranking',], required=True)
args = parser.parse_args()
folder = os.path.join(args.folder, 'derivatives')

subjects = list(range(1 ,45+1))
#subjects = list(range(1 ,5+1))
global colors
colors = {
          'low' : 'seagreen',
          #'mid' : 'deepskyblue',
          'high' : 'hotpink',
          }
global mapper
mapper = {
          '1' : 'low',
          #'2' : 'mid',
          '3' : 'high',
          }
global elec_mapper
elec_mapper = ['A{}'.format(i) for i in range(1, 33)] +['B{}'.format(i) for i in range(1, 33)] +['C{}'.format(i) for i in range(1, 33)] +['D{}'.format(i) for i in range(1, 33)]
elec_mapper = {e_i : e for e_i,e in enumerate(elec_mapper)}
global inverse_mapper
inverse_mapper = {v : k for k, v in elec_mapper.items()}

### read zones
global zones
zones = {i : list() for i in range(1, 14)}
zones[0] = list(range(128))
with open(os.path.join('data', 'ChanPos.tsv')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        zones[int(line[6])].append(inverse_mapper[line[0]])
#for i in range(1, 14):
#    del zones[i]
global zone_names
zone_names = {
              0 : 'whole_brain',
              1 : 'left_frontal',
              2 : 'right_frontal',
              3 : 'left_fronto-central',
              4 : 'right_fronto-central',
              5 : 'left_posterior-central',
              6 : 'right_posterior-central',
              7 : 'left_posterior',
              8 : 'right_posterior',
              9 : 'frontal_midline',
              10 : 'central',
              11 : 'posterior_midline',
              12 : 'left_midline',
              13 : 'right_midline',
              }

### reading norms
global norms
with open(os.path.join('data', 'word_norms.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()
            norms = {h : dict() for h in line[1:]}
            counter += 1
            continue
        for h in norms.keys():
            norms[h][line[0]] = float(line[header.index(h)])

print('now computing pairwise distances...')
### computing all distances
global distances
distances = dict()
all_combs = [tuple(sorted(v)) for v in itertools.combinations(norms[h].keys(), r=2)]
for norm_type in [
                  'OLD20', 
                  'wac_log10_frequency',
                  'wac_raw_frequency',
                  'opensubs_log10_frequency',
                  'opensubs_raw_frequency',
                  'joint_corpora_raw_frequency', 
                  'joint_corpora_log10_frequency', 
                  'word_length', 
                  'semantic_category', 
                  'concreteness', 
                  'aoa',
                  ]:
    distances[norm_type] = dict()
    for w_one, w_two in all_combs:
        distances[norm_type][(w_one, w_two)] = abs(norms[norm_type][w_one] - norms[norm_type][w_two])
### perceptual
senses = ['vision', 'smell', 'taste', 'hearing', 'touch']
distances['perceptual'] = dict()
for w_one, w_two in all_combs:
    vec_one = [norms[s][w_one] for s in senses]
    vec_two = [norms[s][w_two] for s in senses]
    distances['perceptual'][(w_one, w_two)] = scipy.spatial.distance.cosine(vec_one, vec_two)
### levenshtein
distances['levenshtein'] = dict()
for w_one, w_two in all_combs:
    distances['levenshtein'][(w_one, w_two)] = levenshtein(w_one, w_two)

### scaling in 0 to +1
#for h, h_scores in distances.items():
#    #distances[h] = zero_one_norm(h_scores.items())
#    distances[h] = one_two_norm(h_scores.items())
#    distances[h] = minus_one_one_norm(h_scores.items())

### turning distances into similarities
#similarities = {h : {k : 1-val for k, val in v.items()} for h, v in distances.items()}

## turning distances into similarities
## in a scale from 1 to 2
global similarities
similarities = dict()
for h, h_scores in distances.items():
    similarities[h] = invert_and_norm_minus_one_one(h_scores.items())
    #similarities[h] = zero_one_norm(h_scores.items())

### loaading similarities and scaling them
for f in os.listdir('similarities'):
    key = f.split('_')[0]
    f_sims = dict()
    with open(os.path.join('similarities', f)) as i:
        counter = 0
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            f_sims[(line[0], line[1])] = float(line[2])
    similarities[key] = norm_minus_one_one(f_sims.items())

global general_folder
general_folder = 'across-subject-rsa_plots'

for model in [
              'wac_log10_frequency',
              'wac_raw_frequency',
              'opensubs_log10_frequency',
              'opensubs_raw_frequency',
              'w2v',
              'concreteness',
              'joint_corpora_log10_frequency',
              'joint_corpora_raw_frequency',
              'semantic_category', 
              'levenshtein',
              'word_length', 
              'aoa',
              'perceptual',
              'fasttext', 
              'fasttext-aligned', 
              'visual',
              'OLD20',
              ]:
    print(model)
    out_folder = os.path.join(general_folder, model, args.evaluation)
    os.makedirs(out_folder, exist_ok=True)
    results = process_subjects()

    ### reading times
    eeg_f = os.path.join(
                         folder, 
                         'sub-01',
                         'sub-01_task-namereadingimagery_eeg-epo.fif.gz',
                         )
    raw_f = mne.read_epochs(
                            eeg_f,
                            verbose=False,
                            preload=True,
                            )
    xs = raw_f.times
    if args.evaluation in ['rsa', 'correlation']:
        baseline = 0.
    elif args.evaluation in ['pairwise', 'ranking']:
        baseline = 0.5


    for sector, elecs in tqdm(zones.items()):
        sector_data = {c : list() for c in mapper.values()}
        sector_name = zone_names[sector]
        for k, v in results[sector_name].items():
            sector_data[k].append(v)

        fig, ax = pyplot.subplots(figsize=(22,10), constrained_layout=True)
        for k, v in sector_data.items():
            ax.plot(xs, numpy.average(v, axis=0), color=colors[k], label=k)
            ax.fill_between(xs,
                            y1=numpy.average(v, axis=0)-scipy.stats.sem(v, axis=0),
                            y2=numpy.average(v, axis=0)+scipy.stats.sem(v, axis=0),
                            color=colors[k],
                            alpha=0.2,
                            )
        height = 0.5
        ax.vlines(x=0., ymin=baseline-.02, ymax=baseline+.1, color='black')
        ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=baseline-0.02, ymax=baseline+.1, linestyle='dashdot', color='gray', alpha=0.6)
        ax.hlines(y=baseline, xmin=min(xs), xmax=max(xs), color='black')
        ax.legend(fontsize=20)
        title = '{} RSA analysis for in {}'.format(model, sector_name)
        ax.set_title(title)
        pyplot.savefig(os.path.join(out_folder, '{}_rsa_{}.jpg'.format(model, sector_name)))
        pyplot.clf()
        pyplot.close()
