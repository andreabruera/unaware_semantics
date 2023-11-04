import argparse
import itertools
import multiprocessing
import mne
import numpy
import os
import scipy
import sklearn

from matplotlib import pyplot
from scipy import spatial, stats
from sklearn import linear_model
from tqdm import tqdm

def create_adjacency_matrix(clusters):
    data = list()
    indices = list()
    index_pointer = [0]
    for elec_idx, elecs in clusters.items():
        v = elecs[1:]
        for neighbor in v:
            indices.append(int(neighbor))
            data.append(1)
        index_pointer.append(len(indices))

    ### Just checking everything went fine
    mne_sparse_adj_matrix = scipy.sparse.csr_matrix((data, indices, index_pointer), dtype=int)
    for elec_idx, elecs in clusters.items():
        v = elecs[1:]

        assert sorted([i for i, k in enumerate(mne_sparse_adj_matrix.toarray()[elec_idx]) if k == 1]) == sorted(v)

    return mne_sparse_adj_matrix 

def process_subject(current_args):

    s = current_args['subject']
    min_val = current_args['min_val']
    regression_model = current_args['regression_model']
    model = current_args['model']

    all_sectors = dict()

    ### analyses are subject-level
    for elec_idx, elecs in tqdm(clusters.items()):
        sector_data = {c : list() for c in mapper.values()}
        #for s in tqdm(range(1, 3)):

        eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s,))
        raw_f = mne.read_epochs(
                                eeg_f,
                                verbose=False,
                                preload=True,
                                )
        s_data = raw_f.get_data(picks='eeg')
        if args.evaluation not in ['correlation', 'ranking', 'pairwise']:
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
        erp_dict = {k : dict() for k in mapper.values()}

        events_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_events.tsv'.format(s,))
        with open(events_f) as i:
            all_lines = [l.strip().split('\t') for l in i.readlines()]
            header = all_lines[0]
            rel_keys = [header.index(idx) for idx in ['PAS_score', 'accuracy']]
            #rel_keys = [header.index(idx) for idx in ['PAS_score',]]
            lines = all_lines[1:]
        assert len(lines) == len(events)
        for erp, line in zip(s_data, lines):
            word = line[header.index('trial_type')]
            if word in ['_', '']:
                continue
            for key in rel_keys:
                if line[key] in ['2']:
                    continue
                case = mapper[line[key]]
                if word not in erp_dict[case].keys():
                    erp_dict[case][word] = list()
                erp_dict[case][word].append(erp)
            ### joint
            if min_val <4:
                joint_case = '{}_{}'.format(line[rel_keys[0]], line[rel_keys[1]])
                if joint_case in mapper.keys():
                    if word not in erp_dict[mapper[joint_case]].keys():
                        erp_dict[mapper[joint_case]][word] = list()
                    erp_dict[mapper[joint_case]][word].append(erp)
        current_erp_dict = {k : {word : numpy.mean(word_v, axis=0) for word, word_v in v.items() if len(word_v)>=min_val} for k, v in erp_dict.items()}
        for k, v in current_erp_dict.items():
            for word, word_v in v.items():
                assert word_v.shape[-1] == xs.shape[0]

        ### for each condition, computing RSA
        for case, whole_erp_data in current_erp_dict.items():
            #if len(whole_erp_data.keys()) < 8:
            #    continue
            #if len(whole_erp_data.keys()) < 20:
            if len(whole_erp_data.keys()) < 7:
                #print([case, len(whole_erp_data.keys())])
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
                #for t in range(erp.shape[-1]):
                for t_min, t_max in time_clusters: 
                    t = [t_i for t_i, t in enumerate(xs) if t>=t_min and t<t_max]
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
                for t_min, t_max in time_clusters: 
                    t = [t_i for t_i, t in enumerate(xs) if t>=t_min and t<t_max]
                    if regression_model == 'ridge':
                        results.append(compute_ridge_correlation(t, erp_data))
                    elif regression_model == 'rsa':
                        results.append(compute_rsa_correlation(t, erp_data))
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
                for t_min, t_max in time_clusters: 
                    t = [t_i for t_i, t in enumerate(xs) if t>=t_min and t<t_max]
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
                for t_min, t_max in time_clusters: 
                    t = [t_i for t_i, t in enumerate(xs) if t>=t_min and t<t_max]
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
        all_sectors[elec_idx] = sector_data
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
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
    ### z-scoring all items
    #current_data = z_score(t_data, [])
    current_data = t_data.copy()
    t_corrs = [scipy.stats.pearsonr(current_data[w_one], current_data[w_two])[0] for w_one, w_two in combs]
    corr = scipy.stats.pearsonr(rsa_model, t_corrs)[0]
    return (t, corr)

def compute_rsa_correlation(t, erp_data):
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        current_data = z_score(t_data, [test_item])
        #current_data = t_data.copy()
        predictions = rsa_encoding(current_data, [test_item])
        predicted_vector = predictions[0]

        score = scipy.stats.pearsonr(predicted_vector, t_data[test_item])[0]

        accuracies.append(score)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def compute_ridge_correlation(t, erp_data):
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        current_data = z_score(t_data, [test_item])
        #current_data = t_data.copy()
        ridge = sklearn.linear_model.RidgeCV(alphas=(0.01, 0.1, 1., 10, 100., 1000))
        train_items = [w for w in current_data.keys() if w!=test_item]
        train_input = [vectors[model][w] for w in train_items]
        if model not in ['wordnet', 'fasttext']:
            train_input = [numpy.array([t]) for t in train_input]
        train_target = [current_data[w] for w in train_items]
        ridge.fit(train_input, train_target)
        test_input = [vectors[model][test_item]]
        if model not in ['wordnet', 'fasttext']:
            test_input = numpy.array([test_input])
        predicted_vector = ridge.predict(test_input)[0]

        score = scipy.stats.pearsonr(predicted_vector, t_data[test_item])[0]

        accuracies.append(score)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def compute_ranking(t, erp_data):
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
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
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
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

### reading norms
global norms
global vectors
with open(os.path.join('data', 'word_norms.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line.copy()
            norms = {h : dict() for h in line[1:]}
            vectors = {h : dict() for h in line[1:]}
            counter += 1
            continue
        for h in norms.keys():
            norms[h][line[0]] = float(line[header.index(h)])
            vectors[h][line[0]] = float(line[header.index(h)])

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
                  'valence', 
                  'arousal',
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
### emotional
emotions = ['valence', 'arousal', 'dominance',]
distances['emotional'] = dict()
for w_one, w_two in all_combs:
    vec_one = [norms[s][w_one] for s in emotions]
    vec_two = [norms[s][w_two] for s in emotions]
    distances['emotional'][(w_one, w_two)] = scipy.spatial.distance.cosine(vec_one, vec_two)
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

### loading similarities and scaling them
for f in os.listdir('similarities'):
    key = f.split('_')[0]
    ### using wup for wn
    idx = 3
    '''
    if 'wordnet' in key:
        idx = 3
    if 'fasttext' in key:
        idx = 3
    else:
        idx = 2
    '''
    f_sims = dict()
    with open(os.path.join('similarities', f)) as i:
        counter = 0
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            ### correlation/path/ or cosine/wup/
            f_sims[(line[0], line[1])] = float(line[idx])
    similarities[key] = norm_minus_one_one(f_sims.items())

### adding models to vectors
vectors['fasttext'] = dict()
with open(os.path.join('vectors', 'fasttext_vectors.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        vectors['fasttext'][line[0]] = numpy.array(line[1:], dtype=numpy.float32)
vectors['wordnet'] = dict()
for w in vectors['fasttext'].keys():
    w_wn_vec = list()
    for w_two in vectors['fasttext'].keys():
        if w == w_two:
            continue
        w_wn_vec.append(similarities['wordnet'][tuple(sorted([w, w_two]))])
    vectors['wordnet'][w] = numpy.array(w_wn_vec, dtype=numpy.float64)

### testing various things: 
### 20mm vs 30mm
### 75ms vs 100ms vs 125ms vs 150ms

for regression_model in [
                        #'ridge', 
                         'rsa',
                         ]:
    for temp_cluster in [
                         'temporal_cluster', 
                         'isolated_time_points',
                         ]:
        for min_val in [
                        #1,
                        2,
                        #4,
                        ]:
            global mapper
            if min_val >= 2:
                mapper = {
                      '1' : 'low',
                      '3' : 'high',
                      'correct' : 'correct',
                      'wrong' : 'wrong',
                      }
            else:
                mapper = {
                      '1' : 'low',
                      '3' : 'high',
                      'correct' : 'correct',
                      'wrong' : 'wrong',
                      '1_correct' : 'low_correct',
                      '1_wrong' : 'low_wrong',
                      '3_correct' : 'high_correct',
                      '3_wrong' : 'high_wrong',
                      #'2' : 'mid',
                      }
            for space in [30, 20]:
                for time in [100, 75, 125, 150]:
                    global time_clusters
                    if time == 100:
                        time_clusters = [
                                         (0, .1),
                                         (.1, .2),
                                         (.2, .3),
                                         (.3, .4),
                                         (.4, .5),
                                         (.5, .6),
                                         (.6, .7),
                                         (.7, .8),
                                         (.8, .9),
                                         (.9, 1.),
                                         ]
                        sfreq = 10
                    elif time == 75:
                        time_clusters = [
                                         (0, .075),
                                         (.075, .15),
                                         (.15, .225),
                                         (.225, .3),
                                         (.3, .375),
                                         (.375, .45),
                                         (.45, .525),
                                         (.525, .6),
                                         (.6, .675),
                                         (.675, .75),
                                         (.75, .825),
                                         (.825, .9),
                                         ]
                        sfreq = 13.34
                    elif time == 125:
                        time_clusters = [
                                         (0, .125),
                                         (.125, .25),
                                         (.25, .375),
                                         (.375, .5),
                                         (.5, .625),
                                         (.625, .75),
                                         (.75, .875),
                                         (.875, 1.),
                                         ]
                        sfreq = 8
                    elif time == 150:
                        time_clusters = [
                                         (0, .15),
                                         (.15, .3),
                                         (.3, .45),
                                         (.45, .6),
                                         (.6, .75),
                                         (.75, .9),
                                         (.9, 1.),
                                         ]
                        sfreq = 6.67

                    time_points = len(time_clusters)

                    ### reading spatial clusters file
                    global elec_mapper
                    global clusters
                    elec_mapper = dict()
                    clusters = dict()
                    with open(os.path.join('data', 'searchlight_clusters_{}.0mm.txt'.format(space))) as i:
                        for l_i, l in enumerate(i):
                            if l_i == 0:
                                continue
                            line = l.strip().split('\t')
                            elec_mapper[l_i-1] = line[0]
                            clusters[l_i-1] = numpy.array(line[1:], dtype=numpy.int32)
                    mne_adj_matrix = create_adjacency_matrix(clusters)
                    if temp_cluster == 'temporal_cluster':
                        mne_ad_matrix = mne.stats.combine_adjacency(time_points, mne_adj_matrix),

                    global general_folder
                    general_folder = os.path.join(
                                                  'plots', 
                                                  regression_model,
                                                  temp_cluster,
                                                  'min_{}'.format(min_val),
                                                  'searchlight_{}mm_{}ms'.format(space, time)
                                                  )

                    for model in [
                                  #'wac_log10_frequency',
                                  #'wac_raw_frequency',
                                  #'opensubs_log10_frequency',
                                  #'opensubs_raw_frequency',
                                  #'concreteness',
                                  #'joint_corpora_log10_frequency',
                                  #'joint_corpora_raw_frequency',
                                  'fasttext', 
                                  'OLD20',
                                  'semantic_category', 
                                  'visual',
                                  'wordnet',
                                  #'levenshtein',
                                  #'word_length', 
                                  #'valence',
                                  #'arousal',
                                  #'emotional',
                                  #'perceptual',
                                  #'aoa',
                                  #'w2v',
                                  #'w2v-baroni',
                                  ]:
                        print(model)
                        current_args = [{
                             'min_val' : min_val,
                             'regression_model' : regression_model, 
                             'subject' : s,
                             'model' : model,
                             } for s in subjects]
                        out_folder = os.path.join(general_folder, model, args.evaluation)
                        os.makedirs(out_folder, exist_ok=True)
                        if args.debugging:
                            results = map(process_subject, current_args)
                        else:
                            max_procs = 40 if regression_model=='ridge' else 3
                            with multiprocessing.Pool(processes=int(os.cpu_count()/max_procs)) as pool:
                                results = pool.map(process_subject, current_args)
                                pool.terminate()
                                pool.join()

                        ### reconstructing results
                        all_results = {case : numpy.empty(shape=(45, time_points, 128)) for case in mapper.values()}
                        for s_i, s in enumerate(results):
                            to_be_removed = set()
                            s_results = {case : numpy.empty(shape=(128, time_points)) for case in mapper.values()}
                            for elec_idx, elec_results in s.items():
                                for case, case_results in elec_results.items():
                                    if len(case_results) != time_points:
                                        print(case)
                                        to_be_removed.add(case)
                                        continue
                                    s_results[case][elec_idx] = case_results
                            for remove in to_be_removed:
                                del all_results[remove]
                            for case, case_results in s_results.items():
                                all_results[case][s_i] = case_results.T

                        ### plotting each condition separately
                        if args.evaluation in ['rsa', 'correlation']:
                            baseline = 0.
                        elif args.evaluation in ['pairwise', 'ranking']:
                            baseline = 0.5

                        for case, case_results in all_results.items():
                            print(case)
                            t_stats, _, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                                    case_results-baseline,
                                    tail=1, 
                                    adjacency=mne_adj_matrix,
                                    threshold=dict(start=0, step=0.2), 
                                    n_jobs=os.cpu_count()-1, 
                                    n_permutations=4096,
                                    )
                            #print('Minimum p-value for {}: {}'.format(args.input_target_model, min(p_values)))

                            significance = .05
                            original_shape = t_stats.shape
                            avged_subjects = numpy.average(case_results, axis=0)
                            assert avged_subjects.shape == original_shape
                            significance = 0.05

                            reshaped_p = p_values.copy()
                            reshaped_p[reshaped_p>=significance] = 1.0
                            reshaped_p = reshaped_p.reshape(original_shape).T

                            reshaped_p = p_values.copy()
                            reshaped_p = reshaped_p.reshape(original_shape).T

                            #relevant_times
                            tmin = 0.

                            info = mne.create_info(
                                    ch_names=[elec_mapper[i] for i in range(128)],
                                    sfreq=sfreq,
                                    ch_types='eeg',
                                    )

                            evoked = mne.EvokedArray(
                                                avged_subjects.T, 
                                                info=info, 
                                                tmin=tmin,
                                                )

                            montage = mne.channels.make_standard_montage('biosemi128')
                            evoked.set_montage(montage)
                            output_file = os.path.join(
                                                out_folder, 
                                                '{}_{}.jpg'.format(
                                                    case,
                                                    model
                                                )
                                                )


                            ### Writing to txt
                            channels = evoked.ch_names
                            assert isinstance(channels, list)
                            assert len(channels) == reshaped_p.shape[0]
                            #assert len(times) == reshaped_p.shape[-1]
                            assert reshaped_p.shape[-1] == time_points
                            txt_path = output_file.replace('.jpg', '.txt')

                            with open(txt_path, 'w') as o:
                                o.write('Time\tElectrode\tp-value\tt-value\n')
                                for t_i in range(reshaped_p.shape[-1]):
                                    time = t_i
                                    for c_i in range(reshaped_p.shape[0]):
                                        channel = elec_mapper[c_i]
                                        p = reshaped_p[c_i, t_i]
                                        p_value = reshaped_p[c_i, t_i]
                                        t_value = t_stats.T[c_i, t_i]
                                        o.write('{}\t{}\t{}\t{}\n'.format(time, channel, p_value, t_value))

                            title = 'Searchlight for ERP: {} {}'.format(
                                                    case,
                                                    model
                                                   )

                            evoked.plot_topomap(ch_type='eeg', 
                                    time_unit='s', 
                                    times=evoked.times,
                                    ncols='auto',
                                    nrows='auto', 
                                    vmax=0.075,
                                    vmin=0.,
                                    scalings={'eeg':1.}, 
                                    cmap='Spectral_r',
                                    mask=reshaped_p<=significance,
                                    mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='black',
                                        linewidth=0, markersize=4),
                                    #colorbar=False,
                                    size = 3.,
                                    title=title,
                                    )

                            pyplot.savefig(output_file, dpi=600)
                            #pyplot.savefig(output_file.replace('jpg', 'svg'), dpi=600)
                            pyplot.clf()
                            pyplot.close()