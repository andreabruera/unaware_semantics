import itertools
import numpy
import os
import scipy
import sklearn

from scipy import sparse, stats
from sklearn import linear_model

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

def rsa_encoding(current_data, test_items, model):
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

def compute_rsa_correlation(t, erp_data, model):
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
    #t_data = {k : numpy.average(v[:, t], axis=1) for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        current_data = z_score(t_data, [test_item])
        #current_data = t_data.copy()
        predictions = rsa_encoding(current_data, [test_item], model)
        predicted_vector = predictions[0]

        score = scipy.stats.pearsonr(predicted_vector, t_data[test_item])[0]
        if str(score) == 'nan':
            print('corrected')
            score = 0.

        accuracies.append(score)
    corr = numpy.average(accuracies)
    #print(corr)
    return (t, corr)

def compute_ridge_correlation(t, erp_data, model):
    t_data = {k : v[:, t].flatten() for k, v in erp_data.items()}
    accuracies = list()
    for test_item in t_data.keys():
        ### z-scoring
        current_data = z_score(t_data, [test_item])
        #current_data = t_data.copy()
        ridge = sklearn.linear_model.RidgeCV(alphas=(0.01, 0.1, 1., 10, 100., 1000))
        train_items = [w for w in current_data.keys() if w!=test_item]
        if model not in ['visual', 'wordnet', 'fasttext']:
            train_input = [numpy.array([t]) for t in train_input]
        else:
            train_input = [vectors[model][w] for w in train_items]
        train_target = [current_data[w] for w in train_items]
        ridge.fit(train_input, train_target)
        if model not in ['visual', 'wordnet', 'fasttext']:
            test_input = numpy.array([test_input])
        else:
            test_input = [vectors[model][test_item]]
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

def read_norms_vectors_sims_dists():

    global norms
    global vectors
    global distances
    global similarities

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
    distances = dict()
    all_combs = [tuple(sorted(v)) for v in itertools.combinations(norms[h].keys(), r=2)]
    for norm_type in [
                      'coltheart_N',
                      'OLD20', 
                      'wiki_log10_frequency',
                      'wiki_raw_frequency',
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
                      'vision',
                      'touch',
                      'hearing',
                      'smell',
                      'taste',
                     ]:
        distances[norm_type] = dict()
        for w_one, w_two in all_combs:
            distances[norm_type][(w_one, w_two)] = abs(norms[norm_type][w_one] - norms[norm_type][w_two])
    for norm_type in [
                      'log_OLD20', 
                      'log_word_length', 
                      'log_aoa',
                      'log_concreteness', 
                     ]:
        distances[norm_type] = dict()
        for w_one, w_two in all_combs:
            distances[norm_type][(w_one, w_two)] = abs(numpy.log10(norms[norm_type.replace('log_', '')][w_one]) - numpy.log10(norms[norm_type.replace('log_', '')][w_two]))
    ### perceptual
    senses = ['vision', 'smell', 'taste', 'hearing', 'touch']
    distances['perceptual'] = dict()
    for w_one, w_two in all_combs:
        vec_one = [norms[s][w_one] for s in senses]
        vec_two = [norms[s][w_two] for s in senses]
        distances['perceptual'][(w_one, w_two)] = scipy.spatial.distance.euclidean(vec_one, vec_two)
    ### emotional
    emotions = ['valence', 'arousal', 'dominance',]
    distances['emotional'] = dict()
    for w_one, w_two in all_combs:
        vec_one = [norms[s][w_one] for s in emotions]
        vec_two = [norms[s][w_two] for s in emotions]
        distances['emotional'][(w_one, w_two)] = scipy.spatial.distance.euclidean(vec_one, vec_two)
    ### levenshtein
    distances['levenshtein'] = dict()
    for w_one, w_two in all_combs:
        distances['levenshtein'][(w_one, w_two)] = levenshtein(w_one, w_two)
    distances['full_orthographic'] = dict()
    ortho = ['OLD20', 'word_length',]
    vowels = ['a', 'e', 'i', 'o', 'u']
    for w_one, w_two in all_combs:
        vec_one = [norms[s][w_one] for s in ortho] + [sum([1 for let in w_one if let in vowels])]
        vec_two = [norms[s][w_two] for s in ortho] + [sum([1 for let in w_two if let in vowels])]
        distances['full_orthographic'][(w_one, w_two)] = scipy.spatial.distance.euclidean(vec_one, vec_two)

    ### scaling in 0 to +1
    #for h, h_scores in distances.items():
    #    #distances[h] = zero_one_norm(h_scores.items())
    #    distances[h] = one_two_norm(h_scores.items())
    #    distances[h] = minus_one_one_norm(h_scores.items())

    ### turning distances into similarities
    #similarities = {h : {k : 1-val for k, val in v.items()} for h, v in distances.items()}

    ## turning distances into similarities
    ## in a scale from 1 to 2
    similarities = dict()
    for h, h_scores in distances.items():
        similarities[h] = invert_and_norm_minus_one_one(h_scores.items())
        #similarities[h] = zero_one_norm(h_scores.items())

    ### loading similarities and scaling them
    for f in os.listdir('similarities'):
        #key = f.split('_')[0]
        key = f.replace('_similarities.tsv', '')
        if key == 'visual':
            idxs = [2, 3]
        if key == 'wordnet':
            idxs = list(range(2, 12))
        else:
            ### using wup for wn
            idxs = [3]

        for idx in idxs:
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
                    line = l.strip().split('\t')
                    if counter == 0:
                        if len(idxs) >= 2:
                            key = line[idx].lower().replace(' ', '_')
                        counter += 1
                        continue
                    ### correlation/path/ or cosine/wup/
                    f_sims[(line[0], line[1])] = float(line[idx])
            print(key)
            similarities[key] = norm_minus_one_one(f_sims.items())

    ### adding models to vectors
    vectors['fasttext'] = dict()
    with open(os.path.join('vectors', 'fasttext_vectors.tsv')) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            vectors['fasttext'][line[0]] = numpy.array(line[1:], dtype=numpy.float32)
    '''
    vectors['wordnet'] = dict()
    for w in vectors['fasttext'].keys():
        w_wn_vec = list()
        for w_two in vectors['fasttext'].keys():
            if w == w_two:
                continue
            w_wn_vec.append(similarities['wordnet'][tuple(sorted([w, w_two]))])
        vectors['wordnet'][w] = numpy.array(w_wn_vec, dtype=numpy.float64)
    vectors['visual'] = dict()
    for w in vectors['fasttext'].keys():
        w_wn_vec = list()
        for w_two in vectors['fasttext'].keys():
            if w == w_two:
                continue
            w_wn_vec.append(similarities['visual'][tuple(sorted([w, w_two]))])
        vectors['visual'][w] = numpy.array(w_wn_vec, dtype=numpy.float64)
    '''

    return norms, vectors, similarities, distances
