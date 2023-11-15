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

def process_time_resolved_subject(s):
    all_sectors = dict()

    ### analyses are subject-level
    for sector, elecs in tqdm(zones.items()):
        sector_data = {c : list() for c in mapper.values()}
        sector_name = zone_names[sector]
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
        erp_dict = {k : dict() for k in colors.keys()}

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

global general_folder
general_folder = os.path.join('plots', 'rsa', 'time_resolved')

### reading norms
global norms
global vectors
global distances
global similarities

norms, vectors, similarities, distances = read_norms_vectors_sims_dists()

for model in [
              #'wac_log10_frequency',
              #'wac_raw_frequency',
              #'opensubs_log10_frequency',
              #'opensubs_raw_frequency',
              #'w2v',
              #'concreteness',
              #'joint_corpora_log10_frequency',
              #'joint_corpora_raw_frequency',
              #'semantic_category', 
              #'levenshtein',
              #'word_length', 
              #'aoa',
              #'perceptual',
              'OLD20',
              'fasttext', 
              'wordnet', 
              #'fasttext-aligned', 
              #'visual',
              ]:
    print(model)
    out_folder = os.path.join(general_folder, model, args.evaluation)
    os.makedirs(out_folder, exist_ok=True)
    if args.debugging:
        results = map(process_time_resolved_subject, subjects)
    else:
        with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as pool:
            results = pool.map(process_time_resolved_subject, subjects)
            pool.terminate()
            pool.join()

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
        for r in results:
            for k, v in r[sector_name].items():
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
