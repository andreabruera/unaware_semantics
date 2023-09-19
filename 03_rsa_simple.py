import argparse
import itertools
import multiprocessing
import mne
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm
def compute_corrs(t):
    print([sector_name, elecs])
    t_data = {k : v[elecs, t] for k, v in erp_data.items()}
    t_corrs = [1-scipy.stats.pearsonr(t_data[w_one], t_data[w_two])[0] for w_one, w_two in combs]
    corr = scipy.stats.pearsonr(rsa_lengths, t_corrs)[0]
    return (t, corr)

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--debugging', action='store_true',)
args = parser.parse_args()
folder = os.path.join(args.folder, 'derivatives')

subjects = list(range(1 ,45+1))
#subjects = list(range(1 ,5+1))
colors = {
          'low' : 'seagreen',
          #'mid' : 'deepskyblue',
          'high' : 'hotpink',
          }
mapper = {
          '1' : 'low',
          #'2' : 'mid',
          '3' : 'high',
          }
elec_mapper = ['A{}'.format(i) for i in range(1, 33)] +['B{}'.format(i) for i in range(1, 33)] +['C{}'.format(i) for i in range(1, 33)] +['D{}'.format(i) for i in range(1, 33)]
elec_mapper = {e_i : e for e_i,e in enumerate(elec_mapper)}
inverse_mapper = {v : k for k, v in elec_mapper.items()}

### read zones
zones = {i : list() for i in range(1, 14)}
with open(os.path.join('data', 'ChanPos.tsv')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        zones[int(line[6])].append(inverse_mapper[line[0]])
zones[14] = list(range(128))
del zones[1]
zone_names = {
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
              14 : 'whole_brain',
              }


out_folder = 'rsa_plots'
os.makedirs(out_folder, exist_ok=True)

### analyses are subject-level
for sector, elecs in tqdm(zones.items()):
    sector_data = {c : list() for c in mapper.values()}
    sector_name = zone_names[sector]
    #for s in tqdm(range(1, 3)):
    for s in tqdm(subjects):

        eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s,))
        raw_f = mne.read_epochs(
                                eeg_f,
                                verbose=False,
                                preload=True)
        s_data = raw_f.get_data(picks='eeg')
        xs = raw_f.times
        events = raw_f.events
        ### initializing ERPs
        if s == 1:
            #erp_dict = {k : numpy.zeros(shape=s_data.shape[-2:]) for k in colors.keys()}
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

        ### for each condition, computing RSA on word length
        for case, erp_data in current_erp_dict.items():
            if len(erp_data.keys()) < 8:
                continue
            #avg_data = {k : numpy.average(v, axis=0) for k, v in erp_data.items()}
            current_words = sorted(erp_data.keys())
            combs = list(itertools.combinations(current_words, r=2))
            rsa_lengths = [abs(len(w_one)-len(w_two)) for w_one, w_two in combs]
            if args.debugging:
                '''
                corr_vec = list()
                for t in tqdm(range(erp.shape[-1])):
                    t_data = {k : v[:, t] for k, v in erp_data.items()}
                    t_corrs = [1-scipy.stats.pearsonr(t_data[w_one], t_data[w_two])[0] for w_one, w_two in combs]
                    corr = scipy.stats.pearsonr(rsa_lengths, t_corrs)[0]
                    corr_vec.append(corr)
                '''
                results = map(compute_corrs, tqdm(range(erp.shape[-1])))

            else:
                with multiprocessing.Pool(processes=int(os.cpu_count()/4)) as pool:
                    results = pool.map(compute_corrs, range(erp.shape[1]))
                    pool.terminate()
                    pool.join()
            corr_vec = [v[1] for v in sorted(results, key=lambda item : item[0])]

            corr_vec = numpy.array(corr_vec)
            sector_data[case].append(corr_vec)

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
    ax.vlines(x=0., ymin=-.02, ymax=.1, color='black')
    ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=0.02, ymax=.1, linestyle='dashdot', color='gray', alpha=0.6)
    ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
    ax.legend(fontsize=20)
    title = 'Word length RSA analysis for in {}'.format(sector_name)
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'rsa_word_length_{}.jpg'.format(sector_name)))
    pyplot.clf()
    pyplot.close()
