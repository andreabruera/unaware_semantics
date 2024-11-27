import argparse
import matplotlib
import mne
import numpy
import os
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats
from tqdm import tqdm

def font_setup(font_folder):
    ### Font setup
    # Using Helvetica as a font
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

font_setup('../../fonts')

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--stats', choices=['mean', 'median'], required=True)
args = parser.parse_args()
folder = os.path.join(args.folder, 'derivatives')

subjects = list(range(1 ,45+1))
#subjects = list(range(1 ,5+1))
colors = {
          'wrong' : 'gray',
          'correct' : 'orange',
          'low' : 'forestgreen',
          'mid' : 'lightskyblue',
          'high' : 'hotpink',
          }
mapper = {
          'wrong' : 'wrong',
          'correct' : 'correct',
          '1' : 'low',
          '2' : 'mid',
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
              }


for s in tqdm(subjects):

    eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                          'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s,))
    raw_f = mne.read_epochs(
                            eeg_f,
                            verbose=False,
                            preload=True)
    s_data = raw_f.get_data(picks='eeg')
    #s_data = mne.decoding.Scaler(scalings='mean').fit_transform(s_data)
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
        rel_keys = [header.index(idx) for idx in ['PAS_score', 'accuracy']]
        lines = all_lines[1:]
    assert len(lines) == len(events)
    for erp, line in zip(s_data, lines):
        for key in rel_keys:
            try:
                erp_dict[mapper[line[key]]][s].append(erp)
            except KeyError:
                erp_dict[mapper[line[key]]][s] = [erp]
            #if s > 1:
            #    erp_dict[mapper[line[key]]] = erp_dict[mapper[line[key]]] / 2
            ### just checking all is fine
            #assert erp_dict[mapper[line[key]]].shape == s_data.shape[-2:]

out_folder = os.path.join('erp_plots', args.stats)
os.makedirs(out_folder, exist_ok=True)

all_erps = dict()
present = dict()
for k, v in erp_dict.items():
    all_erps[k] = list()
    present[k] = list()
    for _, s in v.items():
        if args.stats == 'median':
            t_avg = numpy.median(s, axis=0)
        if args.stats == 'mean':
            t_avg = numpy.average(s, axis=0)
        assert t_avg.shape == (128, 282)
        all_erps[k].append(t_avg)
        present[k].append(_-1)
    all_erps[k] = numpy.array(all_erps[k])

tests = dict()
### comparisons
for k_one_i, k_one in enumerate(sorted(all_erps.keys())):
    if k_one in ['correct', 'wrong']:
        continue
    for k_two_i, k_two in enumerate(sorted(all_erps.keys())):
        if k_two in ['correct', 'wrong']:
            continue
        if k_two_i <= k_one_i:
            continue
        assert k_one != 'mid'
        ### removing missing subject
        if 'mid' in (k_one, k_two):
            one = all_erps[k_one][present['mid'], :]
        else:
            one = all_erps[k_one]
        if args.stats == 'median':
            one = numpy.median(one[:, zones[10], :], axis=1)
            two = numpy.median(all_erps[k_two][:, zones[10], :], axis=1)
        if args.stats == 'mean':
            #one = numpy.average(one[:, zones[10], :], axis=1)
            #two = numpy.average(all_erps[k_two][:, zones[10], :], axis=1)
            one = one[:, zones[10], :]
            two = all_erps[k_two][:, zones[10], :]
            one = one.reshape(-1, one.shape[-1])
            two = two.reshape(-1, two.shape[-1])
        diff = one-two
        ts, _, ps, __ = mne.stats.permutation_cluster_1samp_test(
                                                 diff, 
                                                 dict(start=0, step=0.2),
                                                 adjacency=None,
                                                 )
        print([p_i for p_i, p in enumerate(ps) if p<0.05])
        assert len(ps) == len(xs)
        tests[tuple(sorted((k_one, k_two)))] = (ts, ps)
'''
### now computing the median across trials
if args.stats == 'median':
    erp_dict = {k : numpy.median(v, axis=0) for k, v in erp_dict.items()}
if args.stats == 'mean':
    erp_dict = {k : numpy.mean(v, axis=0) for k, v in erp_dict.items()}
for k, v in erp_dict.items():
    assert v.shape == s_data.shape[-2:]
'''

print('now saving sectors to file')
### sectors separately
current_out_folder = os.path.join(out_folder, 'sectors')
os.makedirs(current_out_folder, exist_ok=True)
for sector, elecs in zones.items():
    #current_erp_dict = {k : numpy.median(v[elecs, :], axis=0) for k, v in erp_dict.items()}
    current_err_dict = {k : scipy.stats.sem(numpy.mean(v[:, elecs, :], axis=0), axis=0) for k, v in all_erps.items()}
    if args.stats == 'mean':
        current_erp_dict = {k : numpy.mean(numpy.mean(v[:, elecs, :], axis=0), axis=0) for k, v in all_erps.items()}
    if args.stats == 'median':
        current_erp_dict = {k : numpy.mean(numpy.mean(v[:, elecs, :], axis=0), axis=0) for k, v in all_erps.items()}
    for k, v in current_erp_dict.items():
        assert v.shape == xs.shape
    plots = {
             'pas' : ['low',
                      'mid',
                      'high'],
             'accuracy' : ['wrong', 'correct']
             }
    for case, keys in plots.items():
        fig, ax = pyplot.subplots(figsize=(22,10), constrained_layout=True)
        for k in keys:
            ax.plot(xs, current_erp_dict[k], color=colors[k], label=k, linewidth=4)
            ax.fill_between(xs, y1=current_erp_dict[k]-current_err_dict[k], y2=current_erp_dict[k]+current_err_dict[k], color=colors[k], alpha=0.2)
        height = 2
        ax.vlines(x=0., ymin=-1.5*1e-6, ymax=height*1e-6, color='black')
        ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
        ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-1.5*1e-6, ymax=height*1e-6, linestyle='dashdot', color='gray', alpha=0.6)
        ax.legend(fontsize=25, ncols=3, loc=2)
        title = '{} electrodes - ERP analysis per {}'.format(zone_names[sector].capitalize(), case.upper())
        if sector == 10:
            ax.text(
                    x=-.1,
                    y=-.6*1e-6,
                    s='ERP\ndifference\np<0.05',
                    fontsize=18,
                    fontstyle='italic',
                    fontweight='bold',
                    #va='center',
                    #ha='center',
                    )
            counter = 3
            for k, v in tests.items():
                f = os.path.join(current_out_folder, '{}_{}_erp_t-tests_{}-{}.tsv'.format(zone_names[sector], case, k[0], k[1]))
                with open(f, 'w') as o:
                    o.write('time\tt_value\tp_value\n')
                    for t_i in range(len(v[0])):
                        time = xs[t_i]
                        t = v[0][t_i]
                        p = v[1][t_i]
                        o.write('{}\t{}\t{}\n'.format(time, t, p))
                p_xs = [i for i, v in enumerate(v[1]) if v<0.05]
                if len(p_xs) > 0:
                    ax.scatter(
                            [xs[idx] for idx in p_xs],
                            [(.3*-counter)*1e-6 for idx in p_xs],
                            color=colors[k[0]],
                            )
                    ax.scatter(
                            [xs[idx] for idx in p_xs],
                            [((.3*-counter)-0.03)*1e-6 for idx in p_xs],
                            color=colors[k[1]],
                            )
                    ax.text(
                            x=-.1,
                            y=((.3*-counter)-0.03)*1e-6,
                            s='{} vs {}'.format(k[0], k[1]),
                            fontsize=18,
                            )
                    counter += 1
        ax.set_title(title, fontsize=35, fontweight='bold')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        pyplot.ylabel('$\\mu$Volt', fontsize=25, fontweight='bold')
        pyplot.xlabel('Seconds', fontsize=25, fontweight='bold')
        pyplot.yticks(ticks=[-1e-6, 0, 1e-6, 2*1e-6], labels=[-1, 0, 1, 2], fontsize=20)
        pyplot.xticks(fontsize=20)
        pyplot.savefig(os.path.join(current_out_folder, '{}_{}_erps.jpg'.format(zone_names[sector], case)), dpi=300)
        pyplot.clf()
        pyplot.close()

'''
print('now saving electrodes to file')
### all electrodes separately
current_out_folder = os.path.join(out_folder, 'electrodes')
os.makedirs(current_out_folder, exist_ok=True)
for elec in range(erp.shape[-2]):
    current_erp_dict = {k : v[elec, :] for k, v in erp_dict.items()}
    for k, v in current_erp_dict.items():
        assert v.shape == xs.shape
    plots = {
             'pas' : ['low',
                      #'mid',
                      'high'],
             'accuracy' : ['wrong', 'correct']
             }
    for case, keys in plots.items():
        fig, ax = pyplot.subplots(figsize=(22,10), constrained_layout=True)
        for k in keys:
            ax.plot(xs, current_erp_dict[k], color=colors[k], label=k)
        height = 3
        ax.vlines(x=0., ymin=-height*1e-6, ymax=height*1e-6, color='black')
        ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
        ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-height*1e-6, ymax=height*1e-6, linestyle='dashdot', color='gray', alpha=0.6)
        ax.legend(fontsize=20)
        title = 'electrode {} - ERP analysis for {}'.format(elec_mapper[elec], case)
        ax.set_title(title)
        pyplot.savefig(os.path.join(current_out_folder, '{}_{}_erps.jpg'.format(elec_mapper[elec], case)))
        pyplot.clf()
        pyplot.close()
'''
tests = dict()
### comparisons
for k_one_i, k_one in enumerate(sorted(all_erps.keys())):
    if k_one in ['correct', 'wrong']:
        continue
    for k_two_i, k_two in enumerate(sorted(all_erps.keys())):
        if k_two in ['correct', 'wrong']:
            continue
        if k_two_i <= k_one_i:
            continue
        assert k_one != 'mid'
        ### removing missing subject
        if 'mid' in (k_one, k_two):
            one = all_erps[k_one][present['mid'], :]
        else:
            one = all_erps[k_one]
        if args.stats == 'median':
            one = numpy.median(one, axis=1)
            two = numpy.median(all_erps[k_two], axis=1)
        if args.stats == 'mean':
            one = numpy.average(one, axis=1)
            two = numpy.average(all_erps[k_two], axis=1)
        diff = one-two
        ts, _, ps, __ = mne.stats.permutation_cluster_1samp_test(
                                                 diff, 
                                                 dict(start=0, step=0.2),
                                                 adjacency=None,
                                                 )
        print([p_i for p_i, p in enumerate(ps) if p<0.05])
        assert len(ps) == len(xs)
        tests[tuple(sorted((k_one, k_two)))] = (ts, ps)

print('now saving the overall {} to file'.format(args.stats))
### taking the mean of all electrodes
if args.stats == 'median':
    current_erp_dict = {k : numpy.median(numpy.median(v, axis=1), axis=0) for k, v in all_erps.items()}
if args.stats == 'mean':
    current_erp_dict = {k : numpy.mean(numpy.mean(v, axis=1), axis=0) for k, v in all_erps.items()}
for k, v in current_erp_dict.items():
    assert v.shape == xs.shape
plots = {
         'pas' : ['low',
                  'mid',
                  'high'],
         'accuracy' : ['wrong', 'correct']
         }
for case, keys in plots.items():
    fig, ax = pyplot.subplots(figsize=(22,10), constrained_layout=True)
    for k in keys:
        ax.plot(xs, current_erp_dict[k], color=colors[k], label=k)
    height = 0.5
    #ax.vlines(x=0., ymin=-height*1e-7, ymax=height*1e-7, color='black')
    #ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-height*1e-7, ymax=height*1e-7, linestyle='dashdot', color='gray', alpha=0.6)
    ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
    ax.vlines(x=0., ymin=-height*1e-6, ymax=height*1e-6, color='black')
    ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-height*1e-6, ymax=height*1e-6, linestyle='dashdot', color='gray', alpha=0.6)
    ax.legend(fontsize=25, loc=9, ncols=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if case == 'pas':
        counter = 0
        for k, v in tests.items():
            p_xs = [i for i, v in enumerate(v[1]) if v<0.05]
            if len(p_xs) > 0:
                ax.scatter(
                        [xs[idx] for idx in p_xs],
                        [(-3-counter)*1e-7 for idx in p_xs],
                        color=colors[k[0]],
                        )
                ax.scatter(
                        [xs[idx] for idx in p_xs],
                        [(-3.05-counter)*1e-7 for idx in p_xs],
                        color=colors[k[1]],
                        )
                ax.text(
                        x=-.1,
                        y=(-3.-counter)*1e-7,
                        s='{} vs {}'.format(k[0], k[1]),
                        fontsize=18,
                        )
                counter += 1
    pyplot.ylabel('Volt', fontsize=25, fontweight='bold')
    pyplot.xlabel('Seconds', fontsize=25, fontweight='bold')
    pyplot.xticks(fontsize=18)
    pyplot.yticks(fontsize=18)
    title = 'Whole-scalp ERP analysis for {}'.format(case)
    ax.set_title(title, fontweight='bold', fontsize=25)
    pyplot.savefig(os.path.join(out_folder, 'whole_scalp_{}_erps.jpg'.format(case)), dpi=300)
    pyplot.clf()
    pyplot.close()
