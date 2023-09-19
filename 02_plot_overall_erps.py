import argparse
import mne
import numpy
import os

from matplotlib import pyplot
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()
folder = os.path.join(args.folder, 'derivatives')

subjects = list(range(1 ,45+1))
#subjects = list(range(1 ,5+1))
colors = {
          'wrong' : 'gray',
          'correct' : 'orange',
          'low' : 'seagreen',
          'mid' : 'deepskyblue',
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
    xs = raw_f.times
    events = raw_f.events
    ### initializing ERPs
    if s == 1:
        #erp_dict = {k : numpy.zeros(shape=s_data.shape[-2:]) for k in colors.keys()}
        erp_dict = {k : list() for k in colors.keys()}

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
            erp_dict[mapper[line[key]]].append(erp)
            #if s > 1:
            #    erp_dict[mapper[line[key]]] = erp_dict[mapper[line[key]]] / 2
            ### just checking all is fine
            #assert erp_dict[mapper[line[key]]].shape == s_data.shape[-2:]
### now computing the median across trials
#erp_dict = {k : numpy.median(v, axis=0) for k, v in erp_dict.items()}
erp_dict = {k : numpy.mean(v, axis=0) for k, v in erp_dict.items()}
for k, v in erp_dict.items():
    assert v.shape == s_data.shape[-2:]

out_folder = 'erp_plots'
os.makedirs(out_folder, exist_ok=True)

print('now saving sectors to file')
### sectors separately
current_out_folder = os.path.join(out_folder, 'sectors')
os.makedirs(current_out_folder, exist_ok=True)
for sector, elecs in zones.items():
    #current_erp_dict = {k : numpy.median(v[elecs, :], axis=0) for k, v in erp_dict.items()}
    current_erp_dict = {k : numpy.mean(v[elecs, :], axis=0) for k, v in erp_dict.items()}
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
        height = 2
        ax.vlines(x=0., ymin=-height*1e-6, ymax=height*1e-6, color='black')
        ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
        ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-height*1e-6, ymax=height*1e-6, linestyle='dashdot', color='gray', alpha=0.6)
        ax.legend(fontsize=20)
        title = '{} electrodes - ERP analysis for {}'.format(zone_names[sector], case)
        ax.set_title(title)
        pyplot.savefig(os.path.join(current_out_folder, '{}_{}_erps.jpg'.format(zone_names[sector], case)))
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

print('now saving the overall median to file')
### taking the mean of all electrodes
#current_erp_dict = {k : numpy.median(v, axis=0) for k, v in erp_dict.items()}
current_erp_dict = {k : numpy.mean(v, axis=0) for k, v in erp_dict.items()}
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
    height = 0.5
    #ax.vlines(x=0., ymin=-height*1e-7, ymax=height*1e-7, color='black')
    #ax.vlines(x=[0.2, 0.4, 0.6, 0.8, 1.], ymin=-height*1e-7, ymax=height*1e-7, linestyle='dashdot', color='gray', alpha=0.6)
    ax.hlines(y=0., xmin=min(xs), xmax=max(xs), color='black')
    ax.legend(fontsize=20)
    title = 'Whole-scalp ERP analysis for {}'.format(case)
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'whole_scalp_{}_erps.jpg'.format(case)))
    pyplot.clf()
    pyplot.close()
