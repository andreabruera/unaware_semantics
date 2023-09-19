import argparse
import mne
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_type',
                    choices=['derivatives', 'sourcedata'],
                    default='derivatives',
                    help='Derivatives, or sourcedata?')
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()

subjects = range(1, 45+1)
runs = range(1, 24+1)

eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7',
                     'EXG8', 'GSR1', 'GSR2',
                    'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

folder = os.path.join(args.folder, args.data_type)

if args.data_type == 'sourcedata':
    for s in tqdm(subjects):
        for r in runs:
            if s not in [14, 26, 28]:
                marker = True
            else:
                if s == 14 and r == 3:
                    marker = False
                elif s == 26 and r == 10:
                    marker = False
                elif s == 28 and r == 4:
                    marker = False
                else:
                    marker = True
            if marker:
                assert os.path.exists(os.path.join(folder, 'sub-{:02}'.format(s),
                                      'sub-{:02}_task-namereadingimagery_run-{:02}_events.tsv'.format(s, r)))
                assert os.path.exists(os.path.join(folder, 'sub-{:02}'.format(s),
                                      'sub-{:02}_task-namereadingimagery_run-{:02}_eeg.bdf'.format(s, r)))
    print('All files present in folder {}'.format(args.data_type))
    print('Now checking whether EEG and tsv events correspond')

    errors = list()

    for s in tqdm(subjects):
        for r in runs:
            if s == 14 and r == 3:
                continue
            elif s == 26 and r == 10:
                continue
            elif s == 28 and r == 4:
                continue
            marker = False
            eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                                  'sub-{:02}_task-namereadingimagery_run-{:02}_eeg.bdf'.format(s, r))
            raw_f = mne.io.read_raw(eeg_f,
                                        eog=eog_channels,
                                        exclude=excluded_channels,
                                        verbose=False,
                                        preload=True)
            events = mne.find_events(raw_f,
                                     initial_event=False,
                                     verbose=False,
                                     stim_channel='Status',
                                     #min_duration=0.5
                                     )
            max_trigger = 34
            events = events[[i_i for i_i, i in enumerate(events[:, 2] < max_trigger) if i == True], :]

            events_f = os.path.join(folder, 'sub-{:02}'.format(s),
                                  'sub-{:02}_task-namereadingimagery_run-{:02}_events.tsv'.format(s, r))
            with open(events_f) as i:
                lines = [l.strip().split('\t') for l in i.readlines()][1:]
            try:
                assert len(lines) == len(events)
            except AssertionError:
                marker = True
                errors.append([s, r, len(lines), len(events)])
            for l, e in zip(lines, events):
                try:
                    assert int(l[3]) == e[-1]
                except AssertionError:
                    marker = True
            if marker == True:
                errors.append([s, r, [int(l[3]) for l in lines],
                            events[:, 2]])
        if len(errors) >=1:
            print('errors for subject {}: {}'.format(s, errors))
        else:
            print('no errors for subject {}!'.format(s))
        errors = list()

elif args.data_type == 'derivatives':
    errors = list()
    for s in tqdm(subjects):
        marker=False
        eeg_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s,))
        raw_f = mne.read_epochs(eeg_f,
                                    #eog=eog_channels,
                                    #exclude=excluded_channels,
                                    verbose=False,
                                    preload=True)
        #events = mne.find_events(raw_f,
        #                         initial_event=False,
        #                         verbose=False,
        #                         stim_channel='Status',
        #                         #min_duration=0.5
        #                         )
        #max_trigger = 34
        #events = events[[i_i for i_i, i in enumerate(events[:, 2] < max_trigger) if i == True], :]
        events = raw_f.events

        events_f = os.path.join(folder, 'sub-{:02}'.format(s),
                              'sub-{:02}_task-namereadingimagery_events.tsv'.format(s,))
        with open(events_f) as i:
            lines = [l.strip().split('\t') for l in i.readlines()][1:]
        try:
            assert len(lines) == len(events)
        except AssertionError:
            marker = True
            errors.append([s, r, len(lines), len(events)])
        for l, e in zip(lines, events):
            try:
                assert int(l[3]) == e[-1]
            except AssertionError:
                marker = True
        if marker == True:
            errors.append([s, r, [int(l[3]) for l in lines],
                        events[:, 2]])
    if len(errors) >=1:
        print('errors for subject {}: {}'.format(s, errors))
    else:
        print('no errors!')
    errors = list()
