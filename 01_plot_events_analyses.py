import matplotlib
import numpy
import os

from matplotlib import pyplot
from scipy import stats

def read_words_and_triggers(additional_path='data', return_questions=False):

    words_path = os.path.join(additional_path, 'chosen_features.txt')
    with open(words_path, encoding='utf8') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]

    # Checking all's fine with the stimuli
    assert len(lines) == 32
    animals = {l[0] : [f.replace('_', ' ') for f in l[1:]][:3] for l in lines[:16]}
    objects = {l[0] : [f.replace('_', ' ') for f in l[1:]][:3] for l in lines[16:]}

    assert len(animals) == len(objects)
    animal_feat_num = max([len(v) for k, v in animals.items()])
    object_feat_num = max([len(v) for k, v in objects.items()])
    assert animal_feat_num == object_feat_num

    # Finalizing the list of stimuli
    animals_and_objects = animals.copy()
    animals_and_objects.update(objects)

    # Generating triggers

    word_to_trigger = {w : w_i+1 for w_i, w in enumerate(animals_and_objects.keys())}
    word_to_trigger.update({'' : len(word_to_trigger)+1})
    #print(word_to_trigger)

    if return_questions:
        return word_to_trigger, animals_and_objects
    else:
        return word_to_trigger

### Reading questions
_, questions = read_words_and_triggers(return_questions=True)

for folder in [
               'derivatives',
               #'sourcedata'
               ]:

    out_folder = os.path.join(
                    'plots_events_analyses',
                    folder
                            )
    os.makedirs(out_folder, exist_ok=True)

    sub_files = dict()

    full_f = os.path.join('..', '..', 'dataset', 'neuroscience', 'unaware_semantics_bids', folder)
    assert os.path.exists(full_f)
    for root, direc, filez in os.walk(full_f):
        for f in filez:
            #if 'events' in f and 'original' not in f:
            if 'events' in f:
                sub = int(f.split('_')[0].replace('sub-', ''))
                if sub not in sub_files.keys():
                    sub_files[sub] = list()
                sub_files[sub].append(os.path.join(root, f))
#for k, v in sub_files.items():
#    assert len(v) == 24

    sub_data = dict()

    for k, v in sub_files.items():
        sub_data[k] = dict()
        for f_name in v:
            with open(f_name) as i:
                counter = 0
                for l in i:
                    line = l.strip().split('\t')
                    if counter == 0:
                        header = line.copy()
                        for h in header:
                            if h not in sub_data[k].keys():
                                sub_data[k][h] = list()
                                sub_data[k]['required_answer'] = list()
                        counter += 1
                        continue
                    if len(line) == 7:
                        line = ['_'] + line
                    assert len(line) == len(header)
                    for h_i, h in enumerate(header):
                        sub_data[k][h].append(line[h_i])
                        if h == 'question':
                            word = line[header.index('trial_type')]
                            if len(word) > 1:
                                if line[header.index(h)] in questions[word]:
                                    ans = 'YES'
                                else:
                                    ans = 'NO'
                            else:
                                ans = 'NO'
                            sub_data[k]['required_answer'].append(ans)

### easiest violinplot: correct vs wrong
    colors = {
              'wrong' : 'gray',
              'correct' : 'orange'
              }

    right = [len([1 for v in sub_data[s]['accuracy'] if v=='correct']) for s in sub_data.keys()]
    wrong = [len([1 for v in sub_data[s]['accuracy'] if v=='wrong']) for s in sub_data.keys()]
    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, right, 'correct'), (1, wrong, 'wrong')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'frequency of correct vs wrong answers'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'correct_wrong.jpg'))
    pyplot.clf()
    pyplot.close()

### easiest violinplot: correct vs wrong
    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    lo = [len([1 for v in sub_data[s]['PAS_score'] if v=='1']) for s in sub_data.keys()]
    mid = [len([1 for v in sub_data[s]['PAS_score'] if v=='2']) for s in sub_data.keys()]
    hi = [len([1 for v in sub_data[s]['PAS_score'] if v=='3']) for s in sub_data.keys()]
    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, lo, 'low'), (1, mid, 'mid'), (2, hi, 'high')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'frequency of PAS scores'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'frequency_PAS.jpg'))
    pyplot.clf()
    pyplot.close()

### more complicated violinplot: accuracy per cat

    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    lows = list()
    mids = list()
    highs = list()

    accuracies = {s : list() for s in sub_data.keys()}

    for s in sub_data.keys():
        low = numpy.nanmean([1 if v=='correct' else 0 for v, pas in zip(sub_data[s]['accuracy'], sub_data[s]['PAS_score']) if pas=='1'])
        #try:
        #    low = len([v for v in low if v==1]) / len([v for v in low if v==0])
        #except ZeroDivisionError:
        #    low = 1.
        accuracies[s].append(low)
        lows.append(low)
        mid = numpy.nanmean([1 if v=='correct' else 0 for v, pas in zip(sub_data[s]['accuracy'], sub_data[s]['PAS_score']) if pas=='2'])
        #try:
        #    mid = len([v for v in mid if v==1]) / len([v for v in mid if v==0])
        #except ZeroDivisionError:
        #    mid = 1.
        accuracies[s].append(mid)
        if str(mid) != 'nan':
            mids.append(mid)
        high = numpy.nanmean([1 if v == 'correct' else 0 for v, pas in zip(sub_data[s]['accuracy'], sub_data[s]['PAS_score']) if pas =='3'])
        #try:
        #    high = len([v for v in high if v==1]) / len([v for v in high if v==0])
        #except ZeroDivisionError:
        #    high = 1.
        accuracies[s].append(high)
        highs.append(high)


    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, lows, 'low'), (1, mids, 'mid'), (2, highs, 'high')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'accuracy per pas category'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'accuracy_per_pas.jpg'))
    pyplot.clf()
    pyplot.close()

### more complicated violinplot: d-prime per cat

    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    lows = list()
    mids = list()
    highs = list()

    d_primes = {s : list() for s in sub_data.keys()}
    ### smoothing d-primes following Hutus 1995
    ### 1 is added to each cell before division
    ### to avoid having 0s

    for s in sub_data.keys():
        for _, pas_val in enumerate(['1', '2', '3']):
            ### hits
            hits = len([1 for s_p, s_a, req in zip(sub_data[s]['PAS_score'], sub_data[s]['accuracy'], sub_data[s]['required_answer']) if s_p==pas_val and s_a=='correct' and req=='YES'])
            total_yes = len([1 for s_p, req in zip(sub_data[s]['PAS_score'], sub_data[s]['required_answer']) if req=='YES'])
            hit_rate = (hits + 1) / total_yes
            z_hit = stats.norm.ppf(hit_rate)
            ### false alarms
            false_alarms = len([1 for s_p, s_a, req in zip(sub_data[s]['PAS_score'], sub_data[s]['accuracy'], sub_data[s]['required_answer']) if s_p==pas_val and s_a=='wrong' and req=='NO'])
            total_no = len([1 for s_p, req in zip(sub_data[s]['PAS_score'], sub_data[s]['required_answer']) if req=='NO'])
            false_alarms_rate = (false_alarms + 1) / total_no
            z_fa = stats.norm.ppf(false_alarms_rate)
            ### d-prime
            d_prime = z_hit - z_fa
            assert str(d_prime) not in ['inf', '-inf', 'nan']
            #if str(d_prime) in ['inf', '-inf', 'nan']:
            #    d_prime = 0.
            d_primes[s].append(d_prime)
            if pas_val == '1':
                lows.append(d_prime)
            elif pas_val == '2':
                mids.append(d_prime)
            elif pas_val == '3':
                highs.append(d_prime)

    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, lows, 'low'), (1, mids, 'mid'), (2, highs, 'high')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'd-prime per pas category'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'd-prime_per_pas.jpg'))
    pyplot.clf()
    pyplot.close()

    ### writing to file d-primes
    with open(os.path.join(out_folder, 'd-prime_per_subject.tsv'), 'w') as o:
        o.write('subject\tlow_awareness_d-prime\tmid_awareness_d-prime\thigh_awareness_d-prime\n')
        for s, dprimes in d_primes.items():
            o.write('{}\t{}\t{}\t{}\n'.format(s, dprimes[0], dprimes[1], dprimes[2]))

### violinplot: RTs correct vs wrong
    colors = {
              'wrong' : 'gray',
              'correct' : 'orange'
              }

    right = [numpy.nanmean([float(v) for v_i, v in enumerate(sub_data[s]['response_time']) if sub_data[s]['accuracy'][v_i]=='correct']) for s in sub_data.keys()]
    wrong = [numpy.nanmean([float(v) for v_i, v in enumerate(sub_data[s]['response_time']) if sub_data[s]['accuracy'][v_i]=='wrong']) for s in sub_data.keys()]
    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, right, 'correct'), (1, wrong, 'wrong')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'RTs of correct vs wrong answers'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'RTs_correct_wrong.jpg'))
    pyplot.clf()
    pyplot.close()

### more complicated violinplot: accuracy per cat

    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    lows = list()
    mids = list()
    highs = list()

    accuracies = {s : list() for s in sub_data.keys()}

    for s in sub_data.keys():
        low = numpy.nanmean([float(v) for v, pas in zip(sub_data[s]['response_time'], sub_data[s]['PAS_score']) if pas=='1'])
        #try:
        #    low = len([v for v in low if v==1]) / len([v for v in low if v==0])
        #except ZeroDivisionError:
        #    low = 1.
        accuracies[s].append(low)
        lows.append(low)
        mid = numpy.nanmean([float(v) for v, pas in zip(sub_data[s]['response_time'], sub_data[s]['PAS_score']) if pas=='2'])
        #try:
        #    mid = len([v for v in mid if v==1]) / len([v for v in mid if v==0])
        #except ZeroDivisionError:
        #    mid = 1.
        accuracies[s].append(mid)
        if str(mid) != 'nan':
            mids.append(mid)
        high = numpy.nanmean([float(v) for v, pas in zip(sub_data[s]['response_time'], sub_data[s]['PAS_score']) if pas =='3'])
        #try:
        #    high = len([v for v in high if v==1]) / len([v for v in high if v==0])
        #except ZeroDivisionError:
        #    high = 1.
        accuracies[s].append(high)
        highs.append(high)


    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, lows, 'low'), (1, mids, 'mid'), (2, highs, 'high')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'accuracy RTs per pas category'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'RTs_accuracy_per_pas.jpg'))
    pyplot.clf()
    pyplot.close()

    ### barplot: PAS portions
    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    low = [sum([1 for _, v in enumerate(sub_data[s]['PAS_score']) if v=='1']) for s in sorted(sub_data.keys())]
    mid = [sum([1 for _, v in enumerate(sub_data[s]['PAS_score']) if v=='2']) for s in sorted(sub_data.keys())]
    high = [sum([1 for _, v in enumerate(sub_data[s]['PAS_score']) if v=='3']) for s in sorted(sub_data.keys())]
    assert len(low) == 45
    assert len(mid) == 45
    assert len(highs) == 45

    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(21, 10))
    for bottom, y, label in [([0 for l in low], low, 'low'), (low, mid, 'mid'), ([l+m for l, m in zip(low, mid)], high, 'high')]:
        ax.bar(range(1, len(sub_data.keys())+1), height=y, bottom=bottom, color=colors[label], label=label)
    ax.set_xticks(range(1, len(sub_data.keys())+1))
    ax.hlines(y=792, xmin=0., xmax=46,
            linestyles='dashdot',
            label='max number of trials available', color='black')
    ax.legend(fontsize=18)
    title = 'PAS proportions per subject'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'PAS_proportions_per_subject.jpg'))
    pyplot.clf()
    pyplot.close()

    ### barplot: accuracy portions
    colors = {
              'wrong' : 'gray',
              'correct' : 'orange'
              }

    wrong = [sum([1 for _, v in enumerate(sub_data[s]['accuracy']) if v=='wrong']) for s in sorted(sub_data.keys())]
    correct = [sum([1 for _, v in enumerate(sub_data[s]['accuracy']) if v=='correct']) for s in sorted(sub_data.keys())]

    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(21, 10))
    for bottom, y, label in [([0 for l in lows], correct, 'correct'), (correct, wrong, 'wrong')]:
        ax.bar(range(1, len(sub_data.keys())+1), height=y, bottom=bottom, color=colors[label], label=label)
    ax.set_xticks(range(1, len(sub_data.keys())+1))
    ax.hlines(y=792, xmin=0., xmax=46,
              linestyles='dashdot', label='max number of trials available',
              color='black')
    ax.legend(fontsize=18)
    title = 'Accuracy proportions per subject'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'accuracy_proportions_per_subject.jpg'))
    pyplot.clf()
    pyplot.close()

### violinplot: PAS RTs correct vs wrong
    colors = {
              'wrong' : 'gray',
              'correct' : 'orange'
              }

    right = [numpy.nanmean([float(v) if str(v)!='na' else numpy.nan for v_i, v in enumerate(sub_data[s]['PAS_RT']) if sub_data[s]['accuracy'][v_i]=='correct']) for s in sub_data.keys()]
    wrong = [numpy.nanmean([float(v) if str(v)!='na' else numpy.nan for v_i, v in enumerate(sub_data[s]['PAS_RT']) if sub_data[s]['accuracy'][v_i]=='wrong']) for s in sub_data.keys()]
    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, right, 'correct'), (1, wrong, 'wrong')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'PAS RTs of correct vs wrong answers'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'PAS_RTs_correct_wrong.jpg'))
    pyplot.clf()
    pyplot.close()

### more complicated violinplot: accuracy per cat

    colors = {
              'low' : 'seagreen',
              'mid' : 'deepskyblue',
              'high' : 'hotpink',
              }

    lows = list()
    mids = list()
    highs = list()

    accuracies = {s : list() for s in sub_data.keys()}

    for s in sub_data.keys():
        low = numpy.nanmean([float(v) if str(v)!='na' else numpy.nan for v, pas in zip(sub_data[s]['PAS_RT'], sub_data[s]['PAS_score']) if pas=='1'])
        #try:
        #    low = len([v for v in low if v==1]) / len([v for v in low if v==0])
        #except ZeroDivisionError:
        #    low = 1.
        accuracies[s].append(low)
        lows.append(low)
        mid = numpy.nanmean([float(v) if str(v)!='na' else numpy.nan for v, pas in zip(sub_data[s]['PAS_RT'], sub_data[s]['PAS_score']) if pas=='2'])
        #try:
        #    mid = len([v for v in mid if v==1]) / len([v for v in mid if v==0])
        #except ZeroDivisionError:
        #    mid = 1.
        accuracies[s].append(mid)
        if str(mid) != 'nan':
            mids.append(mid)
        high = numpy.nanmean([float(v) if str(v)!='na' else numpy.nan for v, pas in zip(sub_data[s]['PAS_RT'], sub_data[s]['PAS_score']) if pas =='3'])
        #try:
        #    high = len([v for v in high if v==1]) / len([v for v in high if v==0])
        #except ZeroDivisionError:
        #    high = 1.
        accuracies[s].append(high)
        highs.append(high)


    fig, ax = pyplot.subplots(constrained_layout=True)
    for x, y, label in [(0, lows, 'low'), (1, mids, 'mid'), (2, highs, 'high')]:
        parts = ax.violinplot(y, positions=[x], showmeans=True, showextrema=False, )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[label])
            pc.set_edgecolor('grey')
        parts['cmeans'].set_facecolor('grey')
        parts['cmeans'].set_edgecolor('grey')
        ax.bar(0., 0., color=colors[label], label=label)
        ax.plot([x, x], [min(y), max(y)], color='grey')
        ax.plot([x-.05, x+.05], [min(y), min(y)], color='grey')
        ax.plot([x-.05, x+.05], [max(y), max(y)], color='grey')
    ax.legend()
    title = 'PAS RTs per pas category'
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, 'PAS RTs_accuracy_per_pas.jpg'))
    pyplot.clf()
    pyplot.close()


### writing to file

    with open(os.path.join(out_folder, '{}_trials_dprime_accuracies.tsv'.format(folder)), 'w') as o:
        o.write('subject\tdprime_low\tdprime_mid\tdprime_high\taccuracy_low\taccuracy_mid\taccuracy_high\n')
        for s in sorted(sub_data.keys()):
            o.write('{}\t'.format(s))
            ### dprimes
            for d in d_primes[s]:
                o.write('{}\t'.format(d))
            ### accuracies
            for a in accuracies[s]:
                o.write('{}\t'.format(a))
            o.write('\n')
