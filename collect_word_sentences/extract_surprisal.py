import argparse
import random
import numpy
import os

from lang_models_utils import ContextualizedModelCard
from lang_models_utils import extract_surpr

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    type=str, 
                    required=True,
                     choices=[
                             'gpt2',
                             'gpt2-small',
                             'minervapt-350m',
                             'minervapt-1b',
                             'minervapt-3b',
                             'llama-1b',
                             'llama-3b',
                             'xglm-7.5b',
                             'xglm-564m',
                             'xglm-1.7b',
                             'xglm-2.9b',
                             'xglm-4.5b',
                             'xlm-roberta-large',
                             'xlm-roberta-xl',
                             'xlm-roberta-xxl',
                             ], 
                    help = 'Specifies where the vectors are stored')
parser.add_argument('--lang', 
                    type=str, 
                    required=True,
                     choices=[
                         'it', 'de', 'en'
                             ], 
                     )
parser.add_argument('--corpus', 
                    type=str, 
                    default='wac',
                     choices=[
                         'wac',
                             ], 
                     )
args = parser.parse_args()

### reading words per experiment
w_path = os.path.join('trials', args.lang)
cases = dict()
for f in os.listdir(w_path):
    if 'tsv' not in f:
        continue
    #if 'sound' not in f:
    #if  'kan' not in f and 'mitch' not in f and 'dir' not in f:
    #if 'distr' not in f and 'social' not in f and 'phon' not in f and 'prod' not in f and 'sound' not in f:
    #if 'anew' not in f and 'deaf' not in f and 'blind' not in f:
    #if 'distr' not in f and 'social' not in f:
    if 'phon' not in f and 'pmtg' not in f and 'behav' and 'sound' not in f:
        continue
    case = f.split('#')[0]
    cases[case] = set()
    ws = list()
    with open(os.path.join(w_path, f)) as i:
        for l in i:
            #print(l)
            w = [w.strip() for w in l.strip().split('\t')]
            if w[0].lower() == w[1].lower():
                continue
            if '' not in w:
                if args.lang == 'de':
                    cases[case].add('{} [SEP] {}'.format(w[0].capitalize(), w[1].capitalize()))
                else:
                    cases[case].add('{} [SEP] {}'.format(w[0], w[1]))
total = sum([len(v) for v in cases.values()])
print(total)

current_model = ContextualizedModelCard(args, causal=True)

entity_vectors = extract_surpr(
                               args, 
                               current_model,
                               cases,
                               )
out_f = os.path.join('llm_surprisals', args.lang, args.model)
os.makedirs(out_f, exist_ok=True)

print(current_model.n_layers, current_model.required_shape, )
for case, surprs in entity_vectors.items():
    with open(os.path.join(out_f, '{}_{}.tsv'.format(case, args.model)), 'w') as o:
        o.write('word_one\tword_two\tsurprisal_w_two\tentropy_w_one\n')
        for ws, surpr in surprs.items():
            w_one = ws.split()[0]
            w_two = ws.split(']')[1]
            o.write('{}\t{}\t{}\t{}\n'.format(w_one, w_two, surpr[0], surpr[1]))
