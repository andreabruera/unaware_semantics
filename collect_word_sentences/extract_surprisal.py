import argparse
import random
import numpy
import os

from lang_models_utils import ContextualizedModelCard
from lang_models_utils import extract_surpr, read_all_sentences

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

replication_sentences = read_all_sentences(args)

sentences = ['{} [SEP] {}'.format(w_one, w_two) for w_one in replication_sentences.keys() for w_two in replication_sentences.keys() if w_one!=w_two]

current_model = ContextualizedModelCard(args, causal=True)

entity_vectors = extract_surpr(
                               args, 
                               current_model,
                               sentences,
                               )
out_f = os.path.join('..', 'similarities')
os.makedirs(out_f, exist_ok=True)

print(current_model.n_layers, current_model.required_shape, )
with open(os.path.join(out_f, '{}-{}-surprisal_similarities.tsv'.format(args.model, args.lang)), 'w') as o:
    o.write('word_one\tword_two\tavg_surprisals\tavg_surprisals\n')
    for ws, surpr in entity_vectors.items():
        o.write('{}\t{}\t{}\t{}\n'.format(ws[0], ws[1], numpy.average(surpr), numpy.average(surpr)))
