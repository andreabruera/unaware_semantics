import numpy
import os

### reading categories
cat_mapper = {
          'animal' : -1,
          'object' : 1,
          }
cats = dict()
en_to_it = dict()
with open(os.path.join('data', 'chosen_words.txt')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        cats[line[0]] = cat_mapper[line[-1]]
        en_to_it[line[1]] = line[0]

ratings = {
           'aoa' : {w : list() for w in en_to_it.values()},
           'concreteness' : {w : list() for w in en_to_it.values()},
           'vision' : {w : list() for w in en_to_it.values()},
           'smell' : {w : list() for w in en_to_it.values()},
           'touch' : {w : list() for w in en_to_it.values()},
           'hearing' : {w : list() for w in en_to_it.values()},
           'taste' : {w : list() for w in en_to_it.values()},
           }

### reading dataset conc
with open(os.path.join('data', 'brysbaert_conc.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        idx = header.index('Conc.M')
        ratings['concreteness'][en_to_it[word]].append(float(line[idx]))

### reading dataset aoa
with open(os.path.join('data', 'kuperman_aoa.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        idx = header.index('Rating.Mean')
        ratings['aoa'][en_to_it[word]].append(float(line[idx]))


### reading dataset #1
mapper = {
          'Visual.mean' : 'vision',
          'Olfactory.mean' : 'smell',
          'Haptic.mean' : 'touch',
          'Gustatory.mean' : 'taste',
          'Auditory.mean' : 'hearing',
          }
with open(os.path.join('data', 'Lancaster_sensorimotor_norms_for_39707_words.tsv')) as i:
    counter = 0
    for l in i:
        line = l.strip().replace(',', '.').split('\t')
        if counter == 0:
            header = [w.strip() for w in line]
            counter += 1
            continue
        word = line[0].lower()
        if word not in en_to_it.keys():
            continue
        for k, dest in mapper.items():
            idx = header.index(k)
            ratings[dest][en_to_it[word]].append(float(line[idx]))

for k, v in ratings.items():
    for w, w_v in v.items():
        assert len(w_v) >= 1
        if len(w_v) > 1:
            print(w_v)

### writing to file
with open(os.path.join('data', 'word_norms.tsv'), 'w') as o:
    o.write('word\tword_length\tsemantic_category\taoa\tconcreteness\tvision\tsmell\ttouch\ttaste\thearing\n')
    for w, c in cats.items():
        o.write('{}\t{}\t{}\t'.format(w, len(w), c))
        for cat in ['aoa', 'concreteness', 'vision', 'smell', 'touch', 'taste', 'hearing']:
            o.write('{}\t'.format(ratings[cat][w][0]))
        o.write('\n')
