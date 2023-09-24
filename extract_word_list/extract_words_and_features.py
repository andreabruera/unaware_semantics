import os
import numpy
import itertools
import math

from matplotlib import pyplot
from scipy import stats

from orthographic_measures import coltheart_N, OLD_twenty

### Reading the Glasgow Norms

glasgow_path = os.path.join('resources', 'glasgow_norms.csv')
with open(glasgow_path, encoding='utf-8') as i:
    lines = [l.strip().split(',') for l in i.readlines()]
header = lines[0]
relevant_variables = ['Words', 'FAM', 'IMAG']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_variables]
values = [[l[i] for i in indices] for l in lines[2:]]
en_fam_imag = {l[0] : (float(l[1]), float(l[2])) for l in values}

### Reading the Italian norms

lexvar_path = os.path.join('resources', 'lexvar.csv')
with open(lexvar_path, encoding='utf-8') as i:
    lines = [l.strip().split(',') for l in i.readlines()]
header = [l.lstrip() for l in lines[0]]
relevant_variables = ['WORD', 'FAM', 'IMAG']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_variables]
values = [[l[i] for i in indices] for l in lines[2:]]
it_fam_imag = {l[0].lower() : (float(l[1]), float(l[2])) for l in values}

### Reading the frequencies from itwac

absolute_frequencies = os.path.join('resources', 'itwac_absolute_frequencies_50k.txt')
with open(absolute_frequencies, encoding='utf-8') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]
    
### Transforming the frequencies by their logarithm

### Correcting words to their recorded version in ItWac
vocabulary_mapping = {'maglione' : 'pullover', \
                      'passerotto' : 'passero'}

freqs = {l[1] if l[1] not in vocabulary_mapping.keys() else vocabulary_mapping[l[1]] : math.log(int(l[2])) \
                                                                                             for l in lines}

### Opening the Kremer & Baroni word list
kremer_baroni_path = os.path.join('resources', 'concept-measures_it.txt')
with open(kremer_baroni_path, encoding='utf-8', errors='ignore') as i:
    lines = [l.lower().strip().split('\t') for l in i.readlines()]

headers = lines[0]

### Excluding non-animal living things
excluded = ['body_part', 'vegetable', 'fruit', 'building', 'vehicle', 'clothing']
excluded = ['body_part', 'vegetable', 'fruit', 'building', 'vehicle']
data = [l for l in lines[1:] if l[1] not in excluded \
                                and l[3] in freqs.keys()\
                                and l[3] != 'oca' \
                                and l[3] != 'com' \
                                and l[3] != 'pullover' \
                                and l[3] != 'pinza' \
                                and l[3] != 'teglia']

### Selecting the relevant indices: category, word, length, log frequency
relevant_indices = [1,\
                    3, \
                    #5, \
                    6, \
                    ]
headers_mapping = {'ConceptClass'.lower() : 'category', \
                   'ConceptLemma'.lower() : 'word', \
                   'No.Letters'.lower() : 'length', \
                   'logFreqWaCKy'.lower() : 'log_frequency', \
                   }

### Selecting the natural categories
animal_cats = ['mammal', 'bird']

### Putting all words and variables into a dictionary
head_dict = dict()

for h_i, h in enumerate(headers):
    if h_i in relevant_indices:

        ### Category and word
        if h_i <= 3:
            heading_data = [d[h_i].lower() for d in data]

               
            ### Correcting categories and word mentions
            if h_i == 1:
                heading_data = ['object' if w not in animal_cats else 'animal' \
                                                         for w in heading_data]
            if h_i == 3:
                heading_data = [w if w not in vocabulary_mapping.keys() else \
                                    vocabulary_mapping[w] for w in heading_data]

                ### Computing coltheart's N and OLD20 for the words
                colt = coltheart_N(heading_data)
                colt_data = [colt[w] for w in heading_data]
                old = OLD_twenty(heading_data)
                old_data = [old[w] for w in heading_data]
                ### Getting the frequencies from ItWac
                freq_data = [math.log(freqs[w]) for w in heading_data]
                head_dict['coltheart_N'] = colt_data
                head_dict['OLD20'] = old_data
                head_dict['log_frequency'] = freq_data

                ### Collecting familiarity and imageability ratings for Glasgow norms
                en_fam_imag_data = [d[0].lower() for d in data]
                it_to_en = {w : en_fam_imag_data[w_i] for w_i, w in enumerate(heading_data)}
                en_missing_words = [w for w in en_fam_imag_data if w not in en_fam_imag.keys()]
                #print('missing words: {}'.format(missing_words))
                baroni_en_fam_data = numpy.array([en_fam_imag[w][0] if w in en_fam_imag.keys() else numpy.nan \
                                                             for w in en_fam_imag_data])
                baroni_en_imag_data = numpy.array([en_fam_imag[w][1] if w in en_fam_imag.keys() else numpy.nan \
                                                               for w in en_fam_imag_data])

                ### Collecting familiarity and imageability ratings for Italian norms
                it_fam_imag_data = heading_data
                it_missing_words = [w for w in it_fam_imag_data if w not in it_fam_imag.keys()]
                #print('missing words: {}'.format(missing_words))
                baroni_it_fam_data = numpy.array([it_fam_imag[w][0] if w in it_fam_imag.keys() else numpy.nan \
                                                             for w in it_fam_imag_data])
                baroni_it_imag_data = numpy.array([it_fam_imag[w][1] if w in it_fam_imag.keys() else numpy.nan \
                                                               for w in it_fam_imag_data])
            
        ### Length and frequency
        else:
            heading_data = [float(d[h_i]) for d in data]

        h = headers_mapping[h]
        head_dict[h] = heading_data

#head_dict['domain'] = ['Artifacts' if d[1] not in natural else 'Natural' for d in data]

### Now reading the Montefinese et al. 2013 word list
montefinese_path = os.path.join('resources', '13428_2012_291_MOESM3_ESM.txt')
with open(montefinese_path, encoding='utf-8' errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
headers = lines[0]

#excluded = ['plants', 'body_parts', 'housing_buildings', 'clothes', 'vehicles']
excluded = ['plants', 'body_parts', 'housing_buildings', 'vehicles']
### Removing words that are:
    # infrequent 
    # absent from ItWac
    # absent from the previous dictionary
data = [l for l in lines[1:] if \
              float(l[8])>0. and \
              l[1] not in excluded and \
              l[2] in freqs.keys() and \
              l[2] != 'teglia' and \
              l[2] != 'oca' and \
              l[2] != 'com' and \
              l[2] != 'pullover' and \
              l[2] != 'pinza' and \
              l[2] not in head_dict['word']]

#relevant_indices = [0, \
relevant_indices = [1, \
                    2, \
                    4, \
                    #8, \
                    #9, \
                    ]

headers_mapping = {'CATEGORY' : 'category', \
                   'CONCEPT (IT)' : 'word', \
                   'Length (IT)' : 'length', \
                   'LN_Word_Fr' : 'log_frequency', \
                   #'Familiarity_Rating' : 'familiarity'},\
                   #'DOMAIN' : 'domain',\
                   }
                   
for h_i, h in enumerate(headers):

    if h_i in relevant_indices:
        
        ### Word and category
        if h_i <= 2:
            heading_data = [d[h_i] for d in data]
            ### Turning categories into animal/object
            if h_i == 1:
                heading_data = ['animal' if w=='animals' else 'object' \
                                                  for w in heading_data]
            elif h_i == 2:
                ### Computing coltheart's N and OLD20 for the words
                colt = coltheart_N(heading_data)
                colt_data = [colt[w] for w in heading_data]
                old = OLD_twenty(heading_data)
                old_data = [old[w] for w in heading_data]
                head_dict['coltheart_N'].extend(colt_data)
                head_dict['OLD20'].extend(old_data)
                ### Getting the frequencies from ItWac
                freq_data = [math.log(freqs[w]) for w in heading_data]
                head_dict['log_frequency'].extend(freq_data)

                ### Collecting familiarity and imageability ratings for Glasgow norms
                en_fam_imag_data = [d[3].lower() for d in data]
                it_to_en.update({w : en_fam_imag_data[w_i] for w_i, w in enumerate(heading_data)})
                en_missing_words.extend([w for w in en_fam_imag_data if w not in en_fam_imag.keys()])
                en_fam_data = numpy.array([en_fam_imag[w][0] if w in en_fam_imag.keys() else 0 \
                                                             for w in en_fam_imag_data])
                en_imag_data = numpy.array([en_fam_imag[w][1] if w in en_fam_imag.keys() else 0 \
                                                               for w in en_fam_imag_data])
                head_dict['en_familiarity'] = numpy.concatenate((baroni_en_fam_data, \
                                                                            en_fam_data), axis=0)
                head_dict['en_imageability'] = numpy.concatenate((baroni_en_fam_data, \
                                                                           en_imag_data), axis=0)

                ### Collecting familiarity and imageability ratings for Italian norms
                #fam_imag_data = [d[3].lower() for d in data]
                it_fam_imag_data = heading_data
                it_missing_words.extend([w for w in it_fam_imag_data if w not in it_fam_imag.keys()])
                it_fam_data = [it_fam_imag[w][0] if w in it_fam_imag.keys() else 0 \
                                                             for w in it_fam_imag_data]
                it_imag_data = [it_fam_imag[w][1] if w in it_fam_imag.keys() else 0 \
                                                               for w in it_fam_imag_data]
                head_dict['it_familiarity'] = numpy.concatenate((baroni_it_fam_data, \
                                                                             it_fam_data), axis=0)
                head_dict['it_imageability'] = numpy.concatenate((baroni_it_fam_data, \
                                                                            it_imag_data), axis=0)

        ### Length and frequency
        else:
            heading_data = [float(d[h_i]) for d in data]

        h = headers_mapping[h]
        ### Extending the new 
        head_dict[h].extend(heading_data)

assert len(set([len(v) for k, v in head_dict.items()])) == 1

number_words = list(set([len(v) for k, v in head_dict.items()]))[0]

animal_indices = [w_i for w_i, w in enumerate(head_dict['category']) if w=='animal']
assert len(animal_indices) == 16

animal_averages = {k : numpy.nanmedian([v[i] for i in animal_indices]) for k, v in head_dict.items() if k not in ['word', 'category']}
animal_l = [head_dict['length'][i] for i in animal_indices]
object_indices = [i for i in range(number_words) if i not in animal_indices]

object_dict = {k : [v[i] for i in object_indices] for k, v in head_dict.items()}

matched_lengths = list()
for k in sorted(animal_l):

    if len(matched_lengths) == 16:
        break

    marker = False

    for k_two_i, k_two in enumerate(object_dict['length']):
        if k_two == k and marker == False:
            if k_two_i not in matched_lengths:
                matched_lengths.append(k_two_i)
                marker = True
    if marker == False:
        new_k = k-1
        for k_two_i, k_two in enumerate(object_dict['length']):
            if k_two == new_k and k_two <= max(animal_l)\
                and marker == False:
                if k_two_i not in matched_lengths:
                    matched_lengths.append(k_two_i)
                    marker = True

    if marker == False:
        new_k = k+1
        for k_two_i, k_two in enumerate(object_dict['length']):
            if k_two == new_k and k_two <= max(animal_l)\
                and marker == False:
                if k_two_i not in matched_lengths:
                    matched_lengths.append(k_two_i)
                    marker = True

chosen_words = [object_dict['word'][i] for i in matched_lengths]
        
'''
import pdb; pdb.set_trace()

object_counter = {w : [0 for i in range(len(animal_averages.keys()))] for w in object_dict['word']}

for score_i, score_data in enumerate(animal_averages.items()):

    score = score_data[0]
    score_avg = score_data[1]

    score_list = list()
    
    for w_i, w in enumerate(object_dict['word']):

        word_value = object_dict[score][w_i]
        diff = (word_value-score_avg)**2
        score_list.append(diff)
    
    sorted_indices = [k for k, v in sorted(enumerate(score_list), key=lambda item : item[1], reverse=True)]
    for s_i, s in enumerate(sorted_indices):
        word = object_dict['word'][s]
        if score == 'length':
            s_i = s_i ** 2
        object_counter[word][score_i] += s_i

final_scores = [w for w, s in sorted([(k, sum(v)) for k, v in object_counter.items()], key=lambda item : item[1], reverse=True)]
chosen_words = final_scores[:16]
''' 
object_indices = [w_i for w_i, w in enumerate(head_dict['word']) if w in chosen_words]
object_l = [head_dict['length'][i] for i in object_indices]

fig, ax = pyplot.subplots()
ax.hist([animal_l, object_l], label=['animals', 'objects'])
pyplot.savefig('plots/histogram.png')
pyplot.clf()

### Writing the list to file

with open('chosen_words.txt', 'w') as o: 
    o.write('Word\tCategory\n')

    for index in animal_indices:
        animal = head_dict['word'][index]
        english = it_to_en[animal]
        if animal == 'passerotto':
            animal = passero
        o.write('{}\t{}\tanimal\n'.format(animal, english))
        
    for index in object_indices:
        current_object = head_dict['word'][index]
        english = it_to_en[current_object]
        o.write('{}\t{}\tobject\n'.format(current_object, english))

### Plotting
for k in animal_averages.keys():

    animal_data = [head_dict[k][i] for i in animal_indices]
    object_data = [head_dict[k][i] for i in object_indices]
    data = [[k for k in animal_data if k>0], [k for k in object_data if k>0]]
    labels = ['animals', 'objects']

    fig, ax = pyplot.subplots(figsize=(12,5))
    ax.boxplot(data, labels=labels)
    ax.set_title('Measures of {} for the selected words'.format(k))
    os.makedirs('plots', exist_ok=True)
    pyplot.savefig('plots/{}.png'.format(k))

### Extracting the features

### Properties file
montefinese_properties_path = os.path.join('resources', '13428_2012_291_MOESM5_ESM.txt') 
with open(montefinese_properties_path, encoding='utf-8', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
relevant_columns = ['CONCEPT (IT)', 'FEATURE (IT)', 'Prod_Fr', 'Distinctiveness']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
data = [[l[i] for i in indices] for l in lines[1:]]

words = list(set([d[0] for d in data]))
feature_dict = {w : list() for w in words}

for w in words:
    for d in data:
        if d[0] == w:
            feature_dict[w].append([d[1], float(d[2])*float(d[3])])

### Reading experiment words

with open('chosen_words.txt') as i:
    lines = [l.strip().split('\t') for l in i.readlines()][1:]

words = [l[0] for l in lines]
exp_features = dict()
missing_words = list()

for w in words:
    if w in feature_dict.keys():
        exp_features[w] = feature_dict[w]
    else:
        missing_words.append(w)

### Reading the other features

baroni_kremer_features_path = os.path.join('resources', 'concepts-features_it.txt')
with open(baroni_kremer_features_path, encoding='utf-8', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
relevant_columns = ['Concept(IT)', 'Prod.Frequency', 'Feature', 'Distinctiveness']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
data = [[l[i] for i in indices] for l in lines[1:]]

### Reading the translator from English to Italian
eng_to_it = os.path.join('resources', 'production-data_it.txt')
with open(eng_to_it, encoding='utf-8', errors='ignore') as i:
    lines = [l.strip().split('\t') for l in i.readlines()]
header = lines[0]
lines = [[w for w in l if w != ''] for l in lines] #correcting a bug in the dataset
relevant_columns = ['Feature', 'Phrase']
indices = [w_i for w_i, w in enumerate(header) if w in relevant_columns]
assert len(indices) == len(relevant_columns)
translator = {l[indices[0]] : l[indices[1]] for l in lines}

for w in missing_words:
    assert w not in feature_dict.keys()
    feature_dict[w] = list()
    for d in data:
        if d[0] == w:
            feature_dict[w].append([translator[d[2]], float(d[1])*float(d[3])])

for w in missing_words:
    if w in feature_dict.keys():
        exp_features[w] = feature_dict[w]
    else:
        print('problem with {}'.format(w))

exp_features = {k : sorted(v, key=lambda item : item[1], reverse=True) for k, v in exp_features.items()}
exp_features = {k : [val[0] for val in v] for k, v in exp_features.items()}

final_features = {k : list() for k in exp_features.keys()}

for k, v in exp_features.items():
    for feat in v:
        counter = 0
        for k_two, v_two in exp_features.items():
            if k_two != k:
                if feat in v_two:
                    counter += 1
        if counter < 2:
            final_features[k].append(feat)

with open('chosen_features.txt', 'w') as o:
    o.write('Word\tFeatures\n')
    for w in words:
        o.write('{}\t'.format(w))
        for feat in exp_features[w]:
            o.write('{}\t'.format(feat.replace(' ', '_')))
        o.write('\n')
