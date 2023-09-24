import numpy
import collections

from tqdm import tqdm

def read_vocabulary():
    with open('itwac_absolute_frequencies_50k.txt') as input_file:
        lines = [l.strip().split('\t')[1] for l in input_file.readlines()][1:]
    words = [l for l in lines if l.isalpha()][:35000]
    return words

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def coltheart_N(words):

    counter = collections.defaultdict(int)
    vocabulary = read_vocabulary()
    for w in tqdm(words):
        for w_two in vocabulary:
            if w != w_two and len(w) == len(w_two):
                dist = levenshtein(w, w_two)
                if dist == 1:
                    counter[w] += 1
        if w not in counter.keys():
            counter[w] = 0

    '''
    mean = numpy.average([v for k, v in counter.items()])
    std = numpy.std([v for k, v in counter.items()])
    z_counter = {k : (v-mean)/std for k, v in counter.items()}
    '''

    return counter    

def OLD_twenty(words):

    counter = collections.defaultdict(int)
    vocabulary = read_vocabulary()
    for w in tqdm(words):
        w_list = list()
        for w_two in vocabulary:
            if w != w_two:
                dist = levenshtein(w, w_two)
                w_list.append(dist)
        w_list = sorted(w_list)
        w_average = numpy.average(w_list[:20])
        counter[w] = w_average

    mean = numpy.average([v for k, v in counter.items()])
    std = numpy.std([v for k, v in counter.items()])
    z_counter = {k : (v-mean)/std for k, v in counter.items()}
    
    return z_counter
