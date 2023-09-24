import collections
import itertools
import os
import numpy

from matplotlib import image, pyplot
from scipy import stats
from skimage import metrics
from tqdm import tqdm

word_dict = dict()

words = list()
with open(os.path.join('data', 'chosen_words.txt')) as stimuli_file:
    for i, l in enumerate(stimuli_file):
        if i > 0: 
            l = l.strip().split('\t')
            words.append(l[0])

os.makedirs('stimuli_images')

print('Now collecting images arrays...')
for w in words:
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax = pyplot.gca()
    p = pyplot.Rectangle((left, bottom), width, height, alpha=0., fill=False)
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

    ax.text(0.5 * (left + right), 0.5 * (bottom + top), w,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, fontsize='xx-large')

    pyplot.axis('off')
    pyplot.savefig('stimuli_images/{}.jpg'.format(w))
    pyplot.clf()
    im = image.imread('stimuli_images/{}.jpg'.format(w))
    word_dict[w] = im

print('Now computing pairwise similarities...')
combs = [tuple(sorted(k)) for k in itertools.combinations(words, 2)]
sims = dict()
for c in tqdm(combs):
    im = image.imread('stimuli_images/{}.jpg'.format(c[0]))
    im_2 = image.imread('stimuli_images/{}.jpg'.format(c[1]))

    total = 0
    same = 0

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j].tolist() == im_2[i,j].tolist():
                same += 1
            total += 1

    sims[c] = [same/total, metrics.structural_similarity(im, im_2, multichannel=True)]

print('Now unit normalizing...')
norm_sims = collections.defaultdict(list)
for i in range(2):
    
    std = numpy.nanstd([v[i] for k, v in sims.items()])
    mean = numpy.nanmean([v[i] for k, v in sims.items()])
    #l1_norm = sum([v[i] for k, v in sims.items()])
    #max_value = max([v[i] for k, v in sims.items()])
    #min_value = min([v[i] for k, v in sims.items()])
    for c, res in sims.items():
        #norm_sims[c].append(res[i]/l1_norm)
        norm_sims[c].append((res[i]-mean)/std)
        #norm_sims[c].append((res[i]-min_value)/(max_value-min_value))

#print(stats.pearsonr([k[0] for i, k in norm_sims.items()], [k[1] for i, k in norm_sims.items()]))
os.system('rm -r stimuli_images')
out_folder = 'similarities'
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, 'visual_similarities.tsv'), 'w') as o:
    o.write('Word 1\tWord 2\tPixel overlap\tStructural similarity\n')
    for c, res in norm_sims.items():
        o.write('{}\t{}\t{}\t{}\n'.format(c[0], c[1], res[0], res[1]))
