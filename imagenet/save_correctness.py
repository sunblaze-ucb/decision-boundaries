import numpy as np

# usage: python save_correctness.py <modelname> <dataset>
import sys
modelname, dataset = sys.argv[1:]

count = 450

def load_seq(template, count):
    return np.asarray([np.load(template % j) for j in range(count)])

center_preds = load_seq('gxr3_%s/%s%%d_center_preds.npy' % (modelname, dataset), count).squeeze(1)
ground_truth = np.load('../imagenet/val_5k_labels.npy', 'r')[1285:1285+count]

correctness = np.equal(center_preds, ground_truth)
np.save('correctness_%s/%s.npy' % (modelname, dataset), correctness)
print np.count_nonzero(correctness), 'correct' # %%%
