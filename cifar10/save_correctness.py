import numpy as np

# usage: python save_correctness.py <modelname> <step> <dataset>
import sys
modelname, step, dataset = sys.argv[1:]
step = int(step)

count = 200

def load_seq(template, count):
    return np.asarray([np.load(template % j) for j in range(count)])

center_preds = load_seq('gxr3_%s/step%d_%s%%d_center_preds.npy' % (modelname, step, dataset), count).squeeze(1)
ground_truth = np.load('test5415_labels.npy', 'r')[:count]

correctness = np.equal(center_preds, ground_truth)
np.save('correctness_%s/step%d_%s.npy' % (modelname, step, dataset), correctness)
print np.count_nonzero(correctness), 'correct' # %%%
