import sys
import os.path as op
import glob
import numpy as np

TITLE = sys.argv[1]
labels, col = np.loadtxt(op.join(TITLE,'Labels.tsv'), delimiter='\t', dtype='S20,S20', unpack=True)

data = []
for i,name in enumerate(labels):
    lst = glob.glob(op.join(TITLE, name.decode(), '*.png'))
    print('%10s: %d'%(name.decode(), len(lst)))
    for fn in lst:
        data.append([i, fn])

np.random.shuffle(data)
tt = int(len(data)*0.8)
np.savetxt(op.join(TITLE,'train.tsv'), data[:tt], fmt='%s', delimiter='\t')
np.savetxt(op.join(TITLE,'test.tsv'),  data[tt:], fmt='%s', delimiter='\t')