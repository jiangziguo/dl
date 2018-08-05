import numpy as np

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

X = np.array(X).astype(dtype='int16')
Y = np.array(Y).astype(dtype='int16')

lists = [X, Y]
print(lists)
ri = np.random.permutation(len(lists[1]))
print(ri)
out = []
for l in lists:
    print(l)
    out.append(l[ri])
print(out)
np.random.shuffle(lists)
print(lists)