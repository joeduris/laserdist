# -*- coding: iso-8859-1 -*-
 
import numpy as np
import matplotlib.pyplot as plt
import time

filename = 'gaus_flat_triangle.csv'
filename = 'fd3_gaus2square_Thu_21_Nov_2019_15-44-46.csv'

t0 = time.time()
ps = np.genfromtxt(filename, delimiter=',')
print('load csv:',time.time()-t0)

t0 = time.time()
ps = ps[np.any(np.isnan(ps),axis=1)==False] # cut nans (descriptions)
print('nan cut:',time.time()-t0)

t0 = time.time()
ps = ps[ps[:,1].argsort()]
print('sort 1:',time.time()-t0)

t0 = time.time()
ps = ps[ps[:,0].argsort()]
print('sort 0:',time.time()-t0)

filenameroot = ''.join(filename.split('.')[:-1])

t0 = time.time()
np.save(filenameroot+'.npy',ps)
print('save npy:',time.time()-t0)

t0 = time.time()
np.load(filenameroot+'.npy')
print('load npy:',time.time()-t0)

unique_fs = np.unique(ps[:,0])
unique_ts = np.unique(ps[:,1])

minf = min(unique_fs); maxf = max(unique_fs)
print(min(unique_fs), max(unique_fs))
