#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:11:08 2018

@author: tpaulraj
"""

from natsort import natsorted
import pandas as pd
import glob
import os
import pandas as pd
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from soundapi import SoundAPI
#import SoundAPI
import numpy as np
from plotCluster import plotClusters
from hdbscan import HDBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import os

os.chdir('/Volumes/Librosa_Fea/features/')

file_list = []
for file in glob.glob('Librosa_*'):
    file_list.append(file)
    file_list = natsorted(file_list)
    
#len(file_list)
#file_list
    
full_year = pd.DataFrame()
for file in file_list:
    temp_file = pd.read_hdf(file)
    full_year = full_year.append(temp_file)
    
#full_year.shape
#full_year.head()
#full_year.tail()
    
data = full_year

pca = decomposition.PCA(n_components=2)
pca.fit(data)
t = pca.transform(data)
plt.figure()
plt.scatter(t[:,0], t[:,1])

mask = plt.mlab.find((t[:,0]<20) & (t[:,1]<40))
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

mask = plt.mlab.find((t[:,0]<40000) & (t[:,1]<6000))
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

mask = plt.mlab.find((t[:,0]<5000) & (t[:,1]<6000))
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

mask = plt.mlab.find((t[:,0]<1000) & (t[:,1]<6000))
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

pca.fit(data.iloc[mask])
t = pca.transform(data.iloc[mask])
plt.figure()
plt.scatter(t[:,0], t[:,1], s=0.001)


ri=np.random.choice(range(len(data.iloc[mask])),size=np.int(data.shape[0]/30))
#ri_test = np.random.choice(range(len(data.iloc[mask])),size=np.int(data.shape[0]/10))
ri_test = np.random.choice(range(len(data.iloc[mask])),size=data.shape[0])


clusterer = HDBSCAN(min_cluster_size=500, gen_min_span_tree=True)
hdb = clusterer.fit(t[ri,0:3])
data.ix[mask[ri_test],'hdbscan_cluster'] = hdb.fit_predict(t[ri_test, 0:3])

SA= SoundAPI()
#DIRECTORY = '/home/thileepan/data3/auris3/hdf5' #Linux
DIRECTORY = '/Users/thileepan/data3/auris3/hdf5' #MACOS
PREFIX = 'ristilantie_'
SA.fastScan(DIRECTORY, PREFIX)

plotClusters(data.ix[mask[ri_test]], pc=t[ri_test], clusterfieldname='hdbscan_cluster', SA=SA, pointsize = 0.0001)

