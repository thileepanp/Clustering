l#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:11:19 2018

@author: thileepan
"""

import pandas as pd
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import numpy as np
from plotCluster import plotClusters
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import os

#reading data
os.chdir('/media/thileepan/Librosa_Fea/')
data = pd.read_csv('apr_may_17.csv', index_col=0)
#data= pd.read_csv('yes_outlier_am_plus_librosa_22_to_28_apr.csv', index_col=0, parse_dates=[0])

#PCA
pca = decomposition.PCA(n_components=2)
pca.fit(data)
t = pca.transform(data)

#kmeans clustering
kmeans = KMeans(n_clusters=10, n_jobs=-1)
kmeans.fit(t[:,0:3])
data['cluster'] = kmeans.predict(t[:,0:3])

#hdbscan clustering
clusterer = hdbscan.HDBSCAN().fit(data)
color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

clusterer.condensed_tree_.plot()
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))
clusterer.condensed_tree_.plot()
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

clusterer.fit(data)
clusterer.fit(t[:,0:3])
clusterer.labels_.max()
data['cluster'] = clusterer.predict(t[:,0:3])

SA= SoundAPI()

DIRECTORY = '/home/thileepan/data3/auris3/hdf5'
PREFIX = 'ristilantie_'
SA.fastScan(DIRECTORY, PREFIX)

plotClusters(data, pc=t, clusterfieldname='cluster', SA=SA)




ri=np.random.choice(range(len(data)),size=1000)
ri=np.random.choice(range(len(data)),size=1000)
ms=MeanShift()
msc = ms.fit(t[ri,0:3])
data['msc_cluster'] = msc.predict(t[:,0:3])
plotClusters(data, pc=t, clusterfieldname='msc_cluster', SA=SA)

