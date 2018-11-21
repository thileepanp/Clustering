#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:57:27 2018

@author: thileepan
"""

import hypertools as hyp
import numpy as np
import scipy
import pandas as pd
from scipy.linalg import toeplitz
from copy import copy
import os
import seaborn as sb

os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/Librosa')
#os.chdir('/Users/thileepan/Dropbox/PhD/2018_work/feature_extraction/Librosa/') #MAC

data = pd.read_csv('yes_outlier_am_plus_librosa_22_to_28_apr.csv', index_col=0)

geo= hyp.plot(data, '.')
geo=hyp.plot(data, 'b*')

geo=hyp.plot(data, '.', ndims=2)

geo = hyp.plot(data, '.', reduce='SparsePCA')
geo = hyp.plot(data, '.', reduce='IncrementalPCA')
geo = hyp.plot(data, '.', reduce = 'MiniBatchSparsePCA') 
geo = hyp.plot(data, '.', reduce = 'KernelPCA') #not working
geo = hyp.plot(data, '.', reduce = 'FastICA')
geo = hyp.plot(data, '.', reduce = 'FactorAnalysis')
geo = hyp.plot(data, '.', reduce = 'TruncatedSVD')#same results like PCA
geo = hyp.plot(data, '.', reduce = 'DictionaryLearning') #took a long time to run
geo = hyp.plot(data, '.', reduce = 'MiniBatchDictionaryLearning')
geo = hyp.plot(data, '.', reduce = 'TSNE') #takes long time to run
geo = hyp.plot(data, '.', reduce = 'Isomap') #memory error
geo = hyp.plot(data, '.', reduce = 'SpectralEmbedding') #system hangs
geo = hyp.plot(data, '.', reduce = 'LocallyLinearEmbedding')
geo = hyp.plot(data, '.', reduce = 'MDS') #memory error


geo= hyp.plot(data, '.', reduce={'model': 'PCA', 'params': {'whiten': True} }) 

training_set=data.iloc[np.random.choice(len(data), 10000),:]
birch=hyp.cluster(training_set, cluster='Birch')
all_birch=birch.apply(data)

geo_cluster = hyp.plot(training_set, '.', cluster = 'HDBSCAN', n_clusters=6)

#Clustering

geo_cluster = hyp.plot(data, '.', n_clusters = 6)
geo_cluster = hyp.plot(data, '.', cluster = 'KMeans', n_clusters = 8)
geo_cluster = hyp.plot(data, '.', cluster = 'MiniBatchKMeans', n_clusters = 8)
geo_cluster = hyp.plot(data, '.', cluster = 'AgglomerativeClustering', n_clusters = 8) #memory error
geo_cluster = hyp.plot(data, '.', cluster = 'Birch', n_clusters = 8) #works only igeo_cluster = hyp.plot(data, '.', cluster = 'SpectralClustering', n_clusters = 6) #takes long
geo_cluster = hyp.plot(data, '.', cluster = 'HDBSCAN', n_clusters=6) #has a different cluster

labels= hyp.cluster(data)
set(labels)

labels_HDBSCAN = hyp.cluster(data, cluster='HDBSCAN')
geo = hyp.plot(data, '.', hue=labels_HDBSCAN, title='HCBSCAN clustering')

geo_cluster = hyp.plot(training_set, '.', n_clusters = 6)
geo_kmeans = hyp.plot(training_set, '.', cluster = 'KMeans', n_clusters = 8)
geo_minibatchkmeans = hyp.plot(training_set, '.', cluster = 'MiniBatchKMeans', n_clusters = 8)
geo_agglomerative = hyp.plot(data.iloc[np.random.choice(len(data), 10000),:], '.', cluster = 'AgglomerativeClustering', n_clusters = 8)
geo_birch = hyp.plot(data.iloc[np.random.choice(len(data), 10000),:], '.', cluster='Birch', n_clusters=8)
geo_featureagglome = hyp.plot(training_set, '.', cluster = 'FeatureAgglomeration', n_clusters = 2) #don't know
geo_spectralclustering = hyp.plot(training_set, '.', cluster = 'SpectralClustering', n_clusters = 6)
geo_dbscan = hyp.plot(data.iloc[np.random.choice(len(data), 10000),:], '.', cluster = 'HDBSCAN', n_clusters=8) #has a different cluster
geo_dbscan = hyp.plot(data.iloc[np.random.choice(len(data), 10000),:], '.', cluster = 'HDBSCAN', n_clusters=8, ndims=2) #has a different cluster
geo_dbscan = hyp.plot(data.iloc[np.random.choice(len(data), 30000),:], cluster = 'HDBSCAN', animate='parallel', duration=30.00, explore=True, '.') #trying different methods