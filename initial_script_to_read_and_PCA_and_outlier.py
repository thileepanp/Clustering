#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:22:59 2018

@author: thileepan
"""

"""
Created on Fri Aug 24 14:36:59 2018

@author: thileepan
"""


def Read_Librosa_Output_Data_and_PCA_and_Outlier_removal(x1='2017_04/Librosa_2017_*', x2 = '2017_05/Librosa_2017_*'):
    from natsort import natsorted
    import pandas as pd
    import glob
    import os
    from sklearn import decomposition
    from sklearn.cluster import KMeans
    from sklearn.cluster import MeanShift
    from soundapi import SoundAPI
    import numpy as np
    from plotCluster import plotClusters
    import hdbscan
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import svm
    import os
    
    os.chdir('/media/thileepan/Librosa_Fea/')
    file_list = []
    
    for file in glob.glob(x1):
        file_list.append(file)
        file_list = natsorted(file_list)
        
    apr_17 = pd.DataFrame()
    for file in file_list:
        temp_file = pd.read_hdf(file)
        apr_17 = apr_17.append(temp_file)
        
    file_list = []

    for file in glob.glob(x2):
        file_list.append(file)
        file_list = natsorted(file_list)
        
    may_17 = pd.DataFrame()
    for file in file_list:
        temp_file = pd.read_hdf(file)
        may_17 = may_17.append(temp_file)
    
    apr_may_17 = apr_17.iloc[:-60,:].append(may_17)
    data = apr_may_17
    
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    t = pca.transform(data)
    plt.scatter(t[:,0], t[:,1], s= 1)
    
    mask = (t[:,0]<20) & (t[:,1]<40)
    #plt.figure()
    #plt.scatter(t[mask][:,0], t[mask][:,1])
    
    pca = decomposition.PCA(n_components=2)
    pca.fit(data[mask])
    t1 = pca.transform(data[mask])
    plt.figure()
    plt.scatter(t1[:,0], t1[:,1], s=0.01)
    
    ri=np.random.choice(range(len(data[mask])),size=1000)
    ms=MeanShift()
    msc = ms.fit(t1[ri,0:3])
    data.loc[mask,'msc_cluster'] = msc.predict(t1[:,0:3])
    
    SA= SoundAPI()
    DIRECTORY = '/home/thileepan/data3/auris3/hdf5'
    PREFIX = 'ristilantie_'
    SA.fastScan(DIRECTORY, PREFIX)
    
    plotClusters(data[mask], pc=t, clusterfieldname='msc_cluster', SA=SA)