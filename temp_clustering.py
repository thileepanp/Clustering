#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:36:59 2018

@author: thileepan
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
from importlib import reload


os.chdir('/home/thileepan/Projects/Clustering/features/')# Linux_Latest
os.chdir('/media/thileepan/Librosa_Fea/features/') #Linux
os.chdir('/media/tpaulraj/Librosa_Fea/')
#os.chdir('/Volumes/Librosa_Fea') #MACOS
file_list = []
for file in glob.glob('Librosa_*'):
    file_list.append(file)
    file_list = natsorted(file_list)
    
#file_list
data = pd.DataFrame()
for file in file_list:
    temp_file = pd.read_hdf(file)
    data = data.append(temp_file)
    
#apr_17.shape
#apr_17.head()
#apr_17.tail()
#
#apr_17.isnull().sum()
#apr_17.notnull().sum()


file_list = []

for file in glob.glob('2017_05/Librosa_2017_*'):
    file_list.append(file)
    file_list = natsorted(file_list)
    
#file_list
may_17 = pd.DataFrame()
for file in file_list:
    temp_file = pd.read_hdf(file)
    may_17 = may_17.append(temp_file)
    
#may_17.shape
#may_17.isnull().sum()
#may_17.notnull().sum()


#apr_may_17 = apr_17.append(may_17)
apr_may_17 = apr_17.iloc[:-60,:].append(may_17)
#apr_may_17.shape
#apr_may_17.isnull().sum()
#apr_may_17.notnull().sum()

def outlier_removal(data):
    
    pca = decomposition.PCA(n_components=2)
    pca.fit(data)
    t = pca.transform(data)
    #plt.scatter(t[:,0], t[:,1])

    #mask = plt.mlab.find((t[:,0]<4700) & (t[:,1]<7000))
    mask = plt.mlab.find((t[:,0]<100) & (t[:,1]<4500))

    pca_final = decomposition.PCA(n_components=2)
    pca_final.fit(data.iloc[mask])
    #t_final = pca_final.transform(data.iloc[mask])
    t_final = pca_final.transform(data)
    return t_final

inliers = outlier_removal(data)
plt.figure()
plt.scatter(inliers[:,0], inliers[:,1], s=0.0001)

##--------------24.10.2018----------------------
SA=SoundAPI()
SA.load('Auris3_indexall2.csv')

data['cluster']=0
data.ix[mask,'cluster']=1

i=np.arange(100)
plotClusters(data.iloc[i], inliers[i], pointsize=1, SA=SA)

plotClusters(data, inliers, pointsize=1, SA=SA)

#mask = (t[:,0]<500000) & (t[:,1]<500000)
#mask = (t[:,0]<10000) & (t[:,1]<10000)

#==================17.10.2018==============================
### 1. Mask that works for now is mask1 = plt.mlab.find((t[:,0]<20) & (t[:,1]<5000))
### 2. second mask that worked is mask4 = plt.mlab.find((t[:,0]<3600) & (t[:,1]<5000))
        ##(3479137, 2)
### 3. Third mask that worked is mask7 = plt.mlab.find((t[:,0]<4500) & (t[:,1]<5000))
        ##(3479372, 2)
### 4. Fourth mask that worked is mask10 = plt.mlab.find((t[:,0]<4700) & (t[:,1]<5000))
        ##(3479408, 2)
### 5. Fifth mask that worked is mask11 = plt.mlab.find((t[:,0]<4700) & (t[:,1]<6000))
### 6. Sixth mask that worked is mask12 = plt.mlab.find((t[:,0]<4700) & (t[:,1]<7000))
        ## (3479427, 2)
### 7. Seventh mask that worked is mask13 = plt.mlab.find((t[:,0]<4700) & (t[:,1]<9000))
        ## (3479427, 2)
### 8. Eighth mask that worked is mask14 = plt.mlab.find((t[:,0]<4700) & (t[:,1]<10000))
        ## (3479427, 2)
        
    ### Since the last 3 masks resulted in the same number of non-outlier points
        ## I will keep the mask = plt.mlab.find((t[:,0]<4700) & (t[:,1]<7000))
        ## which corresponds to pca12

#================= code for mask iteration ================

#mask7 = plt.mlab.find((t[:,0]<4500) & (t[:,1]<5000))
#pca7 = decomposition.PCA(n_components=2)
#pca7.fit(data.iloc[mask7])
#t7 = pca7.transform(data.iloc[mask7])
#plt.figure(7)
#plt.scatter(t7[:,0], t7[:,1], s=0.0001)

##tomorrow, try to take y axis close to 6000 #18.10.2018
##tomorrow, try to take x-axis close to 5000 and y axis close to 6000 #17.10.2018
#=========================================================
            #19.10.2018
            
### How to split data into smaller chunks for clustering 

#data = full_year
#pca = decomposition.PCA(n_components = 2)
#pca.fit(data)
#t = pca.transform(data) 
#mask = plt.mlab.find((t[:,0]<4700) & (t[:,1]<7000))

#t_final_1 = pca.transform(data.iloc[mask]) #this will give us full year
        # data without outliers and with only two dimensions. 

#t_first_division = t_final_1[:290000,:]
#t_first_division.shape
#plt.figure()
#plt.scatter(t_first_division[:,0], t_first_division[:,1])
#plt.figure()
#plt.scatter(t_first_division[:,0], t_first_division[:,1], s= 0.0001)
#t_second_division = t_final_1[290000:(290000*2),:]
#t_second_division.shape
#plt.figure()
#plt.scatter(t_second_division[:,0], t_second_division[:,1])
#plt.scatter(t_second_division[:,0], t_second_division[:,1], s= 0.0001)
#plt.figure()
#plt.scatter(t_first_division[:,0], t_first_division[:,1], s= 0.0001)
#t_third_division = t_final_1[(290000*2):(290000*3),:]
#t_third_division.shape
#plt.figure()
#plt.scatter(t_third_division[:,0], t_third_division[:,1], s=0.0001)

#=========================================================
mask = plt.mlab.find((t[:,0]<20) & (t[:,1]<40))
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

pca = decomposition.PCA(n_components=2)
pca.fit(data.iloc[mask])
t = pca.transform(data.iloc[mask])
plt.figure()
plt.scatter(t[:,0], t[:,1], s=0.0001)


ri=np.random.choice(range(len(data.iloc[mask])),size=100000)
ri_test = np.random.choice(range(len(data.iloc[mask])),size=data.shape[0])

ms=MeanShift()
msc = ms.fit(t[ri,0:3])
data.loc[mask,'msc_cluster'] = (msc.predict(t[:,0:3]))
#data['msc_cluster'] = msc.predict(t[mask][:,0:3])

kmeans = KMeans(n_clusters=10, n_jobs=-1)
km = kmeans.fit(t[ri,0:3])
data.loc[mask,'km_cluster'] = km.predict(t[:,0:3])


clusterer = HDBSCAN(min_cluster_size=1250, gen_min_span_tree=True)
hdb = clusterer.fit(t[ri,0:3])
#data.loc[mask, 'hdbscan_cluster'] = hdb.fit_predict(t[:,0:3])
#data.iloc[mask[ri], 'hdbscan_cluster'] = hdb.fit_predict(t[ri,0:3])
data.ix[mask[ri_test],'hdbscan_cluster'] = hdb.fit_predict(t[ri_test, 0:3])

SA= SoundAPI()
SA.load('Auris3_indexall2.csv')

#DIRECTORY = '/home/thileepan/data3/auris3/hdf5' #Linux
#DIRECTORY = '/Users/thileepan/data3/auris3/hdf5' #MACOS
#PREFIX = 'ristilantie_'
#SA.fastScan(DIRECTORY, PREFIX)

plotClusters(data[mask], pc=t, clusterfieldname='msc_cluster', SA=SA, pointsize = 0.0001)
plotClusters(data[mask], pc=t, clusterfieldname='km_cluster', SA=SA, pointsize= 0.001)
plotClusters(data[mask], pc=t, clusterfieldname='hdbscan_cluster', SA=SA, pointsize = 0.001)
plotClusters(data.ix[mask[ri_test]], pc=t[ri_test], clusterfieldname='hdbscan_cluster', SA=SA, pointsize = 1)

###Statistics
data.ix[mask[ri_test]]['hdbscan_cluster'].nunique()


keys=set(data.ix[mask[ri_test]]['hdbscan_cluster'].values[:,0])
colorstr='bgrcmykbgrcmykbgrcmykbgrcmyk'
colors = np.array([x for x in colorstr])
colors = np.hstack([colors] * 20)
    
for k in keys:
    i=plt.mlab.find(data.ix[mask[ri_test]].hdbscan_cluster==k)
    print (k , colors[k], len(i)*100.0/len(data.ix[mask[ri_test]]))

data.loc[mask,'cluster'] = kmeans.predict(t[:,0:3])