#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:33:55 2018

This code will 

------PART 1----------
1. Read all the Librosa features for 14 months 04/2016 to 05/2017 into a dataframe. 
2. Use 10% of this dataframe as a good representative set of data for the entire
    time period and fits PCA on it. [This is the PCA object]
3. Performs PCA transform and reduces dimension of this data. 
4. Clusters the dimension reduced data using HDBSCAN. [This is the clusterer object]

-----PART 2-----------
5. Reads one month data at a time into a dataframe
6. Uses the PCA object created in step 2 to tranform this data. 
7. Clusters the PCA transformed (dimension reduced) data using HDBSCAN. 
8. Writes it into an HDF5 file. 

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


#------------- PART 1-------------------------------
os.chdir('/Volumes/Librosa_Fea/features/') # CD to file location

# ------ Reading all file names------------

file_list = []
for file in glob.glob('Librosa_*'):
    file_list.append(file)
    file_list = natsorted(file_list)
    
# -------- Reading full year data into a data frame --------
    
full_year = pd.DataFrame()
for file in file_list:
    temp_file = pd.read_hdf(file)
    full_year = full_year.append(temp_file)

full_year_data = full_year

# -------- Applying PCA --------------

pca = decomposition.PCA(n_components=2)
pca.fit(full_year_data)
t = pca.transform(full_year_data)
    #------- Plotting the transformed data to visualize outliers ----
plt.figure()
plt.scatter(t[:,0], t[:,1])

#-------- Outlier removal in PCA domain -------
mask = plt.mlab.find((t[:,0]<4000) & (t[:,1]<8000)) #removing all outliers 
                                                    # above 1000 in PC1 and 
                                                    # above 6000 in PC2
    
    #------- Plotting and visualizing data after outlier removal ------
plt.figure()
plt.scatter(t[mask][:,0], t[mask][:,1])

# Removing those outliers in feature domain from the original dataframe 
    ##using the mask created in line 72 then applying PCA to the resultant data
    ## and plotting it.
full_year_without_outliers = full_year_data.iloc[mask] #creating a dataframe with no outliers
pca.fit(full_year_without_outliers) #Fitting PCA to that dataframe
t2 = pca.transform(full_year_without_outliers) # transforming data to component space
    
    #------ Plotting the final data with no outliers --------
plt.figure() 
plt.scatter(t2[:,0], t2[:,1], s=0.001)

    # Creating a dataframe with principle component data and timestamps

PCA_data_with_timestamps = pd.DataFrame(data = t2, index = full_year_without_outliers.index )

# -------- Randomly selecting 10% of this data to create training set -------------
ri=np.random.choice(range(len(full_year_without_outliers)),size=np.int(full_year_without_outliers.shape[0]/30))
    ## creating only 10% of positional indices from the full year data frame.

#-------- Clustering the data ------------------
clusterer = HDBSCAN(min_cluster_size=500, gen_min_span_tree=True) #creating a 
    ##clustering object
hdb = clusterer.fit(t2[ri,0:3]) #Fitting the clustering object to only 10% of
    ## the data after PCA.



#------------- PART 2------------------------------
    
os.chdir('/Volumes/Librosa_Fea/monthly_features/2016_04')

def Read_one_month_features(x):
    file_list = []
    for file in glob.glob('Librosa_*'):
        file_list.append(file)
        file_list = natsorted(file_list)
        
    full_month = pd.DataFrame()
    for file in file_list:
        temp_file = pd.read_hdf(file)
        full_month = full_month.append(temp_file)
        
data = full_month

#pca = decomposition.PCA(n_components=2)
#pca.fit(data)
t_monthly = pca.transform(data)
monthly_mask = plt.mlab.find((t_monthly[:,0]<4000) & (t_monthly[:,1]<8000))

plt.figure()
plt.scatter(t_monthly[monthly_mask][:,0], t_monthly[monthly_mask][:,1])

t_monthly_dataframe = pd.DataFrame(data=t_monthly)

t_monthly.ix[:, 'cluster'] = hdb.fit_predict(t_monthly[:, 0:3])