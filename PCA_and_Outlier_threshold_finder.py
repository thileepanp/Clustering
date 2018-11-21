#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:06:29 2018

PCA and OUTLIER removal functions: Version 1.0

The TrainPCA function will apply PCA and identify the right threshold for removing
outliers

The RemoveOutliers function uses any monthly/yearly/respresentative dataset, 
the final PCA object trained from the TrainPCA function, and also the threshold
values from the TrainPCA function to transform the data to PCA domain and apply
threshold values and remove outliers. 

The clustering function will cluster one month of data and return the labels as
a numpy array. 

The dimension reduced (using PCA) array is converted into a dataframe to which 
the labels are attached as the last column and written into a hdf5 file in the 
physical hard disk of the computer. 


@author: thileepan
"""

import pandas as pd
from sklearn.decomposition import PCA
from soundapi import SoundAPI
import numpy as np
from plotCluster import plotClusters
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted
import hdbscan

pattern_list_2016 = ['Librosa_2016_{}_*'.format(num) for num in range(4,13)] #creating pattern lists for 2016
pattern_list_2017 = ['Librosa_2017_{}_*'.format(num) for num in range(1,6)] #creating pattern lists for 2017
pattern_list = pattern_list_2016 + pattern_list_2017

def TrainPCA(training_data):
    pca1 = PCA(n_components=2)
    pca1.fit(training_data)
    t1 = pca1.transform(training_data)
    #plt.scatter(t1[:,0], t1[:,1])
    #plt.figure()
    #plt.hist(t1[:,0], bins=1000) #initial obs t1[:,0] < 3000
    #plt.figure()
    #plt.hist(t1[:,1], bins=1000) #initial obs -4400 < t1[:,1] <8730
    mask1 = plt.mlab.find((t1[:,0]<3000) & (t1[:,1]<8730))
    second_training_set = training_data.iloc[mask1]
    
    pca_final = PCA(n_components=2)
    pca_final.fit(second_training_set)
    t2 = pca_final.transform(second_training_set)
    plt.scatter(t2[:,0], t2[:,1], s=0.01)
    plt.figure()
    plt.hist(t2[:,0], bins=1000)
    plt.figure()
    plt.hist(t2[:,1], bins=1000)
    #mask2 = plt.mlab.find((t2[:,0]<8700) & (t2[:,1]<4000))
    threshold1 = 8700
    threshold2 = 4000
    return (pca_final, threshold1, threshold2)

def ReadMonthlyData(pattern):
    file_list = []
    for file in glob.glob(pattern):
         file_list.append(file)
         file_list = natsorted(file_list)
    data = pd.DataFrame()
    for file in file_list:
        temp_data = pd.read_hdf(file)
        data = data.append(temp_data)
    return data

def RemoveOutliers(data, pca, th1, th2):
    transformed_data = pca.transform(data)
    #mask = plt.mlab.find(transformed_data[(transformed_data[:,0]<th1) & (transformed_data[:,1]<th2)])
    mask = plt.mlab.find([(transformed_data[:,0]<th1) & (transformed_data[:,1]<th2)])
    IL = transformed_data[mask]
    return (IL, mask)

def Clustering(IL, mask, clustering_training_data):
    #ri_test = np.random.choice(range(len(IL)),size=np.int(IL.shape[0]/10))
    clusterer = HDBSCAN(min_cluster_size=1250, gen_min_span_tree=True)
    #hdb = clusterer.fit(clustering_training_data)
    IL.ix[mask,'hdbscan_cluster'] = clusterer.fit_predict(IL)
    
# Function to train cluster object and return the object
def TrainCluster(x):  
    clusterer = HDBSCAN(min_cluster_size=1250, gen_min_span_tree=True, prediction_data=True) # creating a clustering object
    hdb = clusterer.fit(x) #Fitting cluster object on training data
    hdb.prediction_data
    del(x)
    return hdb

def ClusterOneMonthData(month, name):
    ## Month should be an integer (0 = Apr_2016, 8= dec_2016, 9 = jan_2017, 13 = May_2017)
    ## Name should be a string, with which we should save the dataframe
    os.chdir('C:/Users/tpaulraj/Projects/Clustering/features/') #Windows
    One_Month_Data = ReadMonthlyData(pattern_list[month]) #Reading one month data using the pattern list
    IL_One_Month_Data, monthly_data_mask = RemoveOutliers(One_Month_Data, pca_final, th1, th2) #Removing outliers from 
            ##one month data
    labels, strengths = hdbscan.approximate_predict(hdb, IL_One_Month_Data) # predicting labels and strengths 
            ## for one month
    
        ##Writing the two PCs, labels and strengths into a dataframe and storing it as hdf5 file
    os.chdir('C:/Users/tpaulraj/Projects/Clustering/Results/')
    pca_dataframe = pd.DataFrame(IL_One_Month_Data)   
    pca_dataframe['labels'] = labels
    pca_dataframe['strengths'] = strengths
    pca_dataframe.columns = ['pca1', 'pca2', 'labels', 'strengths']
    pca_dataframe.to_hdf(path_or_buf= '{}_clusters'.format(name), key='pca_dataframe')
    One_Month_Data.iloc[monthly_data_mask].to_csv(path_or_buf = '{}_feature_data'.format(name), header = True, index= True)
    del(IL_One_Month_Data, One_Month_Data, labels, monthly_data_mask, pca_dataframe, strengths)
    #print('Length of one month data is {} \n'.format(len(One_Month_Data)), 'Lenght of Cluster data {} \n'.format(len(IL_One_Month_Data)), 
     #     'Length of One Month Feature data is {} \n'. format(len(One_Month_Data.iloc[monthly_data_mask])))



##Main program 
    
    #Reading Training Data

os.chdir('/home/thileepan/Projects/Clustering/features') #Linux
os.chdir('C:/Users/tpaulraj/Projects/Clustering/features/') #Windows

training_data = pd.read_csv('training_data.csv', index_col=0) # Reading training data
pca_final, th1, th2 = TrainPCA(training_data)
IL_training_data, training_data_mask = RemoveOutliers(training_data, pca_final, th1, th2)
hdb = TrainCluster(IL_training_data) #obtaining trainined cluster object
del(training_data, training_data_mask, IL_training_data)


ClusterOneMonthData(month, name)

    
##-------------------------EXCLUDED PART---------------------------------------------------------------------
    #labels, strengths = hdbscan.approximate_predict(hdb, IL_One_Month_Data[0:len(IL_One_Month_Data)/4, :])
    #labels, strengths = hdbscan.approximate_predict(hdb, IL_One_Month_Data)
        ##CPU times: user 56 s, sys: 1.38 s, total: 57.4 s
            ## Wall time: 57.5 s
    #One_Month_Data.ix[mask, 'clusters'] = labels
    #One_Month_Data.isnull().sum()
    #One_Month_Data = One_Month_Data.dropna(inplce=True)
##------------------------------------------------------------------------------------------------------------


##Program to combine features and clustered data into a single dataframe
    
os.chdir('/media/thileepan/USB DISK/Results')

features_list = pd.date_range('2016-04', '2017-06', freq='M').strftime("%B_%Y_feature_data")
clusters_list = pd.date_range('2016-04', '2017-06', freq='M').strftime("%B_%Y_clusters")
full_year_features = pd.DataFrame()


for i in range(len(features_list)):
    features_dataframe = pd.read_csv(features_list[i], parse_dates=[0])
    cluster_dataframe = pd.read_hdf(clusters_list[i])
    features_clusters = pd.concat([features_dataframe, cluster_dataframe], axis=1)
    full_year_features = full_year_features.append(features_clusters.sample(frac=0.10))

full_year_features.set_index(keys='Unnamed: 0', inplace=True)
full_year_features.index.names = ['timestamp']


## Plotting the clustered data points
SA = SoundAPI()
SA.load('Auris3_indexall2.csv')

data = pd.read_csv('features_and_clusters_data', index_col=0, parse_dates=True)

plotClusters(data, pc=data.values, pc1= 65, pc2=66, clusterfieldname='labels', SA=SA, pointsize = 1)



