#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:12:32 2018

@author: thileepan

This code imports the 'timestamp_indexing' function from the 
"timestamp_indexing_for_several_minutes" file and uses a glob.glob command to read all
the features file ('flac.h5') from the directory 
'/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/features_hdf5_files'
and then appends them all together into a single dataframe. 

The dataframe is sorted in the end according to the dates
"""
import pandas as pd
import os
import glob
os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/')
from timestamp_indexing_for_several_minutes import timestamp_indexing


os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/features_hdf5_files')

file_list=[]
for file in glob.glob("result-2017-01-13*.h5.h5"):
    file_list.append(file)
    file_list.sort()

several_days_together = pd.DataFrame()
for i in file_list:
    print(i)
    r = timestamp_indexing(i)
    print(r.shape)
    several_days_together = several_days_together.append(r)
    
several_days_together.sort_index(axis=0, inplace=True)
print("shape of the dataframe containing all YAAFE features file in this folder is {}"
      .format(several_days_together.shape))
several_days_together.to_csv('January_13_2017.csv', header=True, index= True)
