#!/usr/bin/python
# encoding: utf8
"""
Created on Fri Nov 16 18:56:13 2018

@author: thileepan


This code reads 3 features sets mentioned below and combines them together. 

1. Librosa features = 65
2. Amplitude modulation features = 3
3. Sound Pressure levels (A, C and no weighting ) = 3

Total number of features = 71

The final HDF5 file is stored in this location in windows computer

C:\\Users\\tpaulraj\\Projects\\Clustering\\features

Size of the file is about 1.7 GB and dimension is [3011874, 71]

"""
import pandas as pd
import glob
import os

# Read Amplitude Modulation data

os.chdir('C:/Users/tpaulraj/Projects/Clustering/Amplitude_Modulation/10sec_data/') #windows
am_files_list = glob.glob('Amplitude_modulation_dept_10sec_Auris3_*')

AM_full_year = pd.DataFrame()

for file in am_files_list:
    temp_am_file = pd.read_csv(file, index_col=[3], parse_dates=[3])
    AM_full_year = AM_full_year.append(temp_am_file)

#Read Librosa Data

os.chdir('C:/Users/tpaulraj/Projects/Clustering/Results')

month_list_2016 = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']  #creating month list for 2016
month_list_2017 = ['January', 'February', 'March', 'April', 'May'] #creating month list for 2017

pattern_list_2016 = ['{}_2016_feature_data'.format(month) for month in month_list_2016] #creating pattern lists for 2016
pattern_list_2017 = ['{}_2017_feature_data'.format(month) for month in month_list_2017] #creating pattern lists for 2017
pattern_list = pattern_list_2016 + pattern_list_2017 #combining both pattern lists to create a single list

Librosa_data = pd.DataFrame() 

for pattern in pattern_list:
    print(pattern)
    temp_librosa_file = pd.read_csv(pattern, parse_dates=[0], index_col=[0])
    Librosa_data = Librosa_data.append(temp_librosa_file)
    
#Read LA and LC data 

os.chdir('C:/Users/tpaulraj/Projects/Clustering/Petri_features')

Sound_Pressure_data = pd.read_hdf('Auris3_LALC_full_year.hdf5')
Sound_Pressure_data = Sound_Pressure_data.iloc[:,1:4] #not taking the AM data in this file


#Combining all features together
os.chdir('C:/Users/tpaulraj/Projects/Clustering/features')

all_features_full_year = pd.concat([Librosa_data, AM_full_year, Sound_Pressure_data], axis=1, join='inner')