thi#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:37:26 2018

@author: thileepan
"""

from natsort import natsorted
import pandas as pd
import glob
import os

#os.chdir('/media/thileepan/Librosa_Fea/2017_04') #Linux
os.chdir('/Users/thileepan/Librosa_output') #MacOS

file_list = []

for file in glob.glob("Librosa_2017_*"):
    file_list.append(file)
    file_list = natsorted(file_list)


#apr_17 = pd.DataFrame()
#may_17 = pd.DataFrame()
apr_may_17 = pd.DataFrame()

for file in file_list:
    temp_file = pd.read_hdf(file)
    #apr_17 = apr_17.append(temp_file)
    #may_17 = may_17.append(temp_file)
    apr_may_17 = apr_may_17.append(temp_file)