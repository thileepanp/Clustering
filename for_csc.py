#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:46:15 2018

@author: thileepan


"""
import sys
sys.path.append('/home/thileepan/Windsome/WindSoMe_Recordings/Recordings')
import librosa
from soundapi import SoundAPI
from LibrosaExtractFeatures import ExtractFeatures
from datetime import datetime, timedelta
import pandas as pd
import os


if len(sys.argv) <5:
	print "Usage: \n\t%s <site> <year> <month> <date> " % sys.argv[0]
	sys.exit()
site=int(sys.argv[1])
year = int(sys.argv[2])
month = int(sys.argv[3])
day = int(sys.argv[4])
#day = 22


#os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/currently_working')
#from timestamp_creation import timestamp_indexing


os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction')
#os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/flac_files')

#DIRECTORY = '/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/data3/auris3/hdf5'
#DIRECTORY = '/home/thileepan/ownCloud/amplitude_modulation'
DIRECTORY = '/home/thileepan/data3/auris%d/hdf5' % (site)
PREFIX = '*_'



SA = SoundAPI()
SA.fastScan(DIRECTORY, PREFIX)
td=timedelta(seconds=5)
SA.select(datetime(year,month,day,0,0,0)-td, datetime(year,month,day,23,59,59)+td, timedelta(seconds =610),timedelta(seconds=600))
hopsize = 600

one_day_df = pd.DataFrame()
        
for chunk in SA:
    try:
        print(chunk)
        F=ExtractFeatures(chunk)
        if (F.shape[0]<60 | F.shape[0]>60):
            print('Check this time' + chunk['t'])
        print(F.shape)
        one_day_df = one_day_df.append(F)
        prev_rows = one_day_df.shape[0]
        if (prev_rows + F.shape[0] >120 | prev_rows + F.shape[0] <120):
            print('check this time' + chunk['t'])
    except Exception as e:
        print('warning', e)
        

one_day_df = one_day_df.iloc[:-60,:]
print(one_day_df.shape)

os.chdir('/home/thileepan/Dropbox/PhD/2018_work/feature_extraction/Librosa')
one_day_df.to_hdf('Librosa_{}_{}_{}.h5'.format(year,month,day), key='features')



"""
ERROR --> Library exception

C
R
Error:  Can't read data (inflate() failed) /home/thileepan/data3/auris3/hdf5/ristilantie_20170422T171338.579443Z.h5 25600 2017-04-22 17:13:38.579444
R
{'FS': 25600, 't': datetime.datetime(2017, 4, 22, 17, 9, 55), 'data': array([-0.08856805, -0.08979163, -0.08950926, ...,  0.00050825,
       -0.00135535, -0.00284246])}
warning Empty data passed with indices specified.

CHECK TIMESTAMP

C
R
R
{'FS': 25600, 't': datetime.datetime(2017, 4, 22, 2, 29, 55), 'data': array([  8.09442368e-04,   5.64727234e-05,   5.27078751e-04, ...,
         2.51491852e-02,   2.51115374e-02,   2.81987134e-02])}

"""
