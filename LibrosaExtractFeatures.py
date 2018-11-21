#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:15:24 2018

@author: thileepan

A function for Feature extraction using Librosa
"""

import librosa
import numpy as np
import pandas as pd
from datetime import timedelta

def ExtractFeatures(t):
    #----EXTRACTING FEATURES------------    
    
        #--------Energy----------------
    energy = librosa.feature.rmse(y=t['data'], frame_length = 256000, hop_length=256000)[:,1:-1]
        #------melspectogram----------
    mel_spectrum = librosa.feature.melspectrogram(y=t['data'], sr=t['FS'], n_mels=40,hop_length=256000)[:,1:-1]
        #-------MFCC------------------
    mfcc= librosa.feature.mfcc(y=t['data'],sr=t['FS'], n_mfcc=13, hop_length=256000)[:,1:-1]
        #--------Spec Centroid---------
    spec_centr = librosa.feature.spectral_centroid(y=t['data'], sr=t['FS'], hop_length=256000)[:,1:-1]
        #--------Spec_bandwidth--------
    spec_bandwidth = librosa.feature.spectral_bandwidth(y = t['data'], sr = t['FS'], hop_length=256000)[:,1:-1]
        #--------Spec_contrast---------
    spec_contrast = librosa.feature.spectral_contrast(y = t['data'], sr = t['FS'], hop_length=256000)[:,1:-1]
        #------Spec Rolloff--------------
    spec_rolloff = librosa.feature.spectral_rolloff(y=t['data'], sr=t['FS'], hop_length=256000, roll_percent=0.90)[:,1:-1]
        #------Tonal Centroid------------
    #tonal_centroid = librosa.feature.tonnetz(y=t['data'], sr=t['FS'])
        #------ZCR---------------------
    zcr = librosa.feature.zero_crossing_rate(y=t['data'], frame_length=25600, hop_length=256000)[:,1:-1]
    
    #-----TIMESTAMP CREATION------
    start_timestamp=t['t']+timedelta(seconds=5)
    timestamp = pd.date_range(start=start_timestamp,freq='10S', periods=60 )
    
    #----COLUMN NAMES CREATION-----------
    energy_col_name = ['Energy']
    mel_spectrum_col_names = ['melspectrum_{}'.format(i) for i in range(0, mel_spectrum.shape[0])]
    mfcc_feature_col_names = ['mfcc_{}'.format(i) for i in range(0, mfcc.shape[0])]
    spec_centr_col_name = ['Spectral_Centroid']
    spec_bandwidth_col_name = ['Spectral_Bandwidth']
    spec_contrast_band0 = ['Spectral_Contrast_0_200']
    spec_contrast_band1 = ['Spectral_Contrast_200_400']
    spec_contrast_band2 = ['Spectral_Contrast_400_800']
    spec_contrast_band3 = ['Spectral_Contrast_800_1600']
    spec_contrast_band4 = ['Spectral_Contrast_1600_3200']
    spec_contrast_band5 = ['Spectral_Contrast_3200_6400']
    spec_contrast_band6 = ['Spectral_Contrast_6400_12800']
    spec_rolloff_col_name = ['Spectral_Rolloff']
    #toanl_centroid_col_name = ['Tonal_Centroid']
    zcr_col_name = ['Zero_Crossing_Rate']
    column_names = [energy_col_name + mel_spectrum_col_names + mfcc_feature_col_names + 
                    spec_centr_col_name + spec_bandwidth_col_name + spec_contrast_band0 + 
                    spec_contrast_band1 + spec_contrast_band2 + spec_contrast_band3 +
                    spec_contrast_band4 + spec_contrast_band5 + spec_contrast_band6 + 
                    spec_rolloff_col_name + zcr_col_name]
    
    #---CREATING A NUMPY ARRAY OF FEATURES------
    numpy_array_of_features = np.vstack((energy, mel_spectrum, mfcc, spec_centr, spec_bandwidth, spec_contrast, spec_rolloff, zcr))
    
    #----CREATING A PANDAS DATAFRAME OF FEATURES----
    ten_minutes_features= pd.DataFrame(numpy_array_of_features.T, index=timestamp, columns=column_names)
    
    return ten_minutes_features