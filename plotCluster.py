e#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:06:34 2018

@author: thileepan
"""
import sys
import time
import os
import h5py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib.mlab import find
from numpy import argmin
from numpy import  unique
import sounddevice as sd
from soundapi import SoundAPI
#import SoundAPI
import getopt
from numpy import argmin

# PLOTTING
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import cm
from matplotlib.widgets import Button


# PREPROCESSING
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA

# Saving
import soundfile as sf


def plotClusters(D, pc, pc1=0, pc2=1, pointsize=0.001, colorstr=None, clusterfieldname='cluster', SA=None, ax=None):
    def onPick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind


    def saveSample(timestamp, duration=60.0):
        global LastSampleNum
        LastSampleNum+=1
        dt=timedelta(seconds=duration)
        tshift=timedelta(seconds=duration/2.0)
        SA.select(timestamp-tshift, timestamp-tshift+dt, dt, dt)
        chunk=SA.next()
        t=chunk['t']
        filename=t.strftime("sample-%Y%m%dT%H%M%S.flac")
        sf.write(filename, chunk['data'], samplerate=chunk['FS'],
                             format='FLAC', subtype='PCM_24')
        print("Sample %s succesfully saved" % filename)

    #a=action()

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig=None
    cluster=(D.ix[:,clusterfieldname]).astype(np.int)
    if not colorstr:
        colorstr='bgrcmykbgrcmykbgrcmykbgrcmyk'
    colors = np.array([x for x in colorstr])
    colors = np.hstack([colors] * 20)
    ax.scatter(pc[:,pc1], pc[:,pc2], color=colors[cluster].tolist(), s=pointsize)
    #axb=plt.axes([0.9, 0.0, 0.1, 0.075])
    #bsave=Button(axb, 'Save')

    #for i in unique(D['cluster']):
    #    j=find(D.cluster==i)
    #    ax.plot(pc[j,pc1], pc[j,pc2], 'o', color="bgrcmyk"[i%7])

    #axoff = plt.axes([0.59, 0.05, 0.1, 0.075])
    #axspect = plt.axes([0.7, 0.05, 0.1, 0.075])
    #axplay = plt.axes([0.81, 0.05, 0.1, 0.075])
    #boff = Button(axoff, 'Off')
    #boff.on_clicked(a.setOff)
    #bspect = Button(axspect, 'Spectrum')
    #bspect.on_clicked(a.setSpectrum)
    #bplay = Button(axplay, 'Play')
    #bplay.on_clicked(a.setPlay)

    def findNearest(X,Y,x,y):
        return argmin((X-x)**2+(Y-y)**2)

    
    def onclick(event):
        global LastSampleTime
        #print event.ydata, " < ", min(pc[:, pc2]), "?"
        if event.ydata < min( pc[:,pc2]):
            saveSample(LastSampleTime)
        i=findNearest(pc[:,pc1], pc[:,pc2], event.xdata, event.ydata)
        timestamp=D.index[i]
        LastSampleTime=timestamp
        print(timestamp, '[%d, %d]: x=%f, y=%f' % (i, cluster[i], event.xdata, event.ydata))
        if SA:
            try:
                #if a.play or a.spectrum:
                dt=timedelta(seconds=10)
                SA.select(timestamp, timestamp+dt, dt, dt)
                chunk=SA.next()
                print(chunk['t'], chunk['FS'])
                sd.play(chunk['data'], chunk['FS'])
                #    if a.spectrum:
                #        pass
                #(event.button, event.x, event.y, event.xdata, event.ydata))
            except Exception as e:
                print ("Error", e)
    
    if fig:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)