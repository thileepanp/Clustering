#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:34 2018

@author: thileepan
"""


class SoundAPI(object):
    def __init__(self):
        self.sf=pd.DataFrame()
        self.f=None
        self.currentFile=None

    def fastScan(self, directory, prefix, scanall=False):
        filenames = glob("%s/%s*Z.h5" % (directory, prefix))
        filenames.sort()
        l=timedelta(seconds=300)
        dtfix=timedelta(microseconds=1500)
        times=[ x.split('_')[-1].split('Z')[0].split('.') for x in filenames]        
        st=[datetime.strptime(ts[0], "%Y%m%dT%H%M%S")+ timedelta(microseconds=int(ts[1])) for ts in times]
        et=[ t+l+dtfix for t in st]
        df=pd.DataFrame( data={
            'filename' : filenames,
            'start' : st,
            'end' : et},
            index=[filenames])
        self.sf=self.sf.append(df)
        self.f=df


    def scan(self, directory, prefix, scanall=False):
        filenames = glob("%s/%s*.h5" % (directory, prefix))
        filenames.sort()
        self.sf=pd.DataFrame()
        for filename in filenames:
            try:
                with h5py.File(filename, 'r') as f:
                    dset=f['sound_pressure']
                    a=dset.attrs
                    micno=a['teds_serial_number']
                    FS=a['sampling_frequency']
                    st=datetime.strptime(a['tdms_started_at_minus_backlog'],
                                         "%Y-%m-%d %H:%M:%S.%f")
                    et=st+timedelta(seconds=float(len(dset))/FS)
                    #print("%s, %s, %s, %s, %s, %s" % (filename, a['capture_started_at'],
                    #                                     a['tdms_started_at_minus_backlog'],
                    #                                     a['sample_backlog_at_tdms_start'],
                    #                                     st.isoformat(), et.isoformat()))
                    df=pd.DataFrame( data={
                        'filename' : filename,
                        'start' : st,
                        'end' : et},
                        index=[filename])
                    self.sf=self.sf.append(df)
                    self.f=df
            except Exception as e:
                sys.stderr.write("Cannot open file %s: %s" % (filename, str(e)))
                
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.sf.to_csv(index=False))
        print("Data saved")

    def load(self, filename):
        self.sf = pd.read_csv(filename, parse_dates=[0,2])
        print("Data loaded")

    def select(self, start, end, chunksize, hopsize=None):
        self.chunksize=chunksize
        if hopsize==None:
            self.hopsize=chunksize
        else:
            self.hopsize=hopsize
        self.currentTime=start
        self.lastTime=end
        return self.sf[(self.sf.end >= start) & (self.sf.start <= end)]

    def getNchunks(self):
        """
        Returns the number of chunks remaining
        """
        return (self.lastTime-self.currentTime).total_seconds()/self.hopsize.seconds

    def __iter__(self):
        """Return interator interface"""
        return self
    
    def __next__(self):
        """
        Interates over a sequence. This is an interator interface for python 3.X
        """
        return self.next()

    def next(self):
        """
        Interates over a sound data duration. This is an interator interface for python 2.X
        """
        
        print("This is the rightverions")
        # Start and stop time of current chunk
        Tstart=self.currentTime
        Tend=self.currentTime+self.chunksize

        # Have we already reached the end on iterable sound data area
        if Tstart >= self.lastTime:
            raise StopIteration()

        # Read all needed files and append them to single vector, x
        x=np.zeros(0)
        ts=None  # Starting time of concatenation of needed files
        te=None  # Ending time of concatenation of needed files
        FS=0
        for filename in self.sf[(self.sf.end > Tstart) & (self.sf.start < Tend)].filename:
            if self.currentFile==None or self.currentFile.filename!=filename:
                try:
                    newFile=h5py.File(filename, 'r')
                    if self.currentFile:
                        self.currentFile.close()
                    self.currentFile=newFile
                    print("R"),
                except Exception as e:
                    print("F"),
                    sys.stderr.write("Cannot open file %s: %s" % (filename, str(e)))
                    continue
            else:
                print('C'),
            dset=self.currentFile['sound_pressure']
            a=dset.attrs
            FS=a['sampling_frequency']
            t1=datetime.strptime(a['tdms_started_at_minus_backlog'].decode('utf-8'),
                                 "%Y-%m-%d %H:%M:%S.%f")
            if (ts==None) or (t1<ts) :
                ts=t1
            try:
                x=np.hstack((x,dset))
            except Exception as e:
                print("Error: ", e, filename, FS, t1)
        self.currentTime+=self.hopsize
        retval = {'t' : Tstart,
                  'FS' : FS,
                  'data' : np.array([])}

        if len(x)==0:
            sys.stderr.write("Warning: Now sound data at %s" % (Tstart.ctime()))
            return retval
            
        te=ts+timedelta(seconds=float(len(x))/FS)

        # Requested chunk is now within x, return the selected part of x
        Sbegin=int((Tstart-ts).total_seconds()*FS)
        Send=int((te-Tend).total_seconds()*FS)
        retval['data'] = x[Sbegin:-(Send+1)]
        return retval
