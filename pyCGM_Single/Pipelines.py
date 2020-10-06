# This module was contributed by Neil M. Thomas

import numpy as np
from .clusterCalc import targetName,getMarkerLocation,segment_dict,target_dict,getStaticTransform
import itertools
from scipy.signal import butter, filtfilt
import sys

if sys.version_info[0]==2:
    pyver = 2
    print("Using python 2")

else:
    pyver = 3
    print("Using python 3")

def butterFilter(data, cutoff, Fs):
    '''
    Fourth order Butterworth filter to be used in filt() and filter_mask_nans() 
    functions, which are in Utilities. Filter is applied forward and backwards 
    with the filtfilt() function -- see Notes for more details.
    
    Parameters
    ----------
    data: 1darray 
        Data to be filtered.
        
    cutoff: int
        Desired cutoff frequency.
        
    Fs: int
        Sampling frequency signal was acquired at.
    
    
    Returns
    -------
    Out: 1darray
        Filtered signal. 
    
    
    Notes
    -----
    Applying the filter one way will create a phase shift of the output
    signal compared to the input signal. For a 2nd order filter, this will
    be 90 degrees. Thus, filtfilt applies the signal once forward and once
    backward, which is referred to as phase correction. Whilst this brings
    the net phase shift to zero, it also means the cutoff of the filter will
    be twice as sharp when compared to a single filtering. In effect, a 2nd
    order filter applied twice will be a 4th order filter. We can apply a 
    correction factor to the cuttoff frequency to compensate. Correction
    factor C = square root of 2 to the power of 1/n - 1, where n is equal to
    the number of passes. 
    '''
    
    #calculate correction factor for number of passes
    C = (2**0.25-1)**0.25
    #b,a are filter coefficient calculated by scipy butter(). See scipy docs for
    #more information
    b,a = butter(4, (cutoff/C) / (Fs/2), btype = 'low')
    
    return filtfilt(b,a,data, axis = 0)


def filt(data, cutoff, Fs): 
    '''
    Filt applies standard Butterworth filter to signals.
    Useful when filtering (x,y,z) timeseries 
    
    Parameters
    ----------
    data: ndarray 
        Signals to be filtered.
        
    cutoff: int
        Desired cutoff frequency.
        
    Fs: int
        Sampling frequency at which signal was acquired. 
        
    axis: int
        Axis of array to filter (e.g. axis = 0 will filter each column).
        
    
    Returns
    -------
    Out: ndarray
        Filtered data.
        
    Todo
    ----
    Add axis option 
    '''
    #empty array to populate
    filtered = np.empty([len(data), np.shape(data)[1]])
    
    #iterate through each column of array and apply butterFilter(), which is 
    #found in Utilities
    for i in range(np.shape(data)[1]):
        filtered[:,i] = butterFilter(data[:,i], cutoff, Fs)
    
    return filtered


def prep(trajs):
    """Prepare frames function

    Parameters
    ----------
    trajs : dict

    Returns
    -------
    frames : array

    Example
    -------
    >>> from .Pipelines import *
    >>> import numpy as np
    >>> trajs = {'trajOne': [[217.19961548, -82.35484314, 332.2684021 ],
    ...                  [[257.19961548, -32.35484314, 382.2684021 ]]]}
    >>> prep(trajs) #doctest: +NORMALIZE_WHITESPACE
    [{'trajOne': [[217.19961548, -82.35484314, 332.2684021], 
    [[257.19961548, -32.35484314, 382.2684021]]]}, 
    {'trajOne': [[[257.19961548, -32.35484314, 382.2684021]]]}]
    """
    frames=[]
    if pyver == 2:
        for i in range(len(trajs[trajs.keys()[0]])):
            temp={}
            for key, val in trajs.iteritems():
                temp.update({key:val[i,:]})
            frames.append(temp)
        
    if pyver == 3:
        for i in range(len(trajs[list(trajs.keys())[0]])):
            temp={}
            for key, val in trajs.items():
                temp.update({key:val[i,:]})
            frames.append(temp)
    
    return frames

def clearMarker(data,name):
    """Clear Markers function

    Clears the markers in a given dictionary for a given name.

    Parameters
    ----------
    data : dict 
        Dictionaries of marker lists.
            { [], [], [], ...}
    name : str
        Name for the specific marker to be cleared in data.
    
    Returns
    -------
    data : dict
        Returns the original data dictionary with the cleared marker.
    
    Example
    -------
    >>> import numpy as np 
    >>> from .Pipelines import *
    >>> data = [{'LTIL': np.array([-268.1545105 ,  327.53512573,   30.17036057]), 
    ... 'RFOP': np.array([ -38.4509964 , -148.6839447 ,   59.21961594])},
    ... {'LTIL': np.array([-273.1545105 ,  324.53512573,   36.17036057]),
    ... 'RFOP': np.array([ -38.4509964 , -148.6839447 ,   59.21961594])}]
    >>> name = 'LTIL'
    >>> clearMarker(data, name) #doctest: +NORMALIZE_WHITESPACE
    [{'LTIL': array([nan, nan, nan]), 
    'RFOP': array([ -38.4509964 , -148.6839447 ,   59.21961594])}, 
    {'LTIL': array([nan, nan, nan]), 
    'RFOP': array([ -38.4509964 , -148.6839447 ,   59.21961594])}]
    """
    for i in range(len(data)):
        data[i][name] = np.array([np.nan,np.nan,np.nan])
    return data
    
def filtering(Data):
    """Filter function

    Parameters
    ----------
    Data : dict 
        Dictionaries of marker lists.
            { [], [], [], ...}
    
    Returns
    -------
    data : dict
    
    Example
    -------
    >>> import numpy as np 
    >>> from .Pipelines import *
    >>> from .pyCGM_Helpers import getfilenames
    >>> from .pycgmIO import loadData,dataAsDict
    Using...
    >>> dynamic_trial,static_trial,vsk_file,outputfile,CoM_output = getfilenames(x=2)
    >>> motionData = loadData(dynamic_trial)
    Sample...
    >>> motionDataDict = dataAsDict(motionData,npArray=True)
    >>> filtering(motionDataDict)['HEDO'] #doctest: +NORMALIZE_WHITESPACE
    array([[ 250.34095219,  207.52056544, 1612.1177957 ],
           [ 250.3693486 ,  207.63396643, 1612.14030924],
           [ 250.39784291,  207.74607438, 1612.16076916],
           ...,
           [ 278.45835242,  292.56967662, 1612.41087668],
           [ 278.06911338,  293.22769152, 1612.49060244],
           [ 277.6663783 ,  293.88056206, 1612.55739277]])
    """
    data = Data.copy()
    
    if pyver == 2:
        for key,val in data.iteritems():
            data[key] = filt(data[key],20,120)
    
    if pyver == 3:
        for key,val in data.items():
            data[key] = filt(data[key],20,120)
    
    return data
    
    
def transform_from_static(data,static,key,useables,s):
    p = np.mean(static[key],axis=0)
    C = np.mean(static[useables[0]],axis=0),np.mean(static[useables[1]],axis=0),np.mean(static[useables[2]],axis=0)
    
    for i,arr in enumerate(C):
        if np.isnan(arr[0]):
            print('Check static trial for gaps in',useables[i])
            pass
        
    Pm = getStaticTransform(p,C)
    movC = data[useables[0]][s],data[useables[1]][s],data[useables[2]][s]

    return getMarkerLocation(Pm,movC)


def transform_from_mov(data,key,clust,last_time,i):
    p = data[key][last_time]
    C = data[clust[0]][last_time],data[clust[1]][last_time],data[clust[2]][last_time] 
    Pm = getStaticTransform(p,C)
    Cmov = data[clust[0]][i],data[clust[1]][i],data[clust[2]][i] 
    
    return getMarkerLocation(Pm,Cmov)


def segmentFinder(key,data,targetDict,segmentDict,j,missings):
    segment = targetDict[key]
    useables=[]
    for mrker in segmentDict[segment]:
        if mrker != key:
            #this ensures we don't reconstruct a marker based on another
            #reconstructed marker
            if mrker[1:]!='THI' or mrker[1:]!='TIB':
               if mrker in missings and j in missings[mrker]:
                   continue
            try:
                if not np.isnan(data[mrker][j][0]):
                    useables.append(mrker)
            except: continue
    
    return useables
                        

def rigid_fill(Data,static):
    data = Data.copy()
    missingMarkerName=targetName()
    targetDict = target_dict()
    segmentDict = segment_dict()
        
    missings={}
    
    #Need to do something like this to avoid issues with CGM variants
    # missingMarkerName.remove('LPSI')
    # missingMarkerName.remove('RPSI')
    # missingMarkerName.remove('SACR')
    
    removedMarkers = [name for name in missingMarkerName if name not in data.keys()]
    
    for key in removedMarkers:
        #data[key] = np.empty(shape=(len(data[data.keys()[0]]),3))*np.nan
        data[key] = np.empty(shape=(len(data[list(data.keys())[0]]),3))*np.nan
    
    #always use transform from static for removed markers (new one for every 
    #frame)
    if pyver == 2:
        forIter = data.iteritems()
    if pyver == 3:
        forIter = data.items()
        
    for key, val in forIter:
        if key in missingMarkerName and key in removedMarkers:
            traj = data[key]
            
            for i, val in enumerate(traj):
                useables = segmentFinder(key,data,targetDict,segmentDict,i,missings)
                       
                if len(useables) < 3:
                    print('Cannot reconstruct',key,': no valid cluster')
                    continue
                else:
                    data[key][i] = transform_from_static(data,static,key,useables,i)    
                    # try: data[key][i] = transform_from_static(data,static,key,useables,i)    
                    # except: pass #key might not be used which is why it is missing i.e., LPSI vs SACR
            
        #use last known marker position (start of every gap) for transform
        #during movement trial gaps
        if key in missingMarkerName and key not in removedMarkers:  
            traj = data[key]
            gap_bool = False
            last_time = None
            
            missings[key] = []
            for i, val in enumerate(traj):
                if not np.isnan(val[0]):
                    gap_bool = False
                    last_time = None
                   
                    continue
                
                if not gap_bool:
                    gap_bool = True
                    j = i
                    
                    while j >=0:
                        if np.isnan(data[key][j][0]):
                            j -= 1
                            continue

                        useables_last = segmentFinder(key,data,targetDict,segmentDict,j,missings)

                        if len(useables_last) < 3:
                            j-=1
                            continue
                        
                        last_time = j

                        break
                    
                    #print('The target marker',key,' was visible at',last_time)
                
                if last_time:
                    #if np.isnan(data[useables[0]][i][0]) or np.isnan(data[useables[1]][i][0]) or np.isnan(data[useables[2]][i][0]):
                         #print('current clust',useables,'invalid for',key,'at frame',i)
                    useables_current = segmentFinder(key,data,targetDict,segmentDict,i,missings)
                    useables = list(set(useables_last).intersection(useables_current))
                    
                    if len(useables) < 3:
                        print('Not enough cluster markers')
                    
                    opts = []
                    perms = list(itertools.permutations(useables))
                    
                    for p in perms:
                        subset = list(p)
                        try:
                            est_pos = transform_from_mov(data,key,subset,last_time,i) 
                            opts.append([subset,np.mean(abs(est_pos - data[key][last_time]))])
                        except: pass
                    
                    useables = min(opts, key = lambda t: t[1])[0]
                    
                    #print('using new clust',useables,'for key')
                    data[key][i] = transform_from_mov(data,key,useables,last_time,i)
                    continue

                #use static transform for markers missing from the start 
                #of the trial only. Make new one for each missing frame.
                if not last_time:
                    useables = segmentFinder(key,data,targetDict,segmentDict,i,missings)

                    if len(useables) < 3:
                        print('cannot find valid cluster for',key)
                        continue
                   
                    data[key][i] = transform_from_static(data,static,key,useables,i)
                       
                        #print transform_from_static(data,static,key,useables,i)
                #record reconstructed frames
                missings[key].append(i)

    return data                
