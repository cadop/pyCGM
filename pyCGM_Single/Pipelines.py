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
    """Applies a fourth order Butterworth filter.

    Fourth order Butterworth filter to be used in filt() and filter_mask_nans() 
    functions, which are in Utilities. Filter is applied forward and backwards 
    with the filtfilt() function -- see Notes for more details.
    
    Parameters
    ----------
    data : 1darray or list
        Data to be filtered.
    cutoff : int
        Desired cutoff frequency.
    Fs : int
        Sampling frequency signal was acquired at.
    
    Returns
    -------
    1darray
        1D numpy array of the signal after applying the filter. 
    
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

    Examples
    --------
    First, we create a sin wave and add noise to it.

    >>> from numpy import arange, around, pi, random, shape, sin
    >>> Fs = 360.0
    >>> t = 1
    >>> x = arange(0,2,1/Fs)
    >>> f = 10
    >>> y = sin(2*pi*f*x)
    >>> around(y, 8)
    array([ 0.        ,  0.17364818,  0.34202014,  0.5       ,  0.64278761,
            0.76604444,  0.8660254 ,  0.93969262,  0.98480775,  1.        ,
            0.98480775,  0.93969262,  0.8660254 ,  0.76604444,  0.64278761,
            0.5       ,  0.34202014,  0.17364818,  0.        , -0.17364818,
           -0.34202014, -0.5       , -0.64278761, -0.76604444, -0.8660254 ,
           -0.93969262, -0.98480775, -1.        , -0.98480775, -0.93969262,
           -0.8660254 , -0.76604444, -0.64278761, -0.5       , -0.34202014,
           -0.17364818, -0.        ,...

    Add noise.

    >>> noise = random.normal(0, 0.1, shape(y))
    >>> y += noise 
    >>> around(y, 8) #doctest: +SKIP
    array([ 0.07311482,  0.10988896,  0.25388809,  0.34281796,  0.63076505,
            0.80085072,  0.80731281,  1.00976795,  0.98101546,  1.09391764,
            0.94797884,  0.86082217,  0.74357311,  0.77169265,  0.62679276,
            0.58882546,  0.09397977,  0.17420432,  0.05079215, -0.16508813,
           -0.30257866, -0.59281001, -0.73830443, -0.75690063, -0.69030496,
           -0.90486956, -0.93386976, -0.77240548, -0.95216637, -0.89735706,
           -0.9181403 , -0.83423091, -0.53978573, -0.51704481, -0.32342007,
           -0.09202642,  0.18458246,...
    
    Filter the signal.

    >>> filtered = butterFilter(y, 10, Fs)
    >>> filtered #doctest: +SKIP
    array([ 0.08064958,  0.2200619 ,  0.3571366 ,  0.48750588,  0.6068546 ,
            0.71108837,  0.79649951,  0.85992252,  0.89887073,  0.91164625,
            0.89741714,  0.85625827,  0.78915455,  0.69796821,  0.58537283,
            0.45475822,  0.31011048,  0.15587271, -0.00320784, -0.1622398 ,
           -0.31634916, -0.46083652, -0.59132481, -0.70389233, -0.79518671,
           -0.86251753, -0.90392645, -0.91823542, -0.9050733 , -0.86488133,
           -0.79889735, -0.7091183 , -0.59824082, -0.46958083, -0.32697445,
           -0.17466424, -0.01717538,...
    """
    
    #calculate correction factor for number of passes
    C = (2**0.25-1)**0.25
    #b,a are filter coefficient calculated by scipy butter(). See scipy docs for
    #more information
    b,a = butter(4, (cutoff/C) / (Fs/2), btype = 'low')
    
    return filtfilt(b,a,data, axis = 0)


def filt(data, cutoff, Fs): 
    """Applies a Butterworth filter.

    Filt applies standard Butterworth filter to signals.
    Useful when filtering (x,y,z) timeseries.
    
    Parameters
    ----------
    data : ndarray 
        Numpy array of signals to be filtered.
    cutoff : int
        Desired cutoff frequency.
    Fs : int
        Sampling frequency at which signal was acquired.       
    
    Returns
    -------
    filtered : ndarray
        Filtered data.

    To Do
    -----
    Add option to take in axis of the array to filter. E.g. axis = 0
    will filter each column.

    Examples
    --------
    >>> from numpy import array
    >>> data = array([[-1003.58361816, 81.00761414, 1522.23693848],
    ...               [-1003.50396729, 81.02921295, 1522.18493652],
    ...               [-1003.42358398, 81.05059814, 1522.13598633],
    ...               [-1003.3425293,  81.07178497, 1522.09020996],
    ...               [-1003.26068115, 81.09275818, 1522.04760742],
    ...               [-1003.17810059, 81.11352539, 1522.00805664],
    ...               [-1003.09484863, 81.13407898, 1521.97167969],
    ...               [-1003.01080322, 81.1544342,  1521.93835449],
    ...               [-1002.92608643, 81.17457581, 1521.90820312],
    ...               [-1002.84063721, 81.19451141, 1521.88122559],
    ...               [-1002.75445557, 81.21424103, 1521.8572998 ],
    ...               [-1002.6675415,  81.23376465, 1521.83654785],
    ...               [-1002.57995605, 81.25308228, 1521.81884766],
    ...               [-1002.49157715, 81.27219391, 1521.80419922],
    ...               [-1002.40252686, 81.29109955, 1521.79284668],
    ...               [-1002.31274414, 81.30980682, 1521.78442383]])
    >>> cutoff = 20
    >>> Fs = 120
    >>> filt(data, cutoff, Fs) #doctest: +NORMALIZE_WHITESPACE
    array([[-1003.58359202, 81.00761957, 1522.2369362 ],
           [-1003.50390817, 81.02919582, 1522.18514327],
           [-1003.42363665, 81.05060565, 1522.135889  ],
           [-1003.34252947, 81.07178881, 1522.0901444 ],
           [-1003.26066539, 81.09275628, 1522.04763945],
           [-1003.17811173, 81.11352192, 1522.00810375],
           [-1003.09483484, 81.13408185, 1521.97164458],
           [-1003.01081474, 81.15443324, 1521.93835939],
           [-1002.92608178, 81.17457741, 1521.90820438],
           [-1002.8406421 , 81.19451144, 1521.88119076],
           [-1002.7544567 , 81.21423685, 1521.85735721],
           [-1002.66753023, 81.23376495, 1521.83657557],
           [-1002.57993488, 81.25309204, 1521.81874533],
           [-1002.49164743, 81.27219203, 1521.80416229],
           [-1002.40251014, 81.29107514, 1521.79304586],
           [-1002.31267097, 81.30982637, 1521.78437862]])
    """
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
        A dictionary containing arrays.

    Returns
    -------
    frames : list
        Returns a list with multiple dictionaries. 

    Examples
    --------
    >>> import numpy as np
    >>> from .Pipelines import prep
    >>> trajs = {'trajOne': np.array([[217.19961548, -82.35484314, 332.2684021 ],
    ...                               [257.19961548, -32.35484314, 382.2684021 ]])}
    >>> prep(trajs) #doctest: +NORMALIZE_WHITESPACE
    [{'trajOne': array([217.19961548, -82.35484314, 332.2684021 ])}, 
    {'trajOne': array([257.19961548, -32.35484314, 382.2684021 ])}]
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
    
    Examples
    --------
    >>> import numpy as np 
    >>> from .Pipelines import clearMarker
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
    
    Examples
    --------
    >>> import numpy as np 
    >>> from .Pipelines import filtering
    >>> from .pyCGM_Helpers import getfilenames
    >>> from .pycgmIO import loadData, dataAsDict
    >>> motionData = loadData(getfilenames(x=2)[0])
    SampleData/ROM/Sample_Dynamic.c3d
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
    """Use static data for gap filling.

    Uses data from static and clusters to fill estimate
    missing marker values. Only used for markers missing 
    from frames in the start of the trial.

    Parameters
    ----------
    data : array
        Array of dictionaries of marker data.
    static : array
        Array of static marker data.
    key : str
        String representing the missing marker.
    useables : array
        Array of other markers in the same cluster as the missing marker.
    s : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of 
        3 elements.

    Examples
    --------
    >>> from .pyCGM_Helpers import getfilenames
    >>> from .Pipelines import clearMarker
    >>> from numpy import around
    >>> from .pycgmIO import loadData, dataAsDict
    >>> dynamic_trial,static_trial,_,_,_ = getfilenames(x=3)
    >>> motionData = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> motionData[1]['LFHD'] #doctest: +NORMALIZE_WHITESPACE
    array([-1003.50396729, 81.02921295, 1522.18493652])
    >>> motionData = clearMarker(motionData, 'LFHD') #clear LFHD to test gap filling
    >>> staticData = loadData(static_trial)
    SampleData/Sample_2/RoboStatic.c3d
    >>> data = dataAsDict(motionData,npArray=True)
    >>> static = dataAsDict(staticData,npArray=True)
    >>> key = 'LFHD'
    >>> useables = ['RFHD', 'RBHD', 'LBHD'] #Other markers in the cluster
    >>> s = 1
    >>> around(transform_from_static(data,static,key,useables,s), 8) #doctest: +NORMALIZE_WHITESPACE
    array([-1007.73577975, 71.30567599, 1522.60563455])
    """
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
    """Use motion data for gap filling.

    Uses previous positions of markers to estimate locations
    of missing markers.

    Parameters
    ----------
    data : array
        Array of dictionaries of marker data.
    key : str
        String representing the missing marker.
    clust : array
        Array of other markers in the same cluster as the missing marker.
    last_time : int
        Frame number of the last frame that the marker is not missing.
    i : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of 
        3 elements.

    Examples
    --------
    >>> from .pyCGM_Helpers import getfilenames
    >>> from numpy import array, nan, around
    >>> from .pycgmIO import loadData, dataAsDict
    >>> dynamic_trial,static_trial,_,_,_ = getfilenames(x=3)
    >>> motionData = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> motionData[2]['LFHD'] #doctest: +NORMALIZE_WHITESPACE
    array([-1003.42358398, 81.05059814, 1522.13598633])
    >>> motionData[2]['LFHD'] = array([nan, nan, nan]) #clear one frame to test gap filling
    >>> data = dataAsDict(motionData,npArray=True)
    >>> key = 'LFHD'
    >>> clust = ['RFHD', 'RBHD', 'LBHD'] #Other markers in the cluster
    >>> last_time = 1
    >>> i = 2
    >>> around(transform_from_mov(data,key,clust,last_time,i), 8) #doctest: +NORMALIZE_WHITESPACE
    array([-1003.42302695, 81.04948743, 1522.13413529])
    """
    p = data[key][last_time]
    C = data[clust[0]][last_time],data[clust[1]][last_time],data[clust[2]][last_time] 
    Pm = getStaticTransform(p,C)
    Cmov = data[clust[0]][i],data[clust[1]][i],data[clust[2]][i] 
    
    return getMarkerLocation(Pm,Cmov)


def segmentFinder(key,data,targetDict,segmentDict,j,missings):
    """Find markers in the same cluster as `key`.

    Parameters
    ----------
    key : str
        String representing the missing marker.
    data : array
        Array of dictionaries of marker data.
    targetDict : dict
        Dict of marker to segment.
    segmentDict : dict
        Dictionary of segments to marker names.
    j : int
        Frame number that the marker data is missing for.
    missings : dict
        Dicionary of marker to list representing which other frames
        the marker is missing for.
    
    Returns
    -------
    useables : array
        List of marker names in the same cluster as the marker `key`.

    Examples
    --------
    >>> from .pyCGM_Helpers import getfilenames
    >>> from numpy import array, nan
    >>> from .pycgmIO import loadData, dataAsDict
    >>> from .clusterCalc import target_dict, segment_dict
    >>> dynamic_trial,static_trial,_,_,_ = getfilenames(x=3)
    >>> motionData = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> motionData[2]['LFHD'] = array([nan, nan, nan]) #clear one frame to test gap filling
    >>> data = dataAsDict(motionData)
    >>> key = 'LFHD'
    >>> targetDict = target_dict()
    >>> segmentDict = segment_dict()
    >>> j = 2
    >>> missings = {'LFHD': []} #Indicates that LFHD is not missing for any other frame
    >>> segmentFinder(key, data, targetDict, segmentDict, j, missings)
    ['RFHD', 'RBHD', 'LBHD']
    """
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
    """Fills gaps in motion capture data.

    Estimates marker positions from previous marker positions
    or static data to fill in gaps in `Data`.

    Parameters
    ----------
    Data : array
        Array of dictionaries of marker data.
    static : dict
        Dictionary of marker data corresponding to a static trial.

    Returns
    -------
    data : array
        Array of dictionaries of marker data after gap filling is done.

    Examples
    --------
    >>> from .pyCGM_Helpers import getfilenames
    >>> from .pyCGM import pelvisJointCenter
    >>> from numpy import array, nan, around
    >>> from .pycgmIO import loadData, dataAsDict
    >>> dynamic_trial,static_trial,_,_,_ = getfilenames(x=3)
    >>> motionData = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> staticData = loadData(static_trial)
    SampleData/Sample_2/RoboStatic.c3d

    Sacrum must be calculated for this file using ``pyCGM.pelvisJointCenter``.

    >>> for frame in motionData:
    ...     frame['SACR'] = pelvisJointCenter(frame)[2]

    Testing gap filling.

    >>> Data = dataAsDict(motionData,npArray=True)
    >>> Data['LFHD'][2] #doctest: +NORMALIZE_WHITESPACE
    array([-1003.42358398, 81.05059814, 1522.13598633])
    >>> Data['LFHD'][2] = array([nan, nan, nan]) #clear one frame to test gap filling
    >>> static = dataAsDict(staticData,npArray=True)
    >>> data = rigid_fill(Data, static)
    >>> around(data['LFHD'][2], 8) #doctest: +NORMALIZE_WHITESPACE
    array([-1003.42302695, 81.04948743, 1522.13413529])
    """
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
