#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import itertools
from scipy.signal import butter, filtfilt

# Gap Filling
def default_segment_dict():
    """Creates a default dictionary of segments to marker names.

    Used to determine which markers are in the same segments as each other,
    so that they can be used for gap filling. Works with the default pycgm
    markers.

    Returns
    -------
    segment : dict
        Dictionary of segments to marker names.
    
    Examples
    --------
    >>> result = default_segment_dict()
    >>> expected = {'Head': ['RFHD', 'RBHD', 'LFHD', 'LBHD', 'REAR', 'LEAR'],  
    ...             'Trunk': ['C7', 'STRN', 'CLAV', 'T10', 'RBAK', 'RSHO', 'LSHO'], 
    ...             'Pelvis': ['SACR', 'RPSI', 'LPSI', 'LASI', 'RASI'], 
    ...             'RThigh': ['RTHI', 'RTH2', 'RTH3', 'RTH4'], 
    ...             'LThigh': ['LTHI', 'LTH2', 'LTH3', 'LTH4'], 
    ...             'RShin': ['RTIB', 'RSH2', 'RSH3', 'RSH4'], 
    ...             'LShin': ['LTIB', 'LSH2', 'LSH3', 'LSH4'], 
    ...             'RFoot': ['RLFT1', 'RFT2', 'RMFT3', 'RLUP'], 
    ...             'LFoot': ['LLFT1', 'LFT2', 'LMFT3', 'LLUP'], 
    ...             'RHum': ['RMELB', 'RSHO', 'RUPA'], 
    ...             'LHum': ['LMELB', 'LSHO', 'LUPA']}
    >>> result == expected
    True
    """
    segment_dict = {}
    segment_dict['Head'] = ['RFHD','RBHD','LFHD','LBHD','REAR','LEAR'] 
    segment_dict['Trunk'] = ['C7','STRN','CLAV','T10','RBAK','RSHO','LSHO']
    segment_dict['Pelvis'] = ['SACR','RPSI','LPSI','LASI','RASI']
    segment_dict['RThigh'] = ['RTHI','RTH2','RTH3','RTH4']
    segment_dict['LThigh'] = ['LTHI','LTH2','LTH3','LTH4']
    segment_dict['RShin'] = ['RTIB','RSH2','RSH3','RSH4']
    segment_dict['LShin'] = ['LTIB','LSH2','LSH3','LSH4']
    segment_dict['RFoot'] = ['RLFT1','RFT2','RMFT3','RLUP']
    segment_dict['LFoot'] = ['LLFT1','LFT2','LMFT3','LLUP']
    segment_dict['RHum'] = ['RMELB','RSHO','RUPA']
    segment_dict['LHum'] = ['LMELB','LSHO','LUPA']
    
    return segment_dict

def get_marker_location(pm, c):
    """Finds the location of the missing marker in the world frame.

    Parameters
    ----------
    pm : ndarray or list
        Location of the missing marker in the cluster frame. 1x3 list or
        numpy array.
    c : 2darray or list
        2d array or list indicating locations of other markers in the same
        cluster and same frame as the missing marker.
        Contains 3 1x3 lists or numpy arrays: `[X, Y, Z]`.

    Returns
    -------
    pw : array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    
    Examples
    --------
    >>> from numpy import array, around
    >>> pm = [-205.14696889756505, 258.35355899445926, 3.279423067505604]
    >>> c = [array([ 325.82983398,  402.55450439, 1722.49816895]),
    ...      array([ 304.39898682,  242.91339111, 1694.97497559]),
    ...      array([ 197.8621521 ,  251.28889465, 1696.90197754])]
    >>> around(get_marker_location(pm, c), 8) #doctest: +NORMALIZE_WHITESPACE
    array([ 187.23396416, 407.91688108, 1720.71952837])
    """
    #Pm is the location of the missing marker in the cluster frame
    # C = [origin,x_dir,y_dir] 
    # create Tw_c is the cluster frame in the world frame
    # find Pw, the missing marker in the world frame
    
    origin = c[0]
    x_dir = c[1]
    y_dir = c[2]
    
    x_vec = x_dir - origin
    y_vec = y_dir - origin
    x_hat = x_vec / np.linalg.norm(x_vec)
    y_hat = y_vec / np.linalg.norm(y_vec)
    z_vec = np.cross(x_hat,y_hat)
    z_hat = z_vec / np.linalg.norm(z_vec)
    
    #Define the transformation matrix of the cluster in world space, World to Cluster
    tw_c =np.matrix([[x_hat[0],y_hat[0],z_hat[0],origin[0]],
                     [x_hat[1],y_hat[1],z_hat[1],origin[1]],
                     [x_hat[2],y_hat[2],z_hat[2],origin[2]],
                     [0       ,0       ,0       ,1        ]])
            
    #Define the transfomration matrix of the marker in cluster space, cluster to Marker
    tc_m = np.matrix([[1,0,0,pm[0]],
                      [0,1,0,pm[1]],
                      [0,0,1,pm[2]],
                      [0,0,0,1   ]])
        
    #Find Pw, the marker in world space
    # Take the transform from world to cluster, then multiply cluster to marker
    tw_m = tw_c * tc_m
    
    #The marker in the world frame
    pw = [tw_m[0,3],tw_m[1,3],tw_m[2,3]]
    
    return pw

def get_static_transform(p, c):
    """Finds the location of the missing marker in the cluster frame.

    Parameters
    ----------
    p : ndarray or list
        1x3 numpy array or list: `[X, Y, Z]` indicating the location of
        the marker at a previously visible frame.
    c : 2darray or list
        2d array or list indicating locations of other markers in the same
        cluster as the missing marker. Contains 3 1x3 lists or numpy arrays:
        `[X, Y, Z]`.

    Returns
    -------
    pm : array
        Location of the missing marker in the cluster frame. List of
        3 elements `[X, Y, Z]`.
    
    Examples
    --------
    >>> from numpy import array, around
    >>> p = [173.67716164, 325.44079612, 1728.47894043]
    >>> c = [array([314.17024414, 326.98319891, 1731.38964711]), 
    ...      array([302.76412032, 168.80114852, 1688.1522896 ]), 
    ...      array([193.62636014, 171.28945512, 1689.54191939])]
    >>> around(get_static_transform(p, c), 8) #doctest: +NORMALIZE_WHITESPACE
    array([-205.1469689 , 258.353559 , 3.27942307])
    """
    #p = target marker
    #C = [origin,x_dir,y_dir]
    
    origin = c[0]
    x_dir = c[1]
    y_dir = c[2]
    
    x_vec = x_dir - origin
    y_vec = y_dir - origin
    x_hat = x_vec / np.linalg.norm(x_vec)
    y_hat = y_vec / np.linalg.norm(y_vec)
    z_vec = np.cross(x_hat,y_hat)
    z_hat = z_vec / np.linalg.norm(z_vec)

    #If we consider the point to be a frame without rotation, it is simply calculated
    # Consider world frame W, cluster frame C, and marker frame M
    # We know C in relation to W (Tw_c) and we know M in relation to W (Tw_m)
    # To find M in relation to C,  Tc_m = Tc_w * Tw_m
    
    #Define the transfomration matrix of the cluster in world space, World to Cluster
    tw_c =np.matrix([[x_hat[0],y_hat[0],z_hat[0],origin[0]],
                     [x_hat[1],y_hat[1],z_hat[1],origin[1]],
                     [x_hat[2],y_hat[2],z_hat[2],origin[2]],
                     [0       ,0       ,0       ,1        ]])
    
    #Define the transfomration matrix of the marker in world space, World to Marker
    tw_m = np.matrix([[1,0,0,p[0]],
                      [0,1,0,p[1]],
                      [0,0,1,p[2]],
                      [0,0,0,1   ]])
       
    #Tc_m = Tc_w * Tw_m
    tc_m = np.linalg.inv(tw_c) * tw_m
    
    #The marker in the cluster frame
    pm = [tc_m[0,3],tc_m[1,3],tc_m[2,3]]
    
    return pm

def transform_from_static(data, data_mapping, static, static_mapping, key, useables, s):
    """Performs gap filling in the dynamic trial using data from static trials.

    Uses static data to create an inverse transformation matrix that is stored
    between a 4 marker cluster. The matrix is then applied to estimate the position
    of the missing marker `key`.

    Parameters
    ----------
    data, static : 3darray
        3d numpy array of dynamic or static trial data respectively. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping, static_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data` or `static` respectively.
    key : str
        String indicating the missing marker.
    useables : list
        List of other markers in the same cluster as the missing
        marker `key`.
    s : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    
    Examples
    --------
    >>> from numpy import around
    >>> from .io import IO
    >>> dynamic_trial = 'SampleData/Sample_2/RoboWalk.c3d'
    >>> static_trial = 'SampleData/Sample_2/RoboStatic.c3d'
    >>> data, data_mapping = IO.load_marker_data(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> static, static_mapping = IO.load_marker_data(static_trial)
    SampleData/Sample_2/RoboStatic.c3d
    >>> key = 'LFHD'
    >>> useables = ['RFHD', 'RBHD', 'LBHD'] #Other markers in the cluster
    >>> s = 1
    >>> result = transform_from_static(data, data_mapping, static, static_mapping, key, useables, s)
    >>> around(result, 8)
    array([-1007.73577975,    71.30567599,  1522.60563455])
    """
    p = np.mean(static[:,static_mapping[key]], axis=0)
    c = np.mean(static[:,static_mapping[useables[0]]],axis=0),\
        np.mean(static[:,static_mapping[useables[1]]],axis=0),\
        np.mean(static[:,static_mapping[useables[2]]],axis=0)
    
    for i, arr in enumerate(c):
        if np.isnan(arr[0]):
            print('Check static trial for gaps in',useables[i])
            pass
    
    pm = get_static_transform(p, c)
    movc = data[s][data_mapping[useables[0]]],\
           data[s][data_mapping[useables[1]]],\
           data[s][data_mapping[useables[2]]]
    
    return get_marker_location(pm, movc)

def transform_from_mov(data, data_mapping, key, clust, last_time, i):
    """Performs gap filling using previous frames of motion capture data.

    Uses previous frames of motion capture data to create an inverse
    transformation matrix that is stored between a 4 marker cluster.
    The matrix is then applied to estimate the position of the missing
    marker `key`.

    Parameters
    ----------
    data : 3darray
        3d numpy array of dynamic trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data`.
    key : str
        String indicating the missing marker.
    clust : list
        List of other markers in the same cluster as the missing
        marker `key`.
    last_time : int
        Frame number of the last frame in which all markers in `clust`
        and `key` were visible.
    i : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    
    Examples
    --------
    >>> from numpy import around
    >>> from .io import IO
    >>> dynamic_trial = 'SampleData/Sample_2/RoboWalk.c3d'
    >>> data, data_mapping = IO.load_marker_data(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> key = 'LFHD'
    >>> clust = ['RFHD', 'RBHD', 'LBHD'] #Other markers in the cluster
    >>> last_time = 1
    >>> i = 2
    >>> result = transform_from_mov(data, data_mapping, key, clust, last_time, i)
    >>> around(result, 8)
    array([-1003.42302695,    81.04948743,  1522.13413529])
    """
    p = data[last_time][data_mapping[key]]
    c = data[last_time][data_mapping[clust[0]]],\
        data[last_time][data_mapping[clust[1]]],\
        data[last_time][data_mapping[clust[2]]]
    pm = get_static_transform(p, c)
    cmov = data[i][data_mapping[clust[0]]],\
           data[i][data_mapping[clust[1]]],\
           data[i][data_mapping[clust[2]]] 
    
    return get_marker_location(pm, cmov)

def segment_finder(key, data, data_mapping, segment_dict, j, missings):
    """Find markers in the same cluster as `key` to use for gap filling.

    Finds markers in the same cluster as the marker `key` that have visible
    data that can be used to perform gap filling.

    Parameters
    ----------
    key : str
        String representing the missing marker.
    data : 3darray
        3d numpy array of dynamic trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data`.
    segment_dict : dict
        Dictionary of segments to marker names. The marker names used in
        `segment_dict` should exist in `data_mapping` to work properly.
    j : int
        Frame number that the marker data is missing for.
    missings : dict
        Dicionary of marker to list representing which other frames
        the marker is missing for.

    Returns
    -------
    useables : array
        List of marker names in the same cluster as the marker `key` that
        can be used for gap filling.
    
    Examples
    --------
    >>> from numpy import array, nan
    >>> from .io import IO
    >>> dynamic_trial = 'SampleData/Sample_2/RoboWalk.c3d'
    >>> data, data_mapping = IO.load_marker_data(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> key = 'LFHD'
    >>> segment = default_segment_dict()
    >>> j = 2
    >>> missings = {} #Indicates that we are not missing any other markers for any other frame
    >>> segment_finder(key, data, data_mapping, segment, j, missings)
    ['RFHD', 'RBHD', 'LBHD']
    """
    #Find which other markers are in the same segment as the missing key
    segment = []
    for seg in segment_dict:
        if key in segment_dict[seg]:
            segment = segment_dict[seg]
    
    if len(segment) == 0:
        return []

    useables = []
    for marker in segment:
        if marker != key:
            #Ensures we do not reconstruct based on other missing markers
            if marker in missings and j in missings[marker]:
                continue
            try:
                if not np.isnan(data[j][data_mapping[marker]][0]):
                    useables.append(marker)
            except: continue
    return useables


def rigid_fill(data, data_mapping, static, static_mapping, segment_dict):
    """Fills in gaps in motion capture data.

    Estimates marker positions from previous marker positions
    or static data to fill in gaps in `data`. Calls either
    `transform_from_static` or `transform_from_mov` where
    appropriate to estimate positions of markers with missing data.

    Parameters
    ----------
    data, static : 3darray
        3d numpy array of dynamic or static trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping, static_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data` or `static`.
    segment_dict : dict
        Dictionary of segments to marker names. The marker names used in
        `segment_dict` should exist in `data_mapping` and `static_mapping` to 
        work properly.

    Returns
    -------
    3darray
        3d numpy array of the same format as `data` after gap filling
        has been performed.
    
    Examples
    --------
    >>> from .io import IO
    >>> from numpy import array, nan, around
    >>> dynamic_trial = 'SampleData/Sample_2/RoboWalk.c3d'
    >>> static_trial = 'SampleData/Sample_2/RoboStatic.c3d'
    >>> data, data_mapping = IO.load_marker_data(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> static, static_mapping = IO.load_marker_data(static_trial)
    SampleData/Sample_2/RoboStatic.c3d
    >>> segment_dict = default_segment_dict()

    Testing gap filling.

    >>> data[2][data_mapping['LFHD']]
    array([-1003.42358398,    81.05059814,  1522.13598633])
    >>> data[2][data_mapping['LFHD']] = array([nan, nan, nan]) #Clear one frame to test gap filling
    >>> data = rigid_fill(data, data_mapping, static, static_mapping, segment_dict)
    >>> around(data[2][data_mapping['LFHD']], 8)
    array([-1003.42302695,    81.04948743,  1522.13413529])
    """
    data_copy = data.copy()
    missings = {}
    for key in data_mapping:
        traj = data_copy[:,data_mapping[key]]
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

                while j >= 0:
                    if np.isnan(data_copy[j][data_mapping[key]][0]):
                        j -= 1
                        continue

                    useables_last = segment_finder(key, data_copy, data_mapping, segment_dict, j, missings)

                    if len(useables_last) < 3:
                        j -= 1
                        continue

                    last_time = j
                    
                    break

            if last_time:
                useables_current = segment_finder(key, data_copy, data_mapping, segment_dict, j, missings)
                useables = list(set(useables_last).intersection(useables_current))

                if len(useables) < 3:
                    print('Not enough cluster markers')
                
                opts = []
                perms = list(itertools.permutations(useables))

                for p in perms:
                    subset = list(p)
                    try:
                        est_pos = transform_from_mov(data_copy, data_mapping, key, subset, last_time, i)
                        opts.append([subset,np.mean(abs(est_pos - data_copy[last_time][data_mapping[key]]))])
                    except: pass

                useables = min(opts, key = lambda t: t[1])[0]

                data_copy[i][data_mapping[key]] = transform_from_mov(data_copy, data_mapping, key, useables, last_time, i)
                continue

            if not last_time:
                useables = segment_finder(key, data_copy, data_mapping, segment_dict, i, missings)
                if len(useables) < 3:
                    continue

                data_copy[i][data_mapping[key]] = transform_from_static(data_copy, data_mapping, static, static_mapping, key, useables, i)

            missings[key].append(i)

    return data_copy    

# Filtering
def butter_filter(data, cutoff_frequency, sampling_frequency):
    """Applies a fourth order Butterworth filter.

    Fourth order Butterworth filter to be used in filt() and filter_mask_nans()
    functions, which are in Utilities. Filter is applied forward and backwards
    with the filtfilt() function -- see Notes for more details.

    Parameters
    ----------
    data : 1darray or list
        Data to be filtered.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
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
    >>> sampling_frequency = 360.0
    >>> t = 1
    >>> x = arange(0,t,1/sampling_frequency)
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

    >>> filtered = butter_filter(y, 10, sampling_frequency)
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
    c = (2**0.25-1)**0.25
    #b,a are filter coefficient calculated by scipy butter(). See scipy docs for
    #more information
    b, a = butter(4, (cutoff_frequency/c) / (sampling_frequency/2.0), btype = 'low')
    
    return filtfilt(b,a,data,axis = 0)

def filt(data, cutoff_frequency, sampling_frequency):
    """Applies a Butterworth filter to `data`.

    Takes in XYZ time series for one marker and loops over
    all 3 columns of `data`, applying `prep.butter_filter()` to each
    of them.

    Parameters
    ----------
    data : 2darray
        2d numpy array where each index of `data` contains a
        1x3 list of coordinate values: `[X, Y, Z]`.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
        Sampling frequency signal was acquired at.

    Returns
    -------
    filtered_data : 2darray
        2d numpy array of the same format as `data` after the Butterworth
        filter is applied.

    Examples
    --------
    >>> from numpy import array
    >>> data = array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4],
    ...               [3, 4, 5], [4, 5, 6], [5, 6, 7], [4, 5, 6],
    ...               [5, 6, 7], [6, 7, 8], [7, 8, 9], [6, 7, 8],
    ...               [5, 6, 7], [4, 5, 6], [5, 6, 7], [2, 3, 4]])
    >>> cutoff_frequency = 20
    >>> sampling_frequency = 120
    >>> filt(data, cutoff_frequency, sampling_frequency) #doctest: +NORMALIZE_WHITESPACE
    array([[0.99944536, 1.99944536, 2.99944536],
           [1.4397879 , 2.4397879 , 3.4397879 ],
           [1.51018336, 2.51018336, 3.51018336],
           [1.83504387, 2.83504387, 3.83504387],
           [2.97438913, 3.97438913, 4.97438913],
           [4.21680949, 5.21680949, 6.21680949],
           [4.55481712, 5.55481712, 6.55481712],
           [4.35880481, 5.35880481, 6.35880481],
           [4.88555132, 5.88555132, 6.88555132],
           [6.17293824, 7.17293824, 8.17293824],
           [6.7577089 , 7.7577089 , 8.7577089 ],
           [5.96477386, 6.96477386, 7.96477386],
           [4.99909317, 5.99909317, 6.99909317],
           [4.72228036, 5.72228036, 6.72228036],
           [4.01838019, 5.01838019, 6.01838019],
           [1.99933043, 2.99933043, 3.99933043]])
    """
    #Create empty array of the shape of data to populate with filtered data
    filtered = np.empty([len(data), np.shape(data)[1]])
    
    #iterate through each column of array and apply butter_filter()
    for i in range(np.shape(data)[1]):
        filtered[:,i] = butter_filter(data[:,i], cutoff_frequency, sampling_frequency)
    
    return filtered

def filtering(data, cutoff_frequency, sampling_frequency):
    """Applies a Butterworth filter to motion capture data.

    This function takes in motion capture data for several markers,
    loops over all of them and calls `prep.filt()` on each one.

    Parameters
    ----------
    data : 3darray
        3d numpy array of motion capture data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
        Sampling frequency signal was acquired at.

    Returns
    -------
    filtered_data : 3darray
        3d numpy array of the same format as `data` after the Butterworth
        filter is applied.
    
    Examples
    --------
    >>> from .io import IO
    >>> dynamic_trial = 'SampleData/ROM/Sample_Dynamic.c3d'
    >>> data, data_mapping = IO.load_marker_data(dynamic_trial)
    SampleData/ROM/Sample_Dynamic.c3d
    >>> filtered_data = filtering(data, 20, 120)
    >>> filtered_data[:,data_mapping['HEDO']]
    array([[ 250.34095219,  207.52056544, 1612.1177957 ],
           [ 250.3693486 ,  207.63396643, 1612.14030924],
           [ 250.39784291,  207.74607438, 1612.16076916],
           ...,
           [ 278.45835242,  292.56967662, 1612.41087668],
           [ 278.06911338,  293.22769152, 1612.49060244],
           [ 277.6663783 ,  293.88056206, 1612.55739277]])
    """
    filtered_data = np.empty(np.shape(data))
    for i in range(len(filtered_data[0])):
        result = filt(data[:,i], cutoff_frequency, sampling_frequency)
        for j in range(len(result)):
            filtered_data[j][i] = result[j]
    return filtered_data
