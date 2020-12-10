#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

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
    >>> from refactor.io import IO
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
    The matrix is then applied to estimate the positionof the missing
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
    >>> from refactor.io import IO
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
    >>> from refactor.io import IO
    Using...
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
        print("Could not find the missing key in any segments of segment_dict.")
        return

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
    """


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
    """

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
    """

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
    """
