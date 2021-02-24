"""This file is used for calculating cluster transformations to apply
in the case of a missing marker.

It is currently imported directly in pycgmClusters.py and Pipelines.py.
"""

#plotting 
import numpy as np

def normalize(v):
    """Normalizes an input vector

    Parameters
    ----------
    v : ndarray
        Input vector. 1-D list of floating point numbers.

    Returns
    -------
    ndarray
        Normalized form of input vector `v`. Returns `v` if its norm is 0.

    Examples
    --------
    >>> from numpy import array, around
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> around(normalize(v), 8)
    array([0.26726124, 0.53452248, 0.80178373])

    >>> v = np.array([0, 0, 0])
    >>> normalize(v)
    array([0, 0, 0])

    """
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

#Print matrix
def printMat(M):
	"""Prints a matrix.

	Parameters
	----------
	M : array_like
	    Input matrix.

	Examples
	--------
	>>> M = [[1, 2, 3],
	...      [4, 5, 6],
	...      [7, 8, 9]]
	>>> printMat(M)
	[1, 2, 3]
	[4, 5, 6]
	[7, 8, 9]
	"""
	for row in M:
		print(row)


def getMarkerLocation(Pm,C):
    """Finds the missing marker in the world frame.

    Parameters
    ----------
    Pm : array
        Location of the missing marker in the cluster frame. List or
        numpy array of 3 elements.
    C : array
        C is of the form [origin, x_dir, y_dir].

    Returns
    -------
    Pw : array
        Location of the missing marker in the world frame. List of
        3 elements.

    Examples
    --------
    >>> from numpy import array, around
    >>> Pm = [-205.14, 258.35, 3.27]
    >>> C = [array([ 325.82,  402.55, 1722.49]),
    ...      array([ 304.39,  242.91, 1694.97]),
    ...      array([ 197.86,  251.28, 1696.90])]
    >>> around(getMarkerLocation(Pm, C), 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 187.23,  407.9 , 1720.72])
    """
    #Pm is the location of the missing marker in the cluster frame
    # C = [origin,x_dir,y_dir]
    # create Tw_c is the cluster frame in the world frame
    # find Pw, the missing marker in the world frame

    origin = C[0]
    x_dir = C[1]
    y_dir = C[2]

    x_vec = x_dir - origin
    y_vec = y_dir - origin
    x_hat = normalize(x_vec)
    y_hat = normalize(y_vec)
    z_vec = np.cross(x_hat,y_hat)
    z_hat = normalize(z_vec)

    #Define the transformation matrix of the cluster in world space, World to Cluster
    Tw_c =np.matrix([[x_hat[0],y_hat[0],z_hat[0],origin[0]],
                     [x_hat[1],y_hat[1],z_hat[1],origin[1]],
                     [x_hat[2],y_hat[2],z_hat[2],origin[2]],
                     [0       ,0       ,0       ,1        ]])

    # Define the transfomration matrix of the marker in cluster space, cluster to Marker
    Tc_m = np.matrix([[1,0,0,Pm[0]],
                      [0,1,0,Pm[1]],
                      [0,0,1,Pm[2]],
                      [0,0,0,1   ]])

    #Find Pw, the marker in world space
    # Take the transform from world to cluster, then multiply cluster to marker
    Tw_m = Tw_c * Tc_m

    #The marker in the world frame
    Pw = [Tw_m[0,3],Tw_m[1,3],Tw_m[2,3]]

    return Pw


def getStaticTransform(p,C):
    """Find the location of the missing marker in the cluster frame.

    Parameters
    ----------
    p : array
        Location of the target marker. List or numpy array of 3 elements.
    C : array
        C is of the form [origin, x_dir, y_dir]. Each of the elements in
        C is a numpy array of 3 elements.

    Returns
    -------
    Pm : array
        Location of the missing marker in the cluster frame. List of
        3 elements.

    Examples
    --------
    >>> from numpy import array, around
    >>> p = [173.67, 325.44, 1728.47]
    >>> C = [array([314.17, 326.98, 1731.38]),
    ...      array([302.76, 168.80, 1688.15]),
    ...      array([193.62, 171.28, 1689.54])]
    >>> around(getStaticTransform(p, C), 2) #doctest: +NORMALIZE_WHITESPACE
    array([-205.16,  258.37,    3.28])
    """
    #p = target marker
    #C = [origin,x_dir,y_dir]

    origin = C[0]
    x_dir = C[1]
    y_dir = C[2]

    x_vec = x_dir - origin
    y_vec = y_dir - origin
    x_hat = normalize(x_vec)
    y_hat = normalize(y_vec)
    z_vec = np.cross(x_hat,y_hat)
    z_hat = normalize(z_vec)

    #If we consider the point to be a frame without rotation, it is simply calculated
    # Consider world frame W, cluster frame C, and marker frame M
    # We know C in relation to W (Tw_c) and we know M in relation to W (Tw_m)
    # To find M in relation to C,  Tc_m = Tc_w * Tw_m

    #Define the transfomration matrix of the cluster in world space, World to Cluster
    Tw_c =np.matrix([[x_hat[0],y_hat[0],z_hat[0],origin[0]],
                     [x_hat[1],y_hat[1],z_hat[1],origin[1]],
                     [x_hat[2],y_hat[2],z_hat[2],origin[2]],
                     [0       ,0       ,0       ,1        ]])

    #Define the transfomration matrix of the marker in world space, World to Marker
    Tw_m = np.matrix([[1,0,0,p[0]],
                      [0,1,0,p[1]],
                      [0,0,1,p[2]],
                      [0,0,0,1   ]])

    #Tc_m = Tc_w * Tw_m
    Tc_m = np.linalg.inv(Tw_c) * Tw_m

    #The marker in the cluster frame
    Pm = [Tc_m[0,3],Tc_m[1,3],Tc_m[2,3]]

    return Pm

def targetName():
    """Creates an empty list of marker names.

    Returns
    -------
    target_names : array
        Empty list of marker names.

    Examples
    --------
    >>> targetName() #doctest: +NORMALIZE_WHITESPACE
    ['C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LPSI', 'RPSI',
     'RASI', 'LASI', 'SACR', 'LKNE', 'LKNE', 'RKNE', 'RKNE',
     'LANK', 'RANK', 'LHEE', 'RHEE', 'LTOE', 'RTOE', 'LTHI',
     'RTHI', 'LTIB', 'RTIB', 'RBHD', 'RFHD', 'LBHD', 'LFHD',
     'RELB', 'LELB']
    """
    target_names =('C7,T10,CLAV,STRN,RBAK,LPSI,RPSI,RASI,LASI,SACR,'
                    'LKNE,LKNE,RKNE,RKNE,LANK,RANK,LHEE,RHEE,LTOE,RTOE,'
                    'LTHI,RTHI,LTIB,RTIB,'
                    'RBHD,RFHD,LBHD,LFHD,'
                    'RELB,LELB')

    return target_names.split(',')

def target_dict():
    """Creates a dictionary of marker to segment.

    Returns
    -------
    targetDict : dict
        Dict of marker to segment.

    Examples
    --------
    >>> result = target_dict()
    >>> expected = {'LFHD': 'Head', 'LBHD': 'Head', 'RFHD': 'Head', 'RBHD': 'Head',
    ...             'C7': 'Trunk', 'T10': 'Trunk', 'CLAV': 'Trunk', 'STRN': 'Trunk',
    ...             'RBAK': 'Trunk', 'LPSI': 'Pelvis', 'RPSI': 'Pelvis', 'RASI': 'Pelvis',
    ...             'LASI': 'Pelvis', 'SACR': 'Pelvis', 'LKNE': 'LThigh', 'RKNE': 'RThigh',
    ...             'LANK': 'LShin', 'RANK': 'RShin', 'LHEE': 'LFoot', 'LTOE': 'LFoot',
    ...             'RHEE': 'RFoot', 'RTOE': 'RFoot', 'LTHI': 'LThigh', 'RTHI': 'RThigh',
    ...             'LTIB': 'LShin', 'RTIB': 'RShin', 'RELB': 'RHum', 'LELB': 'LHum'}
    >>> result == expected
    True
    """
    targetDict = {}
    targetDict['LFHD'] = 'Head'
    targetDict['LBHD'] = 'Head'
    targetDict['RFHD'] = 'Head'
    targetDict['RBHD'] = 'Head'
    targetDict['C7'] = 'Trunk'
    targetDict['T10'] = 'Trunk'
    targetDict['CLAV'] = 'Trunk'
    targetDict['STRN'] = 'Trunk'
    targetDict['RBAK'] = 'Trunk'
    targetDict['LPSI'] = 'Pelvis'
    targetDict['RPSI'] = 'Pelvis'
    targetDict['RASI'] = 'Pelvis'
    targetDict['LASI'] = 'Pelvis'
    targetDict['SACR'] = 'Pelvis'
    targetDict['LKNE'] = 'LThigh'
    targetDict['RKNE'] = 'RThigh'
    targetDict['LANK'] = 'LShin'
    targetDict['RANK'] = 'RShin'
    targetDict['LHEE'] = 'LFoot'
    targetDict['LTOE'] = 'LFoot'
    targetDict['RHEE'] = 'RFoot'
    targetDict['RTOE'] = 'RFoot'
    targetDict['LTHI'] = 'LThigh'
    targetDict['RTHI'] = 'RThigh'
    targetDict['LTIB'] = 'LShin'
    targetDict['RTIB'] = 'RShin'
    targetDict['RELB'] = 'RHum'
    targetDict['LELB'] = 'LHum'

    return targetDict

def segment_dict():
    """Creates a dictionary of segments to marker names.

    Returns
    -------
    segmentDict : dict
        Dictionary of segments to marker names.

    Examples
    --------
    >>> result = segment_dict()
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
    segmentDict = {}
    segmentDict['Head'] = ['RFHD','RBHD','LFHD','LBHD','REAR','LEAR']
    segmentDict['Trunk'] = ['C7','STRN','CLAV','T10','RBAK','RSHO','LSHO']
    segmentDict['Pelvis'] = ['SACR','RPSI','LPSI','LASI','RASI']
    segmentDict['RThigh'] = ['RTHI','RTH2','RTH3','RTH4']
    segmentDict['LThigh'] = ['LTHI','LTH2','LTH3','LTH4']
    segmentDict['RShin'] = ['RTIB','RSH2','RSH3','RSH4']
    segmentDict['LShin'] = ['LTIB','LSH2','LSH3','LSH4']
    segmentDict['RFoot'] = ['RLFT1','RFT2','RMFT3','RLUP']
    segmentDict['LFoot'] = ['LLFT1','LFT2','LMFT3','LLUP']
    segmentDict['RHum'] = ['RMELB','RSHO','RUPA']
    segmentDict['LHum'] = ['LMELB','LSHO','LUPA']

    return segmentDict
