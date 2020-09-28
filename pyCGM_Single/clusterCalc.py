"""This file is used for calculating cluster transformations to apply
in the case of a missing marker.

It is used in:
pycgmStatic.py - lines 105-115
pycgmCalc.py - lines 132-133
pycgmCalc.py - lines 142-181

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
    >>> import numpy as np
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> normalize(v)
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
	for row in M:
		print(row)
        
        
def getMarkerLocation(Pm,C):
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
    
    #Define the transfomration matrix of the cluster in world space, World to Cluster
    Tw_c =np.matrix([[x_hat[0],y_hat[0],z_hat[0],origin[0]],
                     [x_hat[1],y_hat[1],z_hat[1],origin[1]],
                     [x_hat[2],y_hat[2],z_hat[2],origin[2]],
                     [0       ,0       ,0       ,1        ]])
            
    #Define the transfomration matrix of the marker in cluster space, cluster to Marker
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
    target_names =('C7,T10,CLAV,STRN,RBAK,LPSI,RPSI,RASI,LASI,SACR,'
                    'LKNE,LKNE,RKNE,RKNE,LANK,RANK,LHEE,RHEE,LTOE,RTOE,'
                    'LTHI,RTHI,LTIB,RTIB,'
                    'RBHD,RFHD,LBHD,LFHD,'
                    'RELB,LELB')
    
    return target_names.split(',')

def target_dict():
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

