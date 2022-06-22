"""
This file is used in coordinate and vector calculations.
"""

# pyCGM

# This module was contributed by Neil M. Thomas
# the CoM calculation is not an exact clone of PiG,
# but the differences are not clinically significant.
# We will add updates accordingly.
#

from __future__ import division
import os
import numpy as np
import sys

if sys.version_info[0]==2:
    pyver = 2
else:
    pyver = 3

#helper functions useful for dealing with frames of data, i.e. 1d arrays of (x,y,z)
#coordinate. Also in Utilities but need to clean everything up somewhat!
def f(p, x):
    """ Helper function for working with frames of data.

    Parameters
    ----------
    p : list
        A list of at least length 2.
    x : int or float
        Scaling factor.

    Returns
    -------
    int or float
        Returns the first value in `p` scaled by `x`, added by the second value in `p`.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import f
    >>> p = [1, 2]
    >>> x = 10
    >>> f(p, x)
    12
    """
    return (p[0] * x) + p[1]

def dot(v,w):
    """Calculate dot product of two points.

    Parameters
    ----------
    v : list
        An (x, y, z) coordinate.
    w : list
        An (x, y, z) coordinate.

    Returns
    -------
    int or float
        Returns the dot product of vectors `v` and `w`.

    Examples
    --------
    >>> from .pycgmKinetics import dot
    >>> v = [1, 2, 3]
    >>> w = [4, 5, 6]
    >>> dot(v,w)
    32
    """
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    """Calculate length of a 3D vector.

    Parameters
    ----------
    v : list
        A 3D vector.

    Returns
    -------
    float
        Returns the length of `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import length
    >>> v = [1,2,3]
    >>> np.around(length(v), 2)
    3.74
    """
    x,y,z = v
    return np.sqrt(x*x + y*y + z*z)

def vector(b,e):
    """Subtracts two vectors.

    Parameters
    ----------
    v : list
        First 3D vector.
    e : list
        Second 3D vector.

    Returns
    -------
    tuple
        Returns the vector `e` - `v`.

    Examples
    --------
    >>> from .pycgmKinetics import vector
    >>> v = [1,2,3]
    >>> e = [4,5,6]
    >>> vector(v, e)
    (3, 3, 3)
    """
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    """Calculate unit vector.

    Parameters
    ----------
    v : list
        A 3D vector.

    Returns
    -------
    tuple
        Returns the unit vector of `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import unit
    >>> v = [1,2,3]
    >>> np.around(unit(v), 2)
    array([0.27, 0.53, 0.8 ])
    """
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    """Calculate distance between two points

    Parameters
    ----------
    p0 : list
        An x, y, z coordinate point.
    p1 : list
        An x, y, z coordinate point.

    Returns
    -------
    float
        Returns distance between `p0` and `p1`.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import distance
    >>> p0 = [1,2,3]
    >>> p1 = [4,5,6]
    >>> np.around(distance(p0,p1), 2)
    5.2
    """
    return length(vector(p0,p1))

def scale(v,sc):
    """Scale a vector.

    Parameters
    ----------
    v : list
        A 3D vector.
    sc : int or float
        A scaling factor.

    Returns
    -------
    tuple
        Returns `v` scaled by `sc`.

    Examples
    --------
    >>> from .pycgmKinetics import scale
    >>> v = [1,2,3]
    >>> sc = 2
    >>> scale(v, sc)
    (2, 4, 6)
    """
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    """Add two vectors.

    Parameters
    ----------
    v : list
        A 3D vector.
    w : list
        A 3D vector.

    Returns
    -------
    tuple
        Returns the `v` + `w`.

    Examples
    --------
    >>> from .pycgmKinetics import add
    >>> v = [1, 2, 3]
    >>> w = [4, 5, 6]
    >>> add(v, w)
    (5, 7, 9)
    """
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    """Calculate shortest distance between a point and line.

    The line is represented by the points `start` and `end`.

    Parameters
    ----------
    pnt : list
        An (x, y, z) coordinate point.
    start : list
        An (x, y, z) point on the line.
    end : list
        An (x, y, z) point on the line.

    Returns
    -------
    dist, nearest, pnt : tuple (float, list, list)
        Returns `dist`, the closest distance from the point to the line,
        Returns `nearest`, the closest point on the line from the given pnt as a 1x3 array,
        Returns `pnt`, the original given pnt as a 1x3 array.

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import pnt2line
    >>> pnt = [1, 2, 3]
    >>> start = [4, 2, 3]
    >>> end = [2, 2, 3]
    >>> pnt2line(pnt, start, end)
    (1.0, (2.0, 2.0, 3.0), [1, 2, 3])
    """
    lineVec = vector(start, end)

    pntVec = vector(start, pnt)

    lineLength = length(lineVec)
    lineUnitVec = unit(lineVec)
    pntVecScaled = scale(pntVec, 1.0/lineLength)

    t = dot(lineUnitVec, pntVecScaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(lineVec, t)
    dist = distance(nearest, pntVec)

    nearest = add(nearest, start)

    return dist, nearest, pnt


#def norm3d(v): 
#    try:
#        return np.asarray(sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
#    except:
#        return np.nan


def calc_l5_pelvis(rhip, lhip, pelvis_axis):
    """Calculate L5 Markers given Pelvis Axis.

    Markers used: RHip, LHip

    Parameters
    ----------
    rhip : array
        1x3 RHip marker
    lhip : array
        1x3 LHip marker
    pelvis_axis : array
        4x4 affine matrix containing the pelvis (x, y, z) axes and origin.

    Returns
    -------
    midHip, L5 : array
        2x3 array containing the (x, y, z) positions of the midHip and L5 markers

    Examples
    --------
    >>> import numpy as np
    >>> from .pycgmKinetics import calc_l5_pelvis
    >>> lhip = np.array([308.38, 322.80, 937.98])
    >>> rhip = np.array([182.57, 339.43, 935.52])
    >>> pelvis_axis = np.array([[   0.14,    0.98,   -0.11,  251.6 ],
    ...                         [  -0.99,    0.13,   -0.02,  391.74],
    ...                         [   0.  ,    0.1 ,    0.99, 1032.89],
    ...                         [   0.  ,    0.  ,    0.  ,    0.  ]])
    >>> np.around(calc_l5_pelvis(rhip, lhip, pelvis_axis), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 245.48,  331.12,  936.75],
           [ 268.79,  367.43, 1032.58]])
    """

    rhip, lhip, pelvis_axis = map(np.asarray, [rhip, lhip, pelvis_axis])

    # The L5 position is estimated as (LHJC + RHJC)/2 +
    # (0.0, 0.0, 0.828) * Length(LHJC - RHJC), where the value 0.828
    # is a ratio of the distance from the hip joint centre level to the
    # top of the lumbar 5: this is calculated as in the vertical (z) axis
    midHip = (lhip + rhip) / 2

    offset = distance(rhip, lhip) * .828
    z_axis = pelvis_axis[2, :3] + pelvis_axis[:3, 3]
    norm_dir = np.array(unit(z_axis))
    l5 = midHip + offset * norm_dir

    return [midHip, l5]

def calc_l5_thorax(rhip, lhip, thorax_axis):
    """Calculate L5 Markers given Thorax Axis

    Markers used: RHip, LHip

    Parameters
    ----------
    rhip : array
        1x3 RHip marker
    lhip : array
        1x3 LHip marker
    thorax_axis : array
        4x4 affine matrix containing the thorax (x, y, z) axes and origin.

    Returns
    -------
    L5 : array
        Returns the (x, y, z) marker positions of the L5 in a (1x3) array.

    Examples
    --------
    >>> from .pycgmKinetics import calc_l5_thorax
    >>> import numpy as np
    >>> lhip = np.array([308.38, 322.80, 937.98])
    >>> rhip = np.array([182.57, 339.43, 935.52])
    >>> thorax_axis = np.array([[   0.07,    0.93,   -0.37,  256.27],
    ...                         [   0.99,   -0.1 ,   -0.06,  364.79],
    ...                         [  -0.09,   -0.36,   -0.93, 1462.29],
    ...                         [   0.  ,    0.  ,    0.  ,    0.  ]])
    >>> np.around(calc_l5_thorax(rhip, lhip, thorax_axis), 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 263.1 ,  356.18, 1037.28])
    """
    rhip, lhip, thorax_axis = map(np.asarray, [rhip, lhip, thorax_axis])

    thorax_o = thorax_axis[:3, 3]
    x_axis = thorax_axis[0, :3] + thorax_o
    y_axis = thorax_axis[1, :3] + thorax_o
    z_axis = thorax_axis[2, :3] + thorax_o

    midHip = (lhip+rhip)/2

    norm_dir_y = np.array(unit(y_axis))
    norm_dir = np.array(unit(z_axis))

    offset = distance(rhip, lhip) * .828

    l5 = midHip + offset * norm_dir
    return l5

def getKinetics(data, Bodymass):
    """Estimate center of mass values in the global coordinate system.

    Estimates whole body CoM in global coordinate system using PiG scaling
    factors for determining individual segment CoM.

    Parameters
    -----------
    data : array
        Array of joint centres in the global coordinate system. List indices correspond
        to each frame of trial. Dict keys correspond to name of each joint center,
        dict values are arrays of (x, y, z) coordinates for each joint
        centre.
    Bodymass : float
        Total bodymass (kg) of subject

    Returns
    -------
    CoM_coords : 3D numpy array
        CoM trajectory in the global coordinate system.

    Notes
    -----
    The PiG scaling factors are taken from Dempster -- they are available at:
    http://www.c-motion.com/download/IORGaitFiles/pigmanualver1.pdf

    Examples
    --------
    >>> from .pyCGM_Helpers import getfilenames
    >>> from .pycgmIO import loadData, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .pycgmCalc import calcAngles
    >>> from numpy import around
    >>> dynamic_trial,static_trial,vsk_file,_,_ = getfilenames(x=3)
    >>> motionData  = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> staticData = loadData(static_trial)
    SampleData/Sample_2/RoboStatic.c3d
    >>> vsk = loadVSK(vsk_file,dict=False)
    >>> calSM = getStatic(staticData,vsk,flat_foot=False)
    >>> _,joint_centers=calcAngles(motionData,start=None,end=None,vsk=calSM,
    ...                            splitAnglesAxis=False,formatData=False,returnjoints=True)
    >>> CoM_coords = getKinetics(joint_centers, calSM['Bodymass'])
    >>> around(CoM_coords[0], 2) #doctest: +NORMALIZE_WHITESPACE
    array([-941.92,   -3.61,  861.25])
    """

    #get PiG scaling table
    #PiG_xls =  pd.read_excel(os.path.dirname(os.path.abspath(__file__)) +
    #                    '/segments.xls', skiprows = 0)

    segScale = {}
    with open(os.path.dirname(os.path.abspath(__file__)) +'/segments.csv','r') as f:
        header = False
        for line in f:
            if header == False:
                header = line.rstrip('\n').split(',')
                header = True
            else:
                row = line.rstrip('\n').split(',')
                segScale[row[0]] = {'com':float(row[1]),'mass':float(row[2]),'x':row[3],'y':row[4],'z':row[5]}

    #names of segments
    sides = ['L', 'R']
    segments = ['Foot','Tibia','Femur','Pelvis','Radius','Hand','Humerus','Head','Thorax']

    #empty array for CoM coords
    CoM_coords = np.empty([len(data), 3])

    #iterate through each frame of JC
    for ind, frame in enumerate(data): #enumeration used to populate CoM_coords

        #find distal and proximal joint centres
        segTemp = {}

        for s in sides:
            for seg in segments:
                if seg!='Pelvis' and seg!='Thorax' and seg!='Head':
                    segTemp[s+seg] = {}
                else:
                    segTemp[seg] = {}

                #populate dict with appropriate joint centres
                if seg == 'Foot':
                    #segTemp[s+seg]['Prox'] = frame[s+'Ankle']
                    #segTemp[s+seg]['Dist'] = frame[s+'Foot']
                    segTemp[s+seg]['Prox'] = frame[s+'Foot'] #should be heel to toe?
                    segTemp[s+seg]['Dist'] = frame[s+'HEE']

                if seg == 'Tibia':
                    segTemp[s+seg]['Prox'] = frame[s+'Knee']
                    segTemp[s+seg]['Dist'] = frame[s+'Ankle']

                if seg == 'Femur':
                    segTemp[s+seg]['Prox'] = frame[s+'Hip']
                    segTemp[s+seg]['Dist'] = frame[s+'Knee']

                if seg == 'Pelvis':

                    midHip,L5 = calc_l5_pelvis(frame["RHip"] if "RHip" in frame else None, 
                                               frame["LHip"] if "LHip" in frame else None,
                                               frame["Pelvis_axis"] if "Pelvis_axis" in frame else None)
                                            
                    segTemp[seg]['Prox'] = midHip
                    segTemp[seg]['Dist'] = L5

                if seg == 'Thorax':
                    #The thorax length is taken as the distance between an
                    #approximation to the C7 vertebra and the L5 vertebra in the
                    #Thorax reference frame. C7 is estimated from the C7 marker,
                    #and offset by half a marker diameter in the direction of
                    #the X axis. L5 is estimated from the L5 provided from the
                    #pelvis segment, but localised to the thorax.


                    L5 = calc_l5_thorax(frame["RHip"] if "RHip" in frame else None, 
                                        frame["LHip"] if "LHip" in frame else None,
                                        frame["Thorax_axis"] if "Thorax_axis" in frame else None)
                    #_,L5 = calc_l5_pelvis(frame)
                    C7 = frame['C7']

                    #y_axis = frame['Thorax_axis'][0][0]
                    #norm_dir_y = np.array(unit(y_axis))
                    #if C7_[1] >= 0:
                    #    C7 = C7_ + 100000 * norm_dir_y
                    #else:
                    #    C7 = C7_ + 100000 * norm_dir_y.flip()

                    #C7 = C7_ + 100 * -norm_dir_y

                    CLAV = frame['CLAV']
                    STRN = frame['STRN']
                    T10 = frame['T10']

                    upper = np.array([(CLAV[0]+C7[0])/2.0,(CLAV[1]+C7[1])/2.0,(CLAV[2]+C7[2])/2.0])
                    lower = np.array([(STRN[0]+T10[0])/2.0,(STRN[1]+T10[1])/2.0,(STRN[2]+T10[2])/2.0])

                    #Get the direction of the primary axis Z (facing down)
                    z_vec = upper - lower
                    z_dir = np.array(unit(z_vec))
                    newStart = upper + (z_dir * 300)
                    newEnd = lower - (z_dir * 300)

                    _,newL5,_ = pnt2line(L5, newStart, newEnd)
                    _,newC7,_ = pnt2line(C7, newStart, newEnd)

                    segTemp[seg]['Prox'] = np.array(newC7)
                    segTemp[seg]['Dist'] = np.array(newL5)

                if seg == 'Humerus':
                    segTemp[s+seg]['Prox'] = frame[s+'Shoulder']
                    segTemp[s+seg]['Dist'] = frame[s+'Humerus']

                if seg == 'Radius':
                    segTemp[s+seg]['Prox'] = frame[s+'Humerus']
                    segTemp[s+seg]['Dist'] = frame[s+'Radius']

                if seg == 'Hand':
                    segTemp[s+seg]['Prox'] = frame[s+'Radius']
                    segTemp[s+seg]['Dist'] = frame[s+'Hand']

                if seg == 'Head':
                    segTemp[seg]['Prox'] = frame['Back_Head']
                    segTemp[seg]['Dist'] = frame['Front_Head']


                #iterate through csv scaling values
                for row in list(segScale.keys()):
                    #if row[0] == seg:
                    #scale = row[1] #row[1] contains segment names
                    # print(seg,row,segScale[row]['mass'])
                    scale = segScale[row]['com']
                    mass = segScale[row]['mass']
                    if seg == row:
                        #s = sides, which are added to limbs (not Pelvis etc.)
                        if seg!='Pelvis' and seg!='Thorax' and seg!='Head':

                            prox = segTemp[s+seg]['Prox']
                            dist = segTemp[s+seg]['Dist']

                            #segment length
                            length = prox - dist

                            #segment CoM
                            CoM = dist + length * scale

                            #CoM = prox + length * scale
                            segTemp[s+seg]['CoM'] = CoM

                            #segment mass (kg)
                            mass = Bodymass*mass #row[2] contains mass corrections
                            segTemp[s+seg]['Mass'] = mass

                            #segment torque
                            torque = CoM * mass
                            segTemp[s+seg]['Torque'] = torque

                            #vector
                            Vector = np.array(vector(([0,0,0]), CoM))
                            val = Vector*mass
                            segTemp[s+seg]['val'] = val


                        #this time no side allocation
                        else:
                            prox = segTemp[seg]['Prox']
                            dist = segTemp[seg]['Dist']

                            #segment length
                            length = prox - dist

                            #segment CoM
                            CoM = dist + length * scale
                            #CoM = prox + length * scale

                            segTemp[seg]['CoM'] = CoM

                            #segment mass (kg)
                            mass = Bodymass*mass #row[2] is mass correction factor
                            segTemp[seg]['Mass'] = mass

                            #segment torque
                            torque = CoM * mass
                            segTemp[seg]['Torque'] = torque

                            #vector
                            Vector = np.array(vector(([0,0,0]), CoM))
                            val = Vector*mass
                            segTemp[seg]['val'] = val


        keylabels  = ['LHand', 'RTibia', 'Head', 'LRadius', 'RFoot', 'RRadius', 'LFoot', 'RHumerus', 'LTibia', 'LHumerus', 'Pelvis', 'RHand', 'RFemur', 'Thorax', 'LFemur']
        # print(segTemp['RFoot'])

        # for key in keylabels:
            # print(key,segTemp[key]['val'])

        vals = []

        # for seg in list(segTemp.keys()):
            # vals.append(segTemp[seg]['val'])
        if pyver == 2:
            forIter = segTemp.iteritems()
        if pyver == 3:
            forIter = segTemp.items()

        for attr, value in forIter:
           vals.append(value['val'])
           #print(value['val'])

        CoM_coords[ind,:] = sum(vals) / Bodymass

        #add all torques and masses
        #torques = []
        #masses = []
        #for attr, value in segTemp.iteritems():
        #    torques.append(value['Torque'])
        #    masses.append(value['Mass'])

        #calculate whole body centre of mass coordinates and add to CoM_coords array
        #CoM_coords[ind,:] = sum(torques) / sum(masses)

    return CoM_coords