"""
This file is used in coordinate and vector calculations.

pyCGM

This module was contributed by Neil M. Thomas
the CoM calculation is not an exact clone of PiG,
but the differences are not clinically significant.
We will add updates accordingly.
"""

from __future__ import division
import os
import numpy as np
import sys

if sys.version_info[0] == 2:
    pyver = 2
else:
    pyver = 3

# helper functions useful for dealing with frames of data, i.e. 1d arrays of (x,y,z)
# coordinate. Also in Utilities but need to clean everything up somewhat!


def length(v):
    """Calculate length of a 3D vector using the distance formula.

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
    >>> from .kinetics import length
    >>> v = [1,2,3]
    >>> np.around(length(v), 2)
    3.74
    """
    x, y, z = v
    return np.sqrt(x*x + y*y + z*z)


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
    >>> from .kinetics import unit
    >>> v = [1,2,3]
    >>> np.around(unit(v), 2)
    array([0.27, 0.53, 0.8 ])
    """
    x, y, z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)


def distance(p0, p1):
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
    >>> from .kinetics import distance
    >>> p0 = [1,2,3]
    >>> p1 = [4,5,6]
    >>> np.around(distance(p0,p1), 2)
    5.2
    """
    return length(np.subtract(p1, p0))


def pnt_line(pnt, start, end):
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
    >>> from .kinetics import pnt_line
    >>> pnt = [1, 2, 3]
    >>> start = [4, 2, 3]
    >>> end = [2, 2, 3]
    >>> pnt_line(pnt, start, end)
    (1.0, (2.0, 2.0, 3.0), [1, 2, 3])
    """
    line_vec = np.subtract(end, start)

    pnt_vec = np.subtract(pnt, start)

    line_length = length(line_vec)
    line_unit_vec = unit(line_vec)
    pnt_vec_scaled = (pnt_vec[0]/line_length,
                      pnt_vec[1]/line_length, pnt_vec[2]/line_length)

    t = np.dot(line_unit_vec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = (line_vec[0]*t, line_vec[1]*t, line_vec[2]*t)
    dist = distance(nearest, pnt_vec)

    nearest = tuple(np.add(nearest, start))

    return dist, nearest, pnt


def find_L5_pelvis(frame):
    """Calculate L5 Markers Given Pelvis function

    Markers used: `LHip`, `RHip`, `Pelvis_axis`

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.

    Returns
    -------
    midHip, L5 : tuple
        Returns the (x, y, z) marker positions of the midHip, a (1x3) array,
        and L5, a (1x3) array, in a tuple.

    Examples
    --------
    >>> import numpy as np
    >>> from .kinetics import find_L5_pelvis
    >>> Pelvis_axis = [np.array([251.60, 391.74, 1032.89]),
    ...                np.array([[251.74, 392.72, 1032.78],
    ...                    [250.61, 391.87, 1032.87],
    ...                    [251.60, 391.84, 1033.88]]),
    ...                np.array([231.57, 210.25, 1052.24])]
    >>> LHip = np.array([308.38, 322.80, 937.98])
    >>> RHip = np.array([182.57, 339.43, 935.52])
    >>> frame = { 'Pelvis_axis': Pelvis_axis, 'RHip': RHip, 'LHip': LHip}
    >>> np.around(find_L5_pelvis(frame), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 245.48,  331.12,  936.75],
           [ 271.53,  371.69, 1043.8 ]])
    """
    # The L5 position is estimated as (LHJC + RHJC)/2 +
    # (0.0, 0.0, 0.828) * Length(LHJC - RHJC), where the value 0.828
    # is a ratio of the distance from the hip joint centre level to the
    # top of the lumbar 5: this is calculated as in teh vertical (z) axis
    LHJC = frame['LHip']
    RHJC = frame['RHip']
    midHip = (LHJC+RHJC)/2
    # zOffset = ([0.0,0.0,distance(RHJC, LHJC)*0.925])
    # L5 = midHip + zOffset

    offset = distance(RHJC, LHJC) * .925
    z_axis = frame['Pelvis_axis'][1][2]
    norm_dir = np.array(unit(z_axis))
    L5 = midHip + offset * norm_dir

    return midHip, L5  # midHip + ([0.0, 0.0, zOffset])


def find_L5_thorax(frame):
    """Calculate L5 Markers Given Thorax function.

    Markers used: `C7`, `RHip`, `LHip`, `Thorax_axis`

    Parameters
    ----------
    frame : dict
        Dictionaries of marker lists.

    Returns
    -------
    L5 : array
        Returns the (x, y, z) marker positions of the L5 in a (1x3) array.

    Examples
    --------
    >>> from .kinetics import find_L5_thorax
    >>> import numpy as np
    >>> Thorax_axis = [[[256.34, 365.72, 1461.92],
    ...               [257.26, 364.69, 1462.23],
    ...               [256.18, 364.43, 1461.36]],
    ...               [256.27, 364.79, 1462.29]]
    >>> C7 = np.array([256.78, 371.28, 1459.70])
    >>> LHip = np.array([308.38, 322.80, 937.98])
    >>> RHip = np.array([182.57, 339.43, 935.52])
    >>> frame = { 'C7': C7, 'RHip': RHip, 'LHip': LHip, 'Thorax_axis': Thorax_axis}
    >>> np.around(find_L5_thorax(frame), 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 265.16,  359.12, 1049.06])
    """
    C7_ = frame['C7']
    x_axis, y_axis, z_axis = frame['Thorax_axis'][0]
    norm_dir_y = np.array(unit(y_axis))
    if C7_[1] >= 0:
        C7 = C7_ + 7 * -norm_dir_y
    else:
        C7 = C7_ + 7 * norm_dir_y

    norm_dir = np.array(unit(z_axis))
    LHJC = frame['LHip']
    RHJC = frame['RHip']
    midHip = (LHJC+RHJC)/2
    offset = distance(RHJC, LHJC) * .925

    L5 = midHip + offset * norm_dir
    return L5


def get_kinetics(data, Bodymass):
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

    Todo
    ----
    Tidy up and optimise code

    Joint moments etc.

    Figure out weird offset

    Examples
    --------
    >>> from .helpers import getfilenames
    >>> from .IO import loadData, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .calc import calc_angles
    >>> from numpy import around
    >>> dynamic_trial,static_trial,vsk_file,_,_ = getfilenames(x=3)
    >>> motionData  = loadData(dynamic_trial)
    SampleData/Sample_2/RoboWalk.c3d
    >>> staticData = loadData(static_trial)
    SampleData/Sample_2/RoboStatic.c3d
    >>> vsk = loadVSK(vsk_file,dict=False)
    >>> calSM = getStatic(staticData,vsk,flat_foot=False)
    >>> _,joint_centers=calc_angles(motionData,start=None,end=None,vsk=calSM,
    ...                            splitAnglesAxis=False,formatData=False,returnjoints=True)
    >>> CoM_coords = get_kinetics(joint_centers, calSM['Bodymass'])
    >>> around(CoM_coords[0], 2) #doctest: +NORMALIZE_WHITESPACE
    array([-942.76, -3.58, 865.33])
    """

    # get PiG scaling table
    # PiG_xls =  pd.read_excel(os.path.dirname(os.path.abspath(__file__)) +
    #                    '/segments.xls', skiprows = 0)

    seg_scale = {}
    with open(os.path.dirname(os.path.abspath(__file__)) + '/segments.csv', 'r') as f:
        header = False
        for line in f:
            if header == False:
                # header = line.rstrip('\n').split(',')
                header = True
            else:
                row = line.rstrip('\n').split(',')
                seg_scale[row[0]] = {'com': float(row[1]), 'mass': float(
                    row[2]), 'x': row[3], 'y': row[4], 'z': row[5]}

    # names of segments
    sides = ['L', 'R']
    segments = ['Foot', 'Tibia', 'Femur', 'Pelvis',
                'Radius', 'Hand', 'Humerus', 'Head', 'Thorax']

    # empty array for CoM coords
    CoM_coords = np.empty([len(data), 3])

    # iterate through each frame of JC
    # enumeration used to populate CoM_coords
    for ind, frame in enumerate(data):
        # find distal and proximal joint centres
        seg_temp = {}

        for s in sides:
            for seg in segments:
                if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
                    seg_temp[s+seg] = {}
                else:
                    seg_temp[seg] = {}

                # populate dict with appropriate joint centres
                if seg == 'Foot':
                    seg_temp[s+seg]['Prox'] = frame[s+'Foot']
                    seg_temp[s+seg]['Dist'] = frame[s+'HEE']

                if seg == 'Tibia':
                    seg_temp[s+seg]['Prox'] = frame[s+'Knee']
                    seg_temp[s+seg]['Dist'] = frame[s+'Ankle']

                if seg == 'Femur':
                    seg_temp[s+seg]['Prox'] = frame[s+'Hip']
                    seg_temp[s+seg]['Dist'] = frame[s+'Knee']

                if seg == 'Pelvis':
                    midHip, L5 = find_L5_pelvis(frame)  # see function above
                    seg_temp[seg]['Prox'] = midHip
                    seg_temp[seg]['Dist'] = L5

                if seg == 'Thorax':
                    # The thorax length is taken as the distance between an
                    # approximation to the C7 vertebra and the L5 vertebra in the
                    # Thorax reference frame. C7 is estimated from the C7 marker,
                    # and offset by half a marker diameter in the direction of
                    # the X axis. L5 is estimated from the L5 provided from the
                    # pelvis segment, but localised to the thorax.

                    L5 = find_L5_thorax(frame)
                    C7 = frame['C7']

                    CLAV = frame['CLAV']
                    STRN = frame['STRN']
                    T10 = frame['T10']

                    upper = np.array(
                        [(CLAV[0]+C7[0])/2.0, (CLAV[1]+C7[1])/2.0, (CLAV[2]+C7[2])/2.0])
                    lower = np.array(
                        [(STRN[0]+T10[0])/2.0, (STRN[1]+T10[1])/2.0, (STRN[2]+T10[2])/2.0])

                    # Get the direction of the primary axis Z (facing down)
                    z_vec = upper - lower
                    z_dir = np.array(unit(z_vec))
                    new_start = upper + (z_dir * 300)
                    new_end = lower - (z_dir * 300)

                    _, newL5, _ = pnt_line(L5, new_start, new_end)
                    _, newC7, _ = pnt_line(C7, new_start, new_end)

                    seg_temp[seg]['Prox'] = np.array(newC7)
                    seg_temp[seg]['Dist'] = np.array(newL5)

                if seg == 'Humerus':
                    seg_temp[s+seg]['Prox'] = frame[s+'Shoulder']
                    seg_temp[s+seg]['Dist'] = frame[s+'Humerus']

                if seg == 'Radius':
                    seg_temp[s+seg]['Prox'] = frame[s+'Humerus']
                    seg_temp[s+seg]['Dist'] = frame[s+'Radius']

                if seg == 'Hand':
                    seg_temp[s+seg]['Prox'] = frame[s+'Radius']
                    seg_temp[s+seg]['Dist'] = frame[s+'Hand']

                if seg == 'Head':
                    seg_temp[seg]['Prox'] = frame['Back_Head']
                    seg_temp[seg]['Dist'] = frame['Front_Head']

                # iterate through csv scaling values
                for row in list(seg_scale.keys()):
                    # if row[0] == seg:
                    # scale = row[1] #row[1] contains segment names
                    # print(seg,row,seg_scale[row]['mass'])
                    scale = seg_scale[row]['com']
                    mass = seg_scale[row]['mass']
                    if seg == row:
                        # s = sides, which are added to limbs (not Pelvis etc.)
                        if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':

                            prox = seg_temp[s+seg]['Prox']
                            dist = seg_temp[s+seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment CoM
                            CoM = dist + length * scale

                            # CoM = prox + length * scale
                            seg_temp[s+seg]['CoM'] = CoM

                            # segment mass (kg)
                            # row[2] contains mass corrections
                            mass = Bodymass*mass
                            seg_temp[s+seg]['Mass'] = mass

                            # segment torque
                            torque = CoM * mass
                            seg_temp[s+seg]['Torque'] = torque

                            # vector
                            Vector = np.array(np.subtract(CoM, [0, 0, 0]))
                            val = Vector*mass
                            seg_temp[s+seg]['val'] = val

                        # this time no side allocation
                        else:
                            prox = seg_temp[seg]['Prox']
                            dist = seg_temp[seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment CoM
                            CoM = dist + length * scale
                            # CoM = prox + length * scale

                            seg_temp[seg]['CoM'] = CoM

                            # segment mass (kg)
                            # row[2] is mass correction factor
                            mass = Bodymass*mass
                            seg_temp[seg]['Mass'] = mass

                            # segment torque
                            torque = CoM * mass
                            seg_temp[seg]['Torque'] = torque

                            # vector
                            Vector = np.array(np.subtract(CoM, [0, 0, 0]))
                            val = Vector*mass
                            seg_temp[seg]['val'] = val

        keylabels = ['LHand', 'RTibia', 'Head', 'LRadius', 'RFoot', 'RRadius', 'LFoot',
                     'RHumerus', 'LTibia', 'LHumerus', 'Pelvis', 'RHand', 'RFemur', 'Thorax', 'LFemur']

        vals = []
        if pyver == 2:
            forIter = seg_temp.iteritems()
        if pyver == 3:
            forIter = seg_temp.items()

        for attr, value in forIter:
            vals.append(value['val'])

        CoM_coords[ind, :] = sum(vals) / Bodymass

    return CoM_coords
