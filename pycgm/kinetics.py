"""
This file is used in coordinate and vector calculations.

"""
# pyCGM

# This module was contributed by Neil M. Thomas
# the CoM calculation is not an exact clone of PiG,
# but the differences are not clinically significant.
# We will add updates accordingly.

from __future__ import division
import os
import numpy as np
import sys

if sys.version_info[0] == 2:
    pyver = 2
else:
    pyver = 3


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
    (2.0, 2.0, 3.0)
    """
    line_vec = np.subtract(end, start)
    pnt_vec = np.subtract(pnt, start)

    line_length = np.sqrt(np.dot(line_vec, line_vec))

    line_unit_vec = np.multiply(line_vec, 1/line_length)
    pnt_vec_scaled = np.multiply(pnt_vec, 1/line_length)

    t = np.dot(line_unit_vec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = (line_vec[0]*t, line_vec[1]*t, line_vec[2]*t)
    nearest = tuple(np.add(nearest, start))

    return nearest


def find_L5(frame):
    """Calculate L5 Markers using a thorax or pelvis axis

    Markers used: `LHip`, `RHip`, `Thorax_axis` or `Pelvis_axis`

    Parameters
    ----------
    frame : dict
        axis: a (4x4) array of the (x, y, z) coordinates of the axis
        LHip: position of the left hip
        RHip: position of the right hip

    Returns
    -------
    midHip, L5 : tuple
        Returns the (x, y, z) marker positions of the midHip, a (1x3) array,
        and L5, a (1x3) array, in a tuple.

    Examples
    --------
    >>> import numpy as np
    >>> from .kinetics import find_L5
    >>> Pelvis_axis = [[251.74, 392.72, 1032.78, 0],
    ...               [250.61, 391.87, 1032.87, 0],
    ...               [251.60, 391.84, 1033.88, 0],
    ...               [0, 0, 0, 1]]
    >>> LHip = np.array([308.38, 322.80, 937.98])
    >>> RHip = np.array([182.57, 339.43, 935.52])
    >>> frame = {'axis': Pelvis_axis, 'RHip': RHip, 'LHip': LHip}
    >>> np.around(find_L5(frame), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 245.48,  331.12,  936.75],
           [ 271.53,  371.69, 1043.8 ]])
    """
    z_axis = frame['axis'][2][0:3]

    norm_dir = np.array(np.multiply(z_axis, 1/np.sqrt(np.dot(z_axis, z_axis))))

    LHJC = frame['LHip']
    RHJC = frame['RHip']

    midHip = (LHJC+RHJC)/2
    mid = np.subtract(LHJC, RHJC)
    dist = np.sqrt(np.dot(mid, mid))

    offset = dist * .925
    L5 = midHip + offset * norm_dir
    return midHip, L5


# def get_kinetics(data, Bodymass):
#    """Estimate center of mass values in the global coordinate system.

#    Estimates whole body CoM in global coordinate system using PiG scaling
#    factors for determining individual segment CoM.

#    Parameters
#    -----------
#    data : array
#        Array of joint centres in the global coordinate system. List indices correspond
#        to each frame of trial. Dict keys correspond to name of each joint center,
#        dict values are arrays of (x, y, z) coordinates for each joint
#        centre.
#    Bodymass : float
#        Total bodymass (kg) of subject

#    Returns
#    -------
#    CoM_coords : 3D numpy array
#        CoM trajectory in the global coordinate system.

#    Todo
#    ----
#    Figure out weird offset

#    Examples
#    --------
#    >>> from .helpers import getfilenames
#    >>> from .IO import loadData, loadVSK
#    >>> from .pycgmStatic import getStatic
#    >>> from .calc import calcAngles
#    >>> from numpy import around
#    >>> dynamic_trial,static_trial,vsk_file,_,_ = getfilenames(x=3)
#    >>> motionData  = loadData(dynamic_trial)
#    SampleData/Sample_2/RoboWalk.c3d
#    >>> staticData = loadData(static_trial)
#    SampleData/Sample_2/RoboStatic.c3d
#    >>> vsk = loadVSK(vsk_file,dict=False)
#    >>> calSM = getStatic(staticData,vsk,flat_foot=False)
#    >>> _,joint_centers=calcAngles(motionData,start=None,end=None,vsk=calSM,
#    ...                            splitAnglesAxis=False,formatData=False,returnjoints=True)
#    >>> CoM_coords = get_kinetics(joint_centers, calSM['Bodymass'])
#    >>> around(CoM_coords[0], 2)  #doctest: +NORMALIZE_WHITESPACE
#    array([-943.59, -3.53, 856.69])
#    """

#    seg_scale = {}
#    with open(os.path.dirname(os.path.abspath(__file__)) + '/segments.csv', 'r') as f:
#        next(f)
#        for line in f:
#            row = line.rstrip('\n').split(',')
#            seg_scale[row[0]] = {'com': float(row[1]), 'mass': float(
#                row[2]), 'x': row[3], 'y': row[4], 'z': row[5]}

#    # names of segments
#    segments = ['RFoot', 'RTibia', 'RFemur', 'LFoot', 'LTibia', 'LFemur', 'Pelvis',
#                'RRadius', 'RHand', 'RHumerus', 'LRadius', 'LHand', 'LHumerus', 'Head', 'Thorax']

#    # empty array for CoM coords
#    CoM_coords = np.empty([len(data), 3])

#    # iterate through each frame of JC
#    # enumeration used to populate CoM_coords
#    for ind, frame in enumerate(data):
#        # find distal and proximal joint centres
#        seg_temp = {}

#        for seg in segments:
#            seg_temp[seg] = {}

#            # populate dict with appropriate joint centres
#            if seg == 'LFoot' or seg == 'RFoot':
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Foot']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'HEE']

#            if seg == 'LTibia' or seg == 'RTibia':
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Knee']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'Ankle']

#            if seg == 'LFemur' or seg == 'RFemur':
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Hip']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'Knee']

#            if seg == 'Pelvis':
#                midHip, L5 = find_L5(frame)
#                seg_temp[seg]['Prox'] = midHip
#                seg_temp[seg]['Dist'] = L5

#            if seg == 'Thorax':
#                # The thorax length is taken as the distance between an
#                # approximation to the C7 vertebra and the L5 vertebra in the
#                # Thorax reference frame. C7 is estimated from the C7 marker,
#                # and offset by half a marker diameter in the direction of
#                # the X axis. L5 is estimated from the L5 provided from the
#                # pelvis segment, but localised to the thorax.

#                _, L5 = find_L5(frame)
#                C7 = frame['C7']

#                CLAV = frame['CLAV']
#                STRN = frame['STRN']
#                T10 = frame['T10']

#                upper = np.array(np.multiply(np.add(CLAV, C7), 1/2.0))
#                lower = np.array(np.multiply(np.add(STRN, T10), 1/2.0))

#                # Get the direction of the primary axis Z (facing down)
#                z_vec = upper - lower
#                mag = np.sqrt(np.dot(z_vec, z_vec))

#                z_dir = np.array(np.multiply(z_vec, 1/mag))
#                new_start = upper + (z_dir * 300)
#                new_end = lower - (z_dir * 300)

#                newL5 = pnt_line(L5, new_start, new_end)
#                newC7 = pnt_line(C7, new_start, new_end)

#                seg_temp[seg]['Prox'] = np.array(newC7)
#                seg_temp[seg]['Dist'] = np.array(newL5)

#            if seg == 'LHumerus' or seg == "RHumerus":
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Shoulder']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'Humerus']

#            if seg == 'RRadius' or seg == 'LRadius':
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Humerus']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'Radius']

#            if seg == 'LHand' or seg == 'RHand':
#                seg_temp[seg]['Prox'] = frame[seg[0]+'Radius']
#                seg_temp[seg]['Dist'] = frame[seg[0]+'Hand']

#            if seg == 'Head':
#                seg_temp[seg]['Prox'] = frame['Back_Head']
#                seg_temp[seg]['Dist'] = frame['Front_Head']

#            # iterate through csv scaling values
#            for row in list(seg_scale.keys()):
#                scale = seg_scale[row]['com']
#                mass = seg_scale[row]['mass']

#                if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
#                    seg_norm = seg[1:]
#                else:
#                    seg_norm = seg

#                if seg_norm == row:
#                    prox = seg_temp[seg]['Prox']
#                    dist = seg_temp[seg]['Dist']

#                    # segment length
#                    length = prox - dist

#                    # segment CoM
#                    CoM = dist + length * scale
#                    # CoM = prox + length * scale

#                    seg_temp[seg]['CoM'] = CoM

#                    # segment mass (kg)
#                    # row[2] is mass correction factor
#                    mass = Bodymass*mass
#                    seg_temp[seg]['Mass'] = mass

#                    # segment torque
#                    torque = CoM * mass
#                    seg_temp[seg]['Torque'] = torque

#                    # vector
#                    Vector = np.array(np.subtract(CoM, [0, 0, 0]))
#                    val = Vector*mass
#                    seg_temp[seg]['val'] = val

#        vals = []
#        if pyver == 2:
#            forIter = seg_temp.iteritems()
#        if pyver == 3:
#            forIter = seg_temp.items()

#        for attr, value in forIter:
#            vals.append(value['val'])

#        CoM_coords[ind, :] = sum(vals) / Bodymass

#    return CoM_coords

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
    >>> from pyCGM_Single.pyCGM_Helpers import getfilenames
    >>> from pyCGM_Single.pycgmIO import loadData, loadVSK
    >>> from pyCGM_Single.pycgmStatic import getStatic
    >>> from pyCGM_Single.pycgmCalc import calcAngles
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
    >>> CoM_coords = get_kinetics(joint_centers, calSM['Bodymass'])
    >>> around(CoM_coords[0], 2) #doctest: +NORMALIZE_WHITESPACE
    array([-942.76, -3.58, 865.33])
    """

    # get PiG scaling table
    # PiG_xls =  pd.read_excel(os.path.dirname(os.path.abspath(__file__)) +
    #                    '/segments.xls', skiprows = 0)

    segScale = {}
    with open(os.path.dirname(os.path.abspath(__file__)) + '/segments.csv', 'r') as f:
        header = False
        for line in f:
            if header == False:
                header = line.rstrip('\n').split(',')
                header = True
            else:
                row = line.rstrip('\n').split(',')
                segScale[row[0]] = {'com': float(row[1]), 'mass': float(
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
        segTemp = {}

        for s in sides:
            for seg in segments:
                if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
                    segTemp[s+seg] = {}
                else:
                    segTemp[seg] = {}

                # populate dict with appropriate joint centres
                if seg == 'Foot':
                    #segTemp[s+seg]['Prox'] = frame[s+'Ankle']
                    #segTemp[s+seg]['Dist'] = frame[s+'Foot']
                    # should be heel to toe?
                    segTemp[s+seg]['Prox'] = frame[s+'Foot']
                    segTemp[s+seg]['Dist'] = frame[s+'HEE']

                if seg == 'Tibia':
                    segTemp[s+seg]['Prox'] = frame[s+'Knee']
                    segTemp[s+seg]['Dist'] = frame[s+'Ankle']

                if seg == 'Femur':
                    segTemp[s+seg]['Prox'] = frame[s+'Hip']
                    segTemp[s+seg]['Dist'] = frame[s+'Knee']

                if seg == 'Pelvis':

                    midHip, L5 = findL5_Pelvis(frame)  # see function above
                    segTemp[seg]['Prox'] = midHip
                    segTemp[seg]['Dist'] = L5

                if seg == 'Thorax':
                    # The thorax length is taken as the distance between an
                    # approximation to the C7 vertebra and the L5 vertebra in the
                    # Thorax reference frame. C7 is estimated from the C7 marker,
                    # and offset by half a marker diameter in the direction of
                    # the X axis. L5 is estimated from the L5 provided from the
                    # pelvis segment, but localised to the thorax.

                    L5 = findL5_Thorax(frame)
                    #_,L5 = findL5_Pelvis(frame)
                    C7 = frame['C7']

                    #y_axis = frame['Thorax_axis'][0][0]
                    #norm_dir_y = np.array(unit(y_axis))
                    # if C7_[1] >= 0:
                    #    C7 = C7_ + 100000 * norm_dir_y
                    # else:
                    #    C7 = C7_ + 100000 * norm_dir_y.flip()

                    #C7 = C7_ + 100 * -norm_dir_y

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
                    newStart = upper + (z_dir * 300)
                    newEnd = lower - (z_dir * 300)

                    _, newL5, _ = pnt2line(L5, newStart, newEnd)
                    _, newC7, _ = pnt2line(C7, newStart, newEnd)

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

                # iterate through csv scaling values
                for row in list(segScale.keys()):
                    # if row[0] == seg:
                    # scale = row[1] #row[1] contains segment names
                    # print(seg,row,segScale[row]['mass'])
                    scale = segScale[row]['com']
                    mass = segScale[row]['mass']
                    if seg == row:
                        # s = sides, which are added to limbs (not Pelvis etc.)
                        if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':

                            prox = segTemp[s+seg]['Prox']
                            dist = segTemp[s+seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment CoM
                            CoM = dist + length * scale

                            #CoM = prox + length * scale
                            segTemp[s+seg]['CoM'] = CoM

                            # segment mass (kg)
                            # row[2] contains mass corrections
                            mass = Bodymass*mass
                            segTemp[s+seg]['Mass'] = mass

                            # segment torque
                            torque = CoM * mass
                            segTemp[s+seg]['Torque'] = torque

                            # vector
                            Vector = np.array(vector(([0, 0, 0]), CoM))
                            val = Vector*mass
                            segTemp[s+seg]['val'] = val

                        # this time no side allocation
                        else:
                            prox = segTemp[seg]['Prox']
                            dist = segTemp[seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment CoM
                            CoM = dist + length * scale
                            #CoM = prox + length * scale

                            segTemp[seg]['CoM'] = CoM

                            # segment mass (kg)
                            # row[2] is mass correction factor
                            mass = Bodymass*mass
                            segTemp[seg]['Mass'] = mass

                            # segment torque
                            torque = CoM * mass
                            segTemp[seg]['Torque'] = torque

                            # vector
                            Vector = np.array(vector(([0, 0, 0]), CoM))
                            val = Vector*mass
                            segTemp[seg]['val'] = val

        keylabels = ['LHand', 'RTibia', 'Head', 'LRadius', 'RFoot', 'RRadius', 'LFoot',
                     'RHumerus', 'LTibia', 'LHumerus', 'Pelvis', 'RHand', 'RFemur', 'Thorax', 'LFemur']
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
            # print(value['val'])

        CoM_coords[ind, :] = sum(vals) / Bodymass

        # add all torques and masses
        #torques = []
        #masses = []
        # for attr, value in segTemp.iteritems():
        #    torques.append(value['Torque'])
        #    masses.append(value['Mass'])

        # calculate whole body centre of mass coordinates and add to CoM_coords array
        #CoM_coords[ind,:] = sum(torques) / sum(masses)

    return CoM_coords