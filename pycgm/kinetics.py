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

# helper functions useful for dealing with frames of data, i.e. 1d arrays of (x,y,z) coordinates


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

    line_length = np.sqrt(np.dot(line_vec, line_vec))

    line_unit_vec = np.multiply(line_vec, 1/line_length)
    pnt_vec_scaled = np.multiply(pnt_vec, 1/line_length)

    t = np.dot(line_unit_vec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = (line_vec[0]*t, line_vec[1]*t, line_vec[2]*t)
    arr = np.subtract(pnt_vec, nearest)
    dist = np.sqrt(np.dot(arr, arr))
    nearest = tuple(np.add(nearest, start))

    return dist, nearest, pnt


def find_L5(frame):
    """Calculate L5 Markers using a given axis

    Markers used: `LHip`, `RHip`, `axis`

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
    >>> frame = { 'axis': Pelvis_axis, 'RHip': RHip, 'LHip': LHip}
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
