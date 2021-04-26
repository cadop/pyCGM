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
    mag = np.sqrt(x*x + y*y + z*z)
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
    x, y, z = np.subtract(p1, p0)
    return np.sqrt(x*x + y*y + z*z)


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

    line_length = np.sqrt(
        line_vec[0]*line_vec[0] + line_vec[1]*line_vec[1] + line_vec[2]*line_vec[2])
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


def find_L5(frame):
    """Calculate L5 Markers given an axis.

    Markers used: `C7`, `RHip`, `LHip`, `axis`

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
    >>> from .kinetics import find_L5
    >>> import numpy as np
    >>> Thorax_axis = [[[256.34, 365.72, 1461.92],
    ...               [257.26, 364.69, 1462.23],
    ...               [256.18, 364.43, 1461.36]],
    ...               [256.27, 364.79, 1462.29]]
    >>> C7 = np.array([256.78, 371.28, 1459.70])
    >>> LHip = np.array([308.38, 322.80, 937.98])
    >>> RHip = np.array([182.57, 339.43, 935.52])
    >>> frame = { 'C7': C7, 'RHip': RHip, 'LHip': LHip, 'axis': Thorax_axis}
    >>> np.around(find_L5(frame), 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 265.16,  359.12, 1049.06])
    """
    x_axis, y_axis, z_axis = frame['axis'][0]

    norm_dir = np.array(unit(z_axis))
    LHJC = frame['LHip']
    RHJC = frame['RHip']
    midHip = (LHJC+RHJC)/2
    offset = distance(RHJC, LHJC) * .925

    L5 = midHip + offset * norm_dir
    return L5
