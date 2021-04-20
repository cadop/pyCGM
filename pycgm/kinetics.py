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


def scale(v, sc):
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
    >>> from .kinetics import scale
    >>> v = [1,2,3]
    >>> sc = 2
    >>> scale(v, sc)
    (2, 4, 6)
    """
    x, y, z = v
    return (x * sc, y * sc, z * sc)


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
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_length)

    t = np.dot(line_unit_vec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)

    nearest = tuple(np.add(nearest, start))

    return dist, nearest, pnt
