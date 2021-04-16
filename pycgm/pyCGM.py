# -*- coding: utf-8 -*-
#pyCGM

# Copyright (c) 2015 Mathew Schwartz <umcadop@gmail.com>
# Core Developers: Seungeun Yeon, Mathew Schwartz
# Contributors Filipe Alves Caixeta, Robert Van-wesep
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#pyCGM

"""
This file is used in joint angle and joint center calculations.
"""

import sys
import os
from math import sin, cos, pi, sqrt
import math
import numpy as np

def norm_div(vec):
    """Normalized divison.

    This function calculates the normalization division of a 3-dimensional vector.

    Parameters
    ----------
    vec : list
        A 3D vector.

    Returns
    -------
    array
        The divison normalization of the vector returned as a float in an array.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import norm_div
    >>> vec = [1.44, 1.94, 2.49]
    >>> np.around(norm_div(vec), 2)
    array([0.12, 0.16, 0.21])
    """
    try:
        vec_b = sqrt((vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]))
        vec = [vec[0]/vec_b, vec[1]/vec_b, vec[2]/vec_b]
    except:
        vec_b = np.nan

    return [vec[0]/vec_b, vec[1]/vec_b, vec[2]/vec_b]

def matrixmult (matr_a, matr_b):
    """Matrix multiplication.

    This function returns the product of a matrix multiplication given two matrices.

    Let the dimension of the matrix A be: m by n,
    let the dimension of the matrix B be: p by q,
    multiplication will only possible if n = p,
    creating a matrix of m by q size.

    Parameters
    ----------
    matr_a : list
        First matrix, in a 2D array format.
    matr_b : list
        Second matrix, in a 2D array format.

    Returns
    -------
    matr_ab : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> from .pyCGM import matrixmult
    >>> matr_a = [[11,12,13],[14,15,16]]
    >>> matr_b = [[1,2],[3,4],[5,6]]
    >>> matrixmult(matr_a, matr_b)
    [[112, 148], [139, 184]]
    """

    matr_ab = [[0 for row in range(len(matr_a))] for col in range(len(matr_b[0]))]
    for i in range(len(matr_a)):
        for j in range(len(matr_b[0])):
            for k in range(len(matr_b)):
                matr_ab[i][j] += matr_a[i][k]*matr_b[k][j]
    return matr_ab


def rotmat(x=0, y=0, z=0):
    """Rotation Matrix.

    This function creates and returns a rotation matrix.

    Parameters
    ----------
    x, y, z : float, optional
        Angle, which will be converted to radians, in
        each respective axis to describe the rotations.
        The default is 0 for each unspecified angle.

    Returns
    -------
    Rxyz : list
        The product of the matrix multiplication.

    Examples
    --------
    >>> import numpy as np
    >>> from .pyCGM import rotmat
    >>> x = 0.5
    >>> y = 0.3
    >>> z = 0.8
    >>> np.around(rotmat(x, y, z), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  , -0.01,  0.01],
    [ 0.01,  1.  , -0.01],
    [-0.01,  0.01,  1.  ]])
    >>> x = 0.5
    >>> np.around(rotmat(x), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.  ],
    [ 0.  ,  1.  , -0.01],
    [ 0.  ,  0.01,  1.  ]])
    >>> x = 1
    >>> y = 1
    >>> np.around(rotmat(x,y), 2) #doctest: +NORMALIZE_WHITESPACE
    array([[ 1.  ,  0.  ,  0.02],
    [ 0.  ,  1.  , -0.02],
    [-0.02,  0.02,  1.  ]])
    """
    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    r_x = [ [1,0,0],[0,math.cos(x),math.sin(x)*-1],[0,math.sin(x),math.cos(x)] ]
    r_y = [ [math.cos(y),0,math.sin(y)],[0,1,0],[math.sin(y)*-1,0,math.cos(y)] ]
    r_z = [ [math.cos(z),math.sin(z)*-1,0],[math.sin(z),math.cos(z),0],[0,0,1] ]
    r_xy = matrixmult(r_x,r_y)
    r_xyz = matrixmult(r_xy,r_z)

    return r_xyz
