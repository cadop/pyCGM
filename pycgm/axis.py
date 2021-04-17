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

import math
import numpy as np

def find_joint_center(mark_a, mark_b, mark_c, delta):
    """Calculate the Joint Center.
    This function is based on physical markers, a,b,c and joint center which
    will be calulcated in this function are all in the same plane.
    Parameters
    ----------
    mark_a, mark_b, mark_c : list
        Three markers x,y,z position of a, b, c.
    delta : float
        The length from marker to joint center, retrieved from subject measurement file.
    Returns
    -------
    joint_center : array
        Returns the Joint C x, y, z positions in a 1x3 array.
    Examples
    --------
    >>> import numpy as np
    >>> from .axis import find_joint_center
    >>> mark_a = [468.14, 325.09, 673.12]
    >>> mark_b = [355.90, 365.38, 940.69]
    >>> mark_c = [452.35, 329.06, 524.77]
    >>> delta = 59.5
    >>> find_joint_center(mark_a, mark_b, mark_c, delta).round(2)
    array([396.25, 347.92, 518.63])
    """
    # make the two vector using 3 markers, which is on the same plane.
    vec_1 = (mark_a[0]-mark_c[0], mark_a[1]-mark_c[1], mark_a[2]-mark_c[2])
    vec_2 = (mark_b[0]-mark_c[0], mark_b[1]-mark_c[1], mark_b[2]-mark_c[2])

    # vec_3 is cross vector of vec_1, vec_2
    # and then it normalized.
    # vec_3 = cross(vec_1, vec_2)
    vec_3 = np.cross(vec_1, vec_2)
    vec_3_div = np.linalg.norm(vec_3)
    vec_3 = [vec_3[0]/vec_3_div, vec_3[1]/vec_3_div, vec_3[2]/vec_3_div]

    mid = [(mark_b[0]+mark_c[0])/2.0, (mark_b[1]+mark_c[1])/2.0, (mark_b[2]+mark_c[2])/2.0]
    length = np.subtract(mark_b, mid)
    length = np.linalg.norm(length)

    theta = math.acos(delta/np.linalg.norm(vec_2))

    cs_th = math.cos(theta*2)
    sn_th = math.sin(theta*2)

    u_x = vec_3[0]
    u_y = vec_3[1]
    u_z = vec_3[2]

    # this rotation matrix is called Rodriques' rotation formula.
    # In order to make a plane, at least 3 number of markers is required which
    # means three physical markers on the segment can make a plane.
    # then the orthogonal vector of the plane will be rotating axis.
    # joint center is determined by rotating the one vector of plane around rotating axis.

    rot = np.matrix([
        [cs_th+u_x**2.0*(1.0-cs_th),u_x*u_y*(1.0-cs_th)-u_z*sn_th,u_x*u_z*(1.0-cs_th)+u_y*sn_th],
        [u_y*u_x*(1.0-cs_th)+u_z*sn_th,cs_th+u_y**2.0*(1.0-cs_th),u_y*u_z*(1.0-cs_th)-u_x*sn_th],
        [u_z*u_x*(1.0-cs_th)-u_y*sn_th,u_z*u_y*(1.0-cs_th)+u_x*sn_th,cs_th+u_z**2.0*(1.0-cs_th)]
    ])

    r_vec = rot * (np.matrix(vec_2).transpose())
    r_vec = r_vec * length/np.linalg.norm(r_vec)

    r_vec = [r_vec[0,0], r_vec[1,0], r_vec[2,0]]
    joint_center = np.array([r_vec[0]+mid[0], r_vec[1]+mid[1], r_vec[2]+mid[2]])

    return joint_center
