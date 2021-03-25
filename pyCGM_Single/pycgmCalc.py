"""
This file uses the input data to perform angle and joint calculations.
"""

# pyCGM

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
# -*- coding: utf-8 -*-

from .pyCGM import *
from .pycgmKinetics import getKinetics
import sys
if sys.version_info[0]==2:
    pyver = 2
else:
    pyver = 3

#Used to split the arrays with angles and axis
#Start Joint Angles
SJA=0
#End Joint Angles
EJA=SJA+19*3
#Start Axis
SA=EJA
#End Axis
EA=SA+72*3

def calcKinetics(data, Bodymass):
    """Calculates center of mass values.

    Estimates whole body center of mass in global coordinate system using PiG scaling 
    factors for determining individual segment center of mass.

    See Also
    --------
    pyCGM_Single.pycgmKinetics.getKinetics : equivalent function.
    """
    r = getKinetics(data, Bodymass)
    return r


def calcAngles(data,**kargs):
    """Calculates the joint angles and axis.

    By default, the function will calculate all the data and return angles
    and axis as separate arrays. The values returned by this function currently
    differ in return type and value depending on the keyword arguments of
    **kargs. The function is currently used directly in pyCGM/pycgm_embed.py.

    Parameters
    ----------
    data : array
        Joint centers in the global coordinate system. List indices correspond
        to each frame of trial. Dict keys correspond to name of each joint center,
        dict values are arrays ([],[],[]) of x,y,z coordinates for each joint
        center.
    **kargs : keyword arguments
        start : int, optional
           Indicates which index in `data` to start the calculation.
        end : int, optional
           Indicates which index in `data` to end the calculation.
           The data at index `end` is not included.
        frame : int, optional
            Frame number if the calculation is only for one frame.
            Incompatible with `start` and `end`.
        vsk : dict, required
            Subject measurement values as a dictionary or labels and data.
        angles : bool, optional
            If true, the function will return `angles`. True by default.
        axis : bool, optional
            If true, the function will return `axis`. True by default.
        splitAnglesAxis : bool, optional
            If true, the function will return `angles` and `axis` as
            separate arrays. If false, it will be the same array. True
            by default.
        returnjoints : bool, optional
            If true, the function will return `returnjoints`. False
            by default.
        formatData : bool, optional
            If true, the function will return `angles` and `axis` 
            in one array. True by default.

    Returns
    -------
    r, jcs : array_like
        `r` is a list of joint angle values for each frame.
        `jcs` is a list of dictionaries, each of which holds joint
        center locations for each frame. Returned only if returnjoints
        is True.

    Raises
    ------
    Exception
        If `start` is given and is negative.
        If `start` is larger than `end`.
        If `end` is larger than the length of `data`.

    Examples
    --------
    First, we load motion capture data from Sample_Static.c3d
    and subject measurement values from Sample_SM.vsk in
    /SampleData/ROM/.

    >>> from numpy import around
    >>> from .pycgmIO import loadC3D, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .pyCGM_Helpers import getfilenames
    >>> filenames = getfilenames(x=2)
    >>> c3dFile = filenames[1]
    >>> vskFile = filenames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> vskData = loadVSK(vskFile, False)
    >>> vsk = getStatic(data,vskData,flat_foot=False)

    Example of default behavior.

    >>> result = calcAngles(data, vsk=vsk)
    >>> around(result[0], 2) #Array of joint angles #doctest: +NORMALIZE_WHITESPACE 
    array([[[ -0.46,  -5.76,   4.81],
            [  3.04,  -7.02, -17.41],
            [ -3.  ,  -4.54,  -1.74],
            ...,
            [ 36.3 ,   0.  ,   0.  ],
            [  9.92,  15.25, 126.24],
            [  6.64,  17.64, 123.81]]])
    >>> around(result[1], 2) #Array of axis values #doctest: +NORMALIZE_WHITESPACE
    array([[[[ 246.15,  353.26, 1031.71],
         [ 246.24,  354.25, 1031.61],
         [ 245.16,  353.35, 1031.7 ],
         [ 246.14,  353.36, 1032.71]],
         ...
         [-306.46,  510.14, 1069.26],
         [-307.13,  509.31, 1068.33],
         [-305.75,  509.12, 1068.58]]]])

    Example of returning as a tuple.

    >>> kinematics, joint_centers = calcAngles(data, start=None, end=None, vsk=vsk, splitAnglesAxis=False, formatData=False,returnjoints=True)
    >>> around(kinematics[0][0], 2)
    -0.46
    >>> around(joint_centers[0]['Pelvis'], 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 246.15,  353.26, 1031.71])

    Example without returning joints.

    >>> kinematics = calcAngles(data, vsk=vsk, splitAnglesAxis=False, formatData=False,returnjoints=False)
    >>> around(kinematics[0][0], 2)
    -0.46
    """

    start=0
    end=len(data)
    vsk=None
    returnangles=True
    returnaxis=True
    returnjoints=False
    splitAnglesAxis=True
    formatData=True

    #modified to work between python 2 and 3
    # used to rely on .has_key()
    if 'start' in kargs and kargs['start']!=None:
        start=kargs['start']
        if start <0 and start!=None:
            raise Exception("Start can not be negative")
    if 'end' in kargs and kargs['end']!=None:
        end=kargs['end']
        if start>end:
            raise Exception("Start can not be larger than end")
        if end>len(data):
            raise Exception("Range cannot be larger than data length")
    if 'frame' in kargs:
        start=kargs['frame']
        end=kargs['frame']+1
    if 'vsk' in kargs:
        vsk=kargs['vsk']
    if 'angles' in kargs:
        returnangles=kargs['angles']
    if 'axis' in kargs:
        returnaxis=kargs['axis']
    if 'splitAnglesAxis' in kargs:
        splitAnglesAxis=kargs['splitAnglesAxis']
    if 'formatData' in kargs:
        formatData=kargs['formatData']
    if 'returnjoints' in kargs:
        returnjoints=kargs['returnjoints']

    r=None
    r,jcs=Calc(start,end,data,vsk)

    if formatData==True:
        r=np.transpose(r)
        angles=r[SJA:EJA]
        axis=r[SA:EA]
        angles=np.transpose(angles)
        axis=np.transpose(axis)

        s=np.shape(angles)
        if pyver == 2:
            angles=np.reshape(angles,(s[0],s[1]/3,3))
        else:
            angles=np.reshape(angles,(s[0],s[1]//3,3))

        s=np.shape(axis)
        if pyver == 2:
            axis=np.reshape(axis,(s[0],s[1]/12,4,3))
        else:
            axis=np.reshape(axis,(s[0],s[1]//12,4,3))

        return [angles,axis]

    if splitAnglesAxis==True:
        r=np.transpose(r)
        angles=r[SJA:EJA]
        axis=r[SA:EA]
        if returnangles==True and returnaxis==True:
            return [angles,axis]
        elif returnangles==True and returnaxis==False:
            return angles
        else:
            return axis
    if returnjoints==False:
        return r
    else:
        return r,jcs

def Calc(start,end,data,vsk):
    """Calculates angles and joint values for marker data in a given range

    This function is a wrapper around `calcFrames`. It calls `calcFrames`
    with the given `data` and `vsk` inputs starting at index `start` and
    ending at index `end` in `data`.

    Parameters
    ----------
    start : int
        Start index for the range of frames in `data` to calculate
    end : int
        End index for the range of frames in `data` to calculate. The data
        at index `end` is not included.
    data : array of dict or array
        List of xyz coordinates of marker positions in a frame. Each
        coordinate is a dict where the key is the marker name and the
        value is a 3 element array of its xyz coordinate. Can also pass
        as an array of `[labels, data]`, where labels is a list of
        marker names and data is list of corresponding xyz coordinates.
    vsk : dict or array
        Dictionary containing subject measurement values, or array of
        labels and data `[labels, data]`.

    Returns
    -------
    angles, jcs : tuple
        `angles` is an array of the joint angle values. `jcs` is an array
        of joint center locations. Indices correspond to frames in the
        trial.

    Examples
    --------
    First, we load motion capture data from Sample_Static.c3d
    and subject measurement values from Sample_SM.vsk in
    /SampleData/ROM/.

    >>> from numpy import around
    >>> from .pycgmIO import loadC3D, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .pyCGM_Helpers import getfilenames
    >>> filenames = getfilenames(x=2) #x=2 loads sample data from Sample_Data/ROM
    >>> c3dFile = filenames[1]
    >>> vskFile = filenames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> vskData = loadVSK(vskFile, False)
    >>> vsk = getStatic(data,vskData,flat_foot=False)

    A start value of 0 and an end value of 3 indicates that we want
    to calculate angles for frames 0-2.

    >>> start = 0
    >>> end = 3
    >>> angles, jcs = Calc(start, end, data, vsk)
    >>> around(angles[0][0], 2) #Frame 0
    -0.46
    >>> around(angles[1][0], 2) #Frame 1
    -0.46
    >>> around(angles[2][0], 2) #Frame 2
    -0.46

    >>> around(jcs[0]['Pelvis'], 2)
    array([ 246.15,  353.26, 1031.71])
    >>> around(jcs[1]['Pelvis'], 2)
    array([ 246.16,  353.27, 1031.72])
    """
    d=data[start:end]
    angles,jcs=calcFrames(d,vsk)

    return angles,jcs


def calcFrames(data,vsk):
    """Calculates angles and joint values for given marker data

    Parameters
    ----------
    data : array of dict or array
        List of xyz coordinates of marker positions in a frame. Each
        coordinate is a dict where the key is the marker name and the
        value is a 3 element array of its xyz coordinate. Can also pass
        as a 2 element array of `[labels, data]`, where `labels` is a list of
        marker names and `data` is list of corresponding xyz coordinates.
    vsk : dict or array
        Dictionary containing subject measurement values, or array of labels
        and data `[labels, data]`.

    Returns
    -------
    angles, joints : tuple
        `angles` is an array of the joint angle values. `joints` is an array
        of joint center locations. Indices correspond to frames in the
        trial.

    Examples
    --------
    First, we load motion capture data from Sample_Static.c3d
    and subject measurement values from Sample_SM.vsk in
    /SampleData/ROM/.

    >>> from numpy import around
    >>> from .pycgmIO import loadC3D, loadVSK
    >>> from .pycgmStatic import getStatic
    >>> from .pyCGM_Helpers import getfilenames
    >>> filenames = getfilenames(x=2)
    >>> c3dFile = filenames[1]
    >>> vskFile = filenames[2]
    >>> result = loadC3D(c3dFile)
    >>> data = result[0]
    >>> vskData = loadVSK(vskFile, False)
    >>> vsk = getStatic(data,vskData,flat_foot=False)
    >>> angles, joints = calcFrames(data, vsk)
    >>> around(angles[0][0], 2)
    -0.46
    >>> around(joints[0]['Pelvis'], 2) #doctest: +NORMALIZE_WHITESPACE
    array([ 246.15,  353.26, 1031.71])
    """
    angles=[]
    joints=[] #added this here for normal data
    if type(data[0])!=type({}):
        data=createMotionDataDict(data[0], data[1])
    if type(vsk)!=type({}):
        vsk=createVskDataDict(vsk[0], vsk[1])

    #just accept that the data is missing
    for frame in data:
        angle,jcs = JointAngleCalc(frame,vsk)
        angles.append(angle)
        joints.append(jcs)
    return angles, joints
