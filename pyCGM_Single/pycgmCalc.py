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
    r = getKinetics(data, Bodymass)
    return r
    

def calcAngles(data,**kargs):
    """Calculates the joint angles and axis
    
    By default, the function will calculate all the data and return angles
    and axis as separate arrays

    Parameters
    ----------
    data : list of dict
        Joint centres in the global coordinate system. List indices correspond 
        to each frame of trial. Dict keys correspond to name of each joint centre,
        dict values are arrays ([],[],[]) of x,y,z coordinates for each joint 
        centre
    **kargs : dict of keyword arguments
        start
           Indicates which index in `data` to start the calculation
        end
           Indicates which index in `data` to end the calculation
        frame : int
            Frame number if the calculation is only for one frame
        cores : int
            Number of processes to use on the calculation
        vsk : dict
            Vsk file as a dictionary or label and data
        angles : boolean
            If true, the function will return the angles
        axis : boolean
            If true, the function will return the axis
        splitAnglesAxis : boolean
            If true, the function will return the angles and axis as
            separate arrays. If false, it will be the same array
        multiprocessing : boolean
            If true, the function will use multiprocessing
    
    Returns
    -------
        r : list of list
            List of joint angle values
        jcs : List of dict  
            List of joint center locations for each frame of trial. 

    Raises
    ------
    Exception
        If `start` is given and is negative
        If `start` is larger than `end`
        If `end` is larger than the length of `data`

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
    
    This function is a wrapper around `calcFrames`. It calls calcFrames
    with the given `data` and `vsk` inputs starting at index `start` and 
    ending at index `end` in `data`.

    Parameters
    ----------
    start : int
        Start index for the range of frames in `data` to calculate
    end : int
        End index for the range of frames in `data` to calculate
    data : list of list
        List of xyz coordinates of marker positions in a frame.
    vsk : dict
        Dictionary containing vsk file values

    Returns
    -------
    angles : list of list
        List of lists containing joint angle values
    joints : list of dict
        List of dictionaries containing joint values

    """
    d=data[start:end]
    angles,jcs=calcFrames(d,vsk)
    
    return angles,jcs

def calcFrames(data,vsk):
    """Calculates angles and joint values for given marker data
    
    Parameters
    ----------
    data : list of list
        List of xyz coordinates of marker positions in a frame. 
    vsk : dict
        Dictionary containing vsk file values
 
    Returns
    -------
    angles : list of list
        List of lists containing joint angle values
    joints : list of dict
        List of dictionaries containing joint values
    
    """
    angles=[]
    joints=[] #added this here for normal data
    if type(data[0])!=type({}):
        data=createMotionDataDict(data[0],data[1])
    if type(vsk)!=type({}):
        vsk=createVskDataDict(vsk[0],vsk[1])

    #just accept that the data is missing    
    for frame in data:
        angle,jcs = JointAngleCalc(frame,vsk)
        angles.append(angle)
        joints.append(jcs)
    return angles, joints


            
