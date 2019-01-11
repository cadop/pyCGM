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


from pyCGM import *

#Used to split the arrays with angles and axis
#Start Joint Angles
SJA=0
#End Joint Angles
EJA=SJA+19*3
#Start Axis
SA=EJA
#End Axis
EA=SA+72*3

def calcAngles(data,**kargs):
	"""
	Calculates the joint angles and axis
	@param  data Motion data as a vector of dictionaries like the data in 
	marb or labels and raw data like the data from loadData function
	@param  static Static angles
	@param  Kargs 
		start   Position of the data to start the calculation
		end     Position of the data to end the calculation
		frame   Frame number if the calculation is only for one frame
		cores   Number of processes to use on the calculation
		vsk     Vsk file as a dictionary or label and data
		angles  If true it will return the angles
		axis    If true it will return the axis
		splitAnglesAxis     If true the function will return angles and axis as separete arrays. For false it will be the same array
		multiprocessing     If true it will use multiprocessing

	By default the function will calculate all the data and return angles and axis as separete arrays
	"""
	start=0
	end=len(data)
	vsk=None
	returnangles=True
	returnaxis=True
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

	r=None
	r=Calc(start,end,data,vsk)

	if formatData==True:
		r=np.transpose(r)
		angles=r[SJA:EJA]
		axis=r[SA:EA]
		angles=np.transpose(angles)
		axis=np.transpose(axis)
		s=np.shape(angles)
		angles=np.reshape(angles,(s[0],s[1]/3,3))
		s=np.shape(axis)
		axis=np.reshape(axis,(s[0],s[1]/12,4,3))
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
	else:
		return r

def Calc(start,end,data,vsk):
	d=data[start:end]
	angles=calcFrames(d,vsk)
	return angles

def calcFrames(data,vsk):
	angles=[]
	if type(data[0])!=type({}):
		data=createMotionDataDict(data[0],data[1])
	if type(vsk)!=type({}):
		vsk=createVskDataDict(vsk[0],vsk[1])

	for frame in data:
		angle = JointAngleCalc(frame,vsk)
		angles.append(angle)
	return angles
