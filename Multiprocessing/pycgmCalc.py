# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:16:44 2015

@author: cadop
"""

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

	By default the function will calculate all the data using single processing and return angles and axis as separete arrays
	"""
	start=0
	end=len(data)
	cores=None
	vsk=None
	returnangles=True
	returnaxis=True
	multiprocessing=True
	splitAnglesAxis=True
	formatData=True

	if kargs.has_key('start') and kargs['start']!=None:
		start=kargs['start']
		if start <0 and start!=None:
			raise Exception("Start can not be negative")
	if kargs.has_key('end') and kargs['end']!=None:
		end=kargs['end']
		if start>end:
			raise Exception("Start can not be larger than end")
		if end>len(data):
			raise Exception("Range cannot be larger than data length")
	if kargs.has_key('frame'):
		start=kargs['frame']
		end=kargs['frame']+1
	if kargs.has_key('cores') and kargs['cores']!=None:
		cores=kargs['cores']
		if cores<1:
			raise Exception("Number of cores must be positive")
	if kargs.has_key('multiprocessing') and kargs['multiprocessing']==True:
		multiprocessing=kargs['multiprocessing']
	else:
		multiprocessing=False
		cores=None
	if kargs.has_key('vsk'):
		vsk=kargs['vsk']
	if kargs.has_key('angles'):
		returnangles=kargs['angles']
	if kargs.has_key('axis'):
		returnaxis=kargs['axis']
	if kargs.has_key('splitAnglesAxis'):
		splitAnglesAxis=kargs['splitAnglesAxis']
	if kargs.has_key('formatData'):
		formatData=kargs['formatData']

	r=None
	if multiprocessing==False and cores==None:
		r=singleCalc(start,end,data,vsk)
	else:
		#NEED TO FIX HERE BECAUSE FOR MULTIPROCESSING THE VSK MUST TO EXIST
		if type(data[0])==type({}):
			data=splitMotionDataDict(data)
		if type(vsk)==type({}):
			vsk=splitVskDataDict(vsk)
		print 'multi: ',cores   
		r=multiCalc(start,end,data,cores,vsk)

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


def multiCalc(start,end,data,nprocs,vsk):

	if nprocs == None:
		nprocs = multiprocessing.cpu_count()-1

	results = multiprocessing.Queue()
	labels=data[0]
	motiondata=data[1]
	l=len(motiondata)/nprocs
	procs=[]
	headers=[]
	for i in range(nprocs):
		start=i*l
		end=(i+1)*l
		if i==nprocs-1:
			end=len(motiondata)
		#Do this temporarily, then remove static totally
		static=0
		header=writeToMem((labels,motiondata[start:end]),static,vsk,i)
		headers.append(header)
		ptemp=multiprocessing.Process(target=calcFramesMulti, args=(header,results,i))
		ptemp.daemon=True
		ptemp.start()
		procs.append(ptemp)

	results=[results.get() for i in range(nprocs)]
	results=sorted(results, key=lambda r: r[0])
	results=[r[1] for r in results]

	for ptemp in procs:
		ptemp.join() 
	angles = []
	for i in results:
		angles=angles+i

	for header in headers:
		try:
			os.remove(header[0]['filename'])
		except Exception, e:
			pass
	return angles

def calcFramesMulti(header,result,index):
	data,vsk=readFromMem(header)
	angles=[]

	for frame in data:
		angle = JointAngleCalc(frame,vsk)
		angles.append(angle)
	result.put([index,angles])
	sys.exit()

def singleCalc(start,end,data,vsk):

	d=data[start:end]
	angles=calcFramesSing(d,vsk)
	return angles

def calcFramesSing(data,vsk):
	angles=[]
	if type(data[0])!=type({}):
		data=createMotionDataDict(data[0],data[1])
	if type(vsk)!=type({}):
		vsk=createVskDataDict(vsk[0],vsk[1])

	for frame in data:
		angle = JointAngleCalc(frame,vsk)
		angles.append(angle)
	return angles
 
#Each rank should run this function and return the array of data
# motiondata should be passed individually with scatter
# static and vsk should use bcast
def calcFramesMPI(motiondata,vsk):
    angles=[]
    
    for frame in motiondata:
        angle = JointAngleCalc(frame,vsk)
        angles.append(angle)
    #should be send to master rank
    return angles
