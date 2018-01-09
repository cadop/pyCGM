# Input and output of pycgm functions

import c3d
from math import *
import numpy as np
import xml.etree.ElementTree as ET
from pyCGM import *
import os
import errno
import mmap
import json
import platform
import tempfile

#Used to split the arrays with angles and axis
#Start Joint Angles
SJA=0
#End Joint Angles
EJA=SJA+19*3
#Start Axis
SA=EJA
#End Axis
EA=SA+72*3

def createMotionDataDict(labels,data):
	motiondata = []
	for frame in data:
		mydict={}
		for label,xyz in zip(labels,frame):
			l=str(label.rstrip())
			mydict[l] = xyz
		motiondata.append(mydict)
	return motiondata

def splitMotionDataDict(motiondata):
	labels=motiondata[0].keys()
	data=np.zeros((len(motiondata),len(labels),3))
	counter=0
	for md in motiondata:
		data[counter]=np.asarray(md.values())
		counter+=1
	return labels,data

def createVskDataDict(labels,data):
	vsk={}
	for key,data in zip(labels,data):
		vsk[key]=data
	return vsk

def splitVskDataDict(vsk):
	return vsk.keys(),np.asarray(vsk.values())

def writeToMem(motiondata,static,vsk,counter,namePrefix="pycgm"):
	#Get the binary from the obj
	# p=pickle.dumps(obj)
	motiondata_keys=json.dumps(motiondata[0])
	motiondata_keys_size=len(motiondata_keys)

	motiondata_data=motiondata[1].dumps()
	motiondata_data_size=len(motiondata_data)

	#Do this temporariliy because static should be gone
	static = np.asarray([[0,0,0],[0,0,0]])
	static_data_size=0
	static_data=0
	if type(static_data)!=type(None):
		static_data=static.dumps()
		static_data_size=len(static_data)

	vsk_keys=0
	vsk_keys_size=0
	vsk_data=0
	vsk_data_size=0
	if type(vsk)!=type(None):
		vsk_keys=json.dumps(vsk[0])
		vsk_keys_size=len(vsk_keys)
		vsk_data=vsk[1].dumps()
		vsk_data_size=len(vsk_data)

	#Get the size of the object to alocate in memory
	# objsize=sys.getsizeof(p)
	# objsize=len(p)
	objsize=motiondata_keys_size+motiondata_data_size+static_data_size+vsk_keys_size+vsk_data_size
	#Calculate the number of blocks in memory that need to be allocated
	blocks=int(math.ceil(float(objsize)/mmap.PAGESIZE))
	memsize=blocks*mmap.PAGESIZE
	#Create a unique name
	if platform.system()=='Linux':
		filename=tempfile.gettempdir()+'/'+namePrefix+str(counter)+str(os.getpid())
	else:
		filename=tempfile.gettempdir()+'\\'+namePrefix+str(counter)+str(os.getpid())
	# filename='./'+namePrefix+str(counter)+str(os.getpid())
	#File descriptor for mmap
	fd = os.open(filename, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
	#Make sure the memory is allocated
	assert os.write(fd, '\x00' * memsize) == memsize
	#Map the file to memory
	buff = None
	if platform.system()=='Linux':
		buf = mmap.mmap(fd, memsize, mmap.MAP_SHARED,access=mmap.ACCESS_WRITE)
	else:
		buf = mmap.mmap(fd, memsize,access=mmap.ACCESS_WRITE)
	#Put the data
	buf.seek(0)
	buf.write(motiondata_keys)
	buf.write(motiondata_data)
	if type(static_data)!=type(None):
		buf.write(static_data)
	if type(vsk)!=type(None):
		buf.write(vsk_keys)
		buf.write(vsk_data)
	buf.seek(0)
	buf.close()
	os.close(fd)
	header={'filename':filename,'memsize':memsize,'objsize':
		[motiondata_keys_size,motiondata_data_size,static_data_size,vsk_keys_size,vsk_data_size]}
	return [header]

def readFromMem(header):
	header=header[0]
	temp=''
	fd=os.open(header['filename'], os.O_RDONLY)
	buf=mmap.mmap(fd, header['memsize'],access=mmap.ACCESS_READ)

	motiondata_keys=json.loads(buf.read(header['objsize'][0]))
	motiondata_data=np.loads(buf.read(header['objsize'][1]))
	static_data=None
	if header['objsize'][2]!=0:
		static_data=np.loads(buf.read(header['objsize'][2]))
	vsk=None
	if header['objsize'][3]!=0 and header['objsize'][4]!=0:
		vsk_keys=json.loads(buf.read(header['objsize'][3]))
		vsk_data=np.loads(buf.read(header['objsize'][4]))
		vsk=createVskDataDict(vsk_keys,vsk_data)

	motiondata=createMotionDataDict(motiondata_keys,motiondata_data)
	try:
		os.remove(header['filename'])
	except Exception, e:
		pass
	return [motiondata,vsk]
def loadC3D(filename):
    #Calls the py c3d file
    reader = c3d.Reader(open(filename, 'rb'))

    labels = reader.get('POINT:LABELS').string_array
    mydict = {}
    mydictunlabeled ={}
    data = []
    dataunlabeled = []
    prog_val = 1
    counter = 0
    data_length = reader.last_frame() - reader.first_frame()
    markers=[str(label.rstrip()) for label in labels]
    
    for frame_no, points, analog in reader.read_frames(True,True):
        for label, point in zip(markers, points):
            #Create a dictionary with format LFHDX: 123 
            if label[0]=='*':
                if point[0]!=np.nan:
                    mydictunlabeled[label]=point
            else:
                mydict[label] = point
            
        data.append(mydict)
        dataunlabeled.append(mydictunlabeled)
        mydict = {}
    return [data,dataunlabeled,markers]

def loadCSV(filename):
    if filename == '':
        self.returnedData.emit(None)
    import numpy as np
    from numpy.compat import asbytes
    fh=file(filename,'r')
    fh=iter(fh)
    delimiter=','

    def rowToDict(row,labels):
        dic={}
        unlabeleddic={}
        row=zip(row[0::3],row[1::3],row[2::3])
        empty=np.asarray([np.nan,np.nan,np.nan],dtype=np.float64)
        for coordinates,label in zip(row,labels):
            #unlabeled data goes to a different dictionary
            if label[0]=="*":
                try:
                    unlabeleddic[label]=np.float64(coordinates)
                except:
                    pass
            else:
                try:
                    dic[label]=np.float64(coordinates)
                except:
                    #Missing data from labeled marker is NaN
                    dic[label]=empty.copy()
        return dic,unlabeleddic

    def split_line(line):
        line = asbytes(line).strip(asbytes('\r\n'))
        if line:
            return line.split(delimiter)
        else:
            return []

    def parseTrajectories(fh,framesNumber):
        delimiter=','
        freq=np.float64(split_line(fh.next())[0])
        labels=split_line(fh.next())[1::3]
        fields=split_line(fh.next())
        delimiter = asbytes(delimiter)
        rows=[]
        rowsUnlabeled=[]
        first_line=fh.next()
        first_elements=split_line(first_line)[1:]
        colunsNum=len(first_elements)
        first_elements,first_elements_unlabeled=rowToDict(first_elements,labels)
        rows.append(first_elements)
        rowsUnlabeled.append(first_elements_unlabeled)

        for row in fh:
            row=split_line(row)[1:]
            if len(row)!=colunsNum:
                break
            elements,unlabeled_elements=rowToDict(row,labels)
            rows.append(elements)
            rowsUnlabeled.append(unlabeled_elements)

        return labels,rows,rowsUnlabeled,freq

    ###############################################
    ### Find the trajectories
    framesNumber=0
    for i in fh:
        if i.startswith("TRAJECTORIES"):
            #First elements with freq,labels,fields
            rows=[fh.next(),fh.next(),fh.next()]
            for j in fh:
                if j.startswith("\r\n"):
                    break
                framesNumber=framesNumber+1
                rows.append(j)
            break
    rows=iter(rows)
    labels,motionData,unlabeledMotionData,freq=parseTrajectories(rows,framesNumber)
    
    return [motionData,unlabeledMotionData,labels]

def loadData(filename,rawData=True):
        
        print filename
        if str(filename).endswith('.c3d'):
                return loadC3D(filename)[0]
                
        elif str(filename).endswith('.csv'):
                return loadCSV(filename)[0]		

def writeResult(data,filename,**kargs):
        """
        Writes the result of the calculation into a csv file 
        @param data Motion Data as a matrix of frames as rows
        @param filename Name to save the csv
        @param kargs
                delimiter Delimiter for the csv. By default it's using ','
                angles True or false to save angles. Or a list of angles to save
                axis True of false to save axis. Or a list of axis to save
        Examples
        #save angles and axis
        writeResultNumPy(result,"outputfile0.csv")
        #save 'R Hip' angles 'L Foot' and all the axis
        writeResultNumPy(result,"outputfile1.csv",angles=['R Hip','L Foot'])
        #save only axis "R ANKZ","L ANKO","L ANKX"
        writeResultNumPy(result,"outputfile4.csv",angles=False,axis=["R ANKZ","L ANKO","L ANKX"])
        #save only angles
        writeResultNumPy(result,"outputfile6.csv",axis=False)
        """
        labelsAngs =['Pelvis','R Hip','L Hip','R Knee','L Knee','R Ankle',
                                'L Ankle','R Foot','L Foot',
                                'Head','Thorax','Neck','Spine','R Shoulder','L Shoulder',
                                'R Elbow','L Elbow','R Wrist','L Wrist']

        labelsAxis =["PELO","PELX","PELY","PELZ","HIPO","HIPX","HIPY","HIPZ","R KNEO","R KNEX","R KNEY","R KNEZ","L KNEO","L KNEX","L KNEY","L KNEZ","R ANKO","R ANKX","R ANKY","R ANKZ","L ANKO","L ANKX","L ANKY","L ANKZ","R FOOO","R FOOX","R FOOY","R FOOZ","L FOOO","L FOOX","L FOOY","L FOOZ","HEAO","HEAX","HEAY","HEAZ","THOO","THOX","THOY","THOZ","R CLAO","R CLAX","R CLAY","R CLAZ","L CLAO","L CLAX","L CLAY","L CLAZ","R HUMO","R HUMX","R HUMY","R HUMZ","L HUMO","L HUMX","L HUMY","L HUMZ","R RADO","R RADX","R RADY","R RADZ","L RADO","L RADX","L RADY","L RADZ","R HANO","R HANX","R HANY","R HANZ","L HANO","L HANX","L HANY","L HANZ"]

        outputAngs=True
        outputAxis=True
        dataFilter=None
        delimiter=","
        filterData=[]
        if kargs.has_key('delimiter'):
                delimiter=kargs['delimiter']
        if kargs.has_key('angles'):
                if kargs['angles']==True:
                        outputAngs=True
                elif kargs['angles']==False:
                        outputAngs=False
                        labelsAngs=[]
                elif isinstance(kargs['angles'], (list, tuple)):
                        filterData=[i*3 for i in range(len(labelsAngs)) if labelsAngs[i] not in kargs['angles']]
                        if len(filterData)==0:
                                outputAngs=False
                        labelsAngs=[i for i in labelsAngs if i in kargs['angles']]

        if kargs.has_key('axis'):
                if kargs['axis']==True:
                        outputAxis=True
                elif kargs['axis']==False:
                        outputAxis=False
                        labelsAxis=[]
                elif isinstance(kargs['axis'], (list, tuple)):
                        filteraxis=[i*3+SA for i in range(len(labelsAxis)) if labelsAxis[i] not in kargs['axis']]
                        filterData=filterData+filteraxis
                        if len(filteraxis)==0:
                                outputAxis=False
                        labelsAxis=[i for i in labelsAxis if i in kargs['axis']]

        if len(filterData)>0:
                filterData=np.repeat(filterData,3)
                filterData[1::3]=filterData[1::3]+1
                filterData[2::3]=filterData[2::3]+2

        if outputAngs==outputAxis==False:
                return
        elif outputAngs==False:
                print np.shape(data)
                dataFilter=np.transpose(data)
                dataFilter=dataFilter[SA:EA]
                dataFilter=np.transpose(dataFilter)
                print np.shape(dataFilter)
                print filterData
                filterData=[i-SA for i in filterData]
                print filterData
        elif outputAxis==False:
                dataFilter=np.transpose(data)
                dataFilter=dataFilter[SJA:EJA]
                dataFilter=np.transpose(dataFilter)

        if len(filterData)>0:
                if dataFilter==None:
                        dataFilter=np.delete(data, filterData, 1)
                else:
                        dataFilter=np.delete(dataFilter, filterData, 1)
        if dataFilter==None:
                dataFilter=data
        header=","
        headerAngs=["Joint Angle,,,",",,,x = flexion/extension angle",",,,y= abudction/adduction angle",",,,z = external/internal rotation angle",",,,"]
        headerAxis=["Joint Coordinate",",,,###O = Origin",",,,###X = X axis orientation",",,,###Y = Y axis orientation",",,,###Z = Z axis orientation"]
        for angs,axis in zip(headerAngs,headerAxis):
                if outputAngs==True:
                        header=header+angs+",,,"*(len(labelsAngs)-1)
                if outputAxis==True:
                        header=header+axis+",,,"*(len(labelsAxis)-1)
                header=header+"\n"
        labels=","
        if len(labelsAngs)>0:
                labels=labels+",,,".join(labelsAngs)+",,,"
        if len(labelsAxis)>0:
                labels=labels+",,,".join(labelsAxis)
        labels=labels+"\n"
        xyz="frame num,"+"X,Y,Z,"*(len(dataFilter[0])/3)
        header=header+labels+xyz
        #Creates the frame numbers
        frames=np.arange(len(dataFilter),dtype=dataFilter[0].dtype)
        #Put the frame numbers in the first dimension of the data
        dataFilter=np.column_stack((frames,dataFilter))
        start = 1500
        end = 3600
        #dataFilter = dataFilter[start:]
        np.savetxt(filename+'.csv', dataFilter, delimiter=delimiter,header=header,fmt="%.15f")
        #np.savetxt(filename, dataFilter, delimiter=delimiter,fmt="%.15f")
        #np.savez_compressed(filename,dataFilter)

def loadVSK(filename):
        #Check if the filename is valid
        #if not, return None
        if filename == '':
                return None

        # Create Dictionary to store values from VSK file
        viconVSK = {}
        vskMarkers = []
        
        #Create an XML tree from file
        tree = ET.parse(filename)
        #Get the root of the file
        # <KinematicModel>
        root = tree.getroot()
        
        # for i in range(len(root[2][0])):
        #     vskMarkers.append(root[2][0][i].get('NAME'))
        
        #Store the values of each parameter in a dictionary
        # the format is (NAME,VALUE)
        vsk_keys=[r.get('NAME') for r in root[0]]
        vsk_data = []
        for R in root[0]:
            val = (R.get('VALUE'))
            if val == None:
                val = 0
            vsk_data.append(float(val))

        #vsk_data=np.asarray([float(R.get('VALUE')) for R in root[0]])

        return [vsk_keys,vsk_data]


def splitDataDict(motionData):        
    labels = motionData[0].keys()
    values = []
    for i in range(len(motionData)):
        values.append(np.asarray(motionData[i].values()))
        
    return values,labels

def combineDataDict(values,labels):
    data = []
    tmp_dict = {}
    for i in range (len(values)):
        for j in range (len(values[i])):
            tmp_dict[labels[j]]=values[i][j]
        data.append(tmp_dict)
        tmp_dict = {}
        
    return data
        

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise