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
# Input and output of pycgm functions

import sys
if sys.version_info[0]==2:
    import c3d
    pyver = 2
    print("Using python 2 c3d loader")

else:
    import c3dpy3 as c3d
    pyver = 3
    print("Using python 3 c3d loader - c3dpy3")
    
from math import *
import numpy as np
import xml.etree.ElementTree as ET
from pyCGM import *
import os
import errno

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
    if pyver == 2:
        labels=motiondata[0].keys()
        data=np.zeros((len(motiondata),len(labels),3))
        counter=0
        for md in motiondata:
            data[counter]=np.asarray(md.values())
            counter+=1
        return labels,data
    if pyver == 3:
        labels=list(motiondata[0].keys())
        data=np.zeros((len(motiondata),len(labels),3))
        counter=0
        for md in motiondata:
            data[counter]=np.asarray(list(md.values()))
            counter+=1
        return labels,data

def createVskDataDict(labels,data):
	vsk={}
	for key,data in zip(labels,data):
		vsk[key]=data
	return vsk

def splitVskDataDict(vsk):
    if pyver == 2: return vsk.keys(),np.asarray(vsk.values())
    if pyver == 3: return list(vsk.keys()),np.asarray(list(vsk.values()))
        
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
        if pyver == 2: row=zip(row[0::3],row[1::3],row[2::3])
        if pyver == 3: row=list(zip(row[0::3],row[1::3],row[2::3]))
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
        if pyver == 2:
            freq=np.float64(split_line(fh.next())[0])
            labels=split_line(fh.next())[1::3]
            fields=split_line(fh.next())
        elif pyver == 3:
            freq=np.float64(split_line(next(fh))[0])
            labels=split_line(next(fh))[1::3]
            fields=split_line(next(fh))
        delimiter = asbytes(delimiter)
        rows=[]
        rowsUnlabeled=[]
        if pyver == 2: first_line=fh.next()
        elif pyver == 3: first_line=next(fh)
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
            if pyver == 2: rows=[fh.next(),fh.next(),fh.next()]
            if pyver == 3: rows=[next(fh),next(fh),next(fh)]
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
        
        print(filename)
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
        if 'delimiter' in kargs:
                delimiter=kargs['delimiter']
        if 'angles' in kargs:
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

        if 'axis' in kargs:
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
                print(np.shape(data))
                dataFilter=np.transpose(data)
                dataFilter=dataFilter[SA:EA]
                dataFilter=np.transpose(dataFilter)
                print(np.shape(dataFilter))
                print(filterData)
                filterData=[i-SA for i in filterData]
                print(filterData)
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
        if pyver == 2:
            xyz="frame num,"+"X,Y,Z,"*(len(dataFilter[0])/3)
        else:
            xyz="frame num,"+"X,Y,Z,"*(len(dataFilter[0])//3)
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
    if pyver == 2:
        labels = motionData[0].keys()
        values = []
        for i in range(len(motionData)):
            values.append(np.asarray(motionData[i].values()))
            
        return values,labels
        
    if pyver == 3:
        labels = list(motionData[0].keys())
        values = []
        for i in range(len(motionData)):
            values.append(np.asarray(list(motionData[i].values())))
            
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
            