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
from .pyCGM import *
if sys.version_info[0]==2:
    import c3d
    pyver = 2
    print("Using python 2 c3d loader")

else:
    from . import c3dpy3 as c3d
    pyver = 3
    print("Using python 3 c3d loader - c3dpy3")

try:
    from ezc3d import c3d as ezc
    useEZC3D = True
    print("EZC3D Found, using instead of Python c3d")
except:
    useEZC3D = False

from math import *
import numpy as np
import xml.etree.ElementTree as ET
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
	"""Creates an array of motion capture data given labels and data.

	Parameters
	----------
	labels : array
	    List of marker position names. 
	data : array
	    List of xyz coordinates corresponding to the marker names in `labels`. 
	    Indices of `data` correspond to frames in the trial.

	Returns
	-------
	motiondata : array
	    List of dict. Indices of `motiondata` correspond to frames
	    in the trial. Keys in the dictionary are marker names and 
	    values are xyz coordinates of the corresponding marker.
    
	Examples
	--------
	This example uses a loop and ``numpy.array_equal`` to test the equality
	of individual dictionary elements since python does not guarantee 
	the order of dictionary elements. 

	Example for three markers and two frames of trial.

	>>> from numpy import array, array_equal
	>>> labels = ['LFHD', 'RFHD', 'LBHD']
	>>> data = [[array([184.55160796, 409.68716101, 1721.34289625]), 
	...          array([325.82985131, 402.55452959, 1722.49816649]), 
	...          array([197.8621642 , 251.28892152, 1696.90197756])],
	...         [array([185.55160796, 408.68716101, 1722.34289625]), 
	...          array([326.82985131, 403.55452959, 1723.49816649]), 
	...          array([198.8621642 , 252.28892152, 1697.90197756])]]
	>>> result = createMotionDataDict(labels, data)
	>>> expected = [{'RFHD': array([325.82985131, 402.55452959, 1722.49816649]), 
	...              'LBHD': array([197.8621642 , 251.28892152, 1696.90197756]),
	...              'LFHD': array([184.55160796, 409.68716101, 1721.34289625])},
	...             {'RFHD': array([326.82985131, 403.55452959, 1723.49816649]), 
	...              'LBHD': array([198.8621642 , 252.28892152, 1697.90197756]),
	...              'LFHD': array([185.55160796, 408.68716101, 1722.34289625])}]

	>>> flag = True #False if any values are not equal
	>>> for i in range(len(result)):
	...     for key in result[i]:
	...         if (not array_equal(result[i][key], expected[i][key])):
	...             flag = False
	>>> flag
	True
	"""
	motiondata = []
	for frame in data:
		mydict={}
		for label,xyz in zip(labels,frame):
			l=str(label.rstrip())
			mydict[l] = xyz
		motiondata.append(mydict)
	return motiondata

def splitMotionDataDict(motiondata):
    """Splits an array of motion capture data into separate labels and data.

    Parameters
    ----------
    motiondata : array
        List of dict. Indices of `motiondata` correspond to frames
        in the trial. Keys in the dictionary are marker names and 
        values are xyz coordinates of the corresponding marker.

    Returns
    -------
    labels, data : tuple
        `labels` is a list of marker position names from the dictionary
        keys in `motiondata`. `data` is a list of xyz coordinate 
        positions corresponding to the marker names in `labels`. 
        Indices of `data` correspond to frames in the trial.

    Examples
    --------
    Example for three markers and two frames of trial.

    >>> from numpy import array
    >>> motiondata = [{'RFHD': array([325.82985131, 402.55452959, 1722.49816649]), 
    ...                'LFHD': array([184.55160796, 409.68716101, 1721.34289625]),
    ...                'LBHD': array([197.8621642 , 251.28892152, 1696.90197756])},
    ...               {'RFHD': array([326.82985131, 403.55452959, 1723.49816649]), 
    ...                'LFHD': array([185.55160796, 408.68716101, 1722.34289625]),
    ...                'LBHD': array([198.8621642 , 252.28892152, 1697.90197756])}]
    >>> labels, data = splitMotionDataDict(motiondata)
    >>> labels
    ['RFHD', 'LFHD', 'LBHD']
    >>> data #doctest: +NORMALIZE_WHITESPACE
    array([[[ 325.82985131,  402.55452959, 1722.49816649],
            [ 184.55160796,  409.68716101, 1721.34289625],
            [ 197.8621642 ,  251.28892152, 1696.90197756]],
           [[ 326.82985131,  403.55452959, 1723.49816649],
            [ 185.55160796,  408.68716101, 1722.34289625],
            [ 198.8621642 ,  252.28892152, 1697.90197756]]])
    """
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
	"""Creates a dictionary of vsk file values from labels and data.

	Parameters
	----------
	labels : array
	    List of label names for vsk file values.
	data : array
	    List of subject measurement values corresponding to the label 
            names in `labels`.

	Returns
	-------
	vsk : dict
	    Dictionary of vsk file values. Dictionary keys correspond to 
	    names in `labels` and dictionary values correspond to values in 
	    `data`.

	Examples
	--------
	This example tests for dictionary equality through python instead of 
	doctest since python does not guarantee the order in which dictionary 
	elements are printed.

	>>> labels = ['MeanLegLength', 'LeftKneeWidth', 'RightAnkleWidth']
	>>> data = [940.0, 105.0, 70.0]
	>>> res = createVskDataDict(labels, data)
	>>> res == {'MeanLegLength':940.0, 'LeftKneeWidth':105.0, 'RightAnkleWidth':70.0}
	True
	"""
	vsk={}
	for key,data in zip(labels,data):
		vsk[key]=data
	return vsk

def splitVskDataDict(vsk):
    """Splits a dictionary of vsk file values into labels and data arrays
   
    Parameters
    ----------
    vsk : dict
        dictionary of vsk file values. Dictionary keys correspond to 
        names in `labels` and dictionary values correspond to values in 
        `data`.

    Returns
    -------
    labels, data : tuple
        `labels` is a list of label names for vsk file values. `data` 
        is a numpy array holding the corresponding values.

    Examples
    --------
    >>> from numpy import array, array_equal #used to compare numpy arrays
    >>> import sys
    >>> vsk = {'MeanLegLength':940.0, 'LeftKneeWidth':105.0, 'RightAnkleWidth':70.0}
    >>> labels, data = splitVskDataDict(vsk)
    >>> flag = True #False if any values do not match
    >>> for i in range(len(labels)):
    ...     if (vsk[labels[i]] != data[i]):
    ...         flag = False
    >>> flag
    True
    """
    if pyver == 2: return vsk.keys(),np.asarray(vsk.values())
    if pyver == 3: return list(vsk.keys()),np.asarray(list(vsk.values()))

def markerKeys():
    """A list of marker names.

    Returns
    -------
    marker_keys : array
        List of marker names.

    Examples
    --------
    >>> markerKeys() #doctest: +NORMALIZE_WHITESPACE
    ['RASI', 'LASI', 'RPSI', 'LPSI', 'RTHI', 'LTHI', 'RKNE', 'LKNE', 'RTIB',
     'LTIB', 'RANK', 'LANK', 'RTOE', 'LTOE', 'LFHD', 'RFHD', 'LBHD', 'RBHD', 
     'RHEE', 'LHEE', 'CLAV', 'C7', 'STRN', 'T10', 'RSHO', 'LSHO', 'RELB', 'LELB', 
     'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']
    """
    marker_keys = ['RASI','LASI','RPSI','LPSI','RTHI','LTHI','RKNE','LKNE','RTIB',
               'LTIB','RANK','LANK','RTOE','LTOE','LFHD','RFHD','LBHD','RBHD',
               'RHEE','LHEE','CLAV','C7','STRN','T10','RSHO','LSHO','RELB','LELB',
               'RWRA','RWRB','LWRA','LWRB','RFIN','LFIN']
    return marker_keys

def loadEZC3D(filename):
    """Use c3dez to load a c3d file.

    Parameters
    ----------
    filename : str
        Path to the c3d file to be loaded.

    Returns
    -------
    [data, None, None] : array
        `data` is the array representation of the loaded c3d file.
    """
    #Relative import mod for python 2 and 3
    try: from . import c3dez
    except: import c3dez

    dataclass = c3dez.C3DData(None, filename)
    data = dataAsArray(dataclass.Data['Markers'])
    return [data,None,None]

def loadC3D(filename):
    """Open and load a C3D file of motion capture data

    Keys in the returned data dictionaries are marker names, and 
    the corresponding values are a numpy array with the associated
    value. ``data[marker] = array([x, y, z])``

    Parameters
    ----------
    filename : str
        File name of the C3D file to be loaded
 
    Returns
    -------
    [data, dataunlabeled, markers] : array
        `data` is a list of dict. Each dict represents one frame in 
        the trial. `dataunlabeled` contains a list of dictionaries
        of the same form as in `data`, but for unlabeled points. 
        `markers` is a list of marker names. 

    Examples
    --------
    The files 59993_Frame_Static.c3d and RoboStatic.c3d in 
    SampleData are used to test the output.

    >>> from .pyCGM_Helpers import getfilenames
    >>> from numpy import around, array
    >>> filename_59993 = getfilenames(x=1)[1]
    >>> result_59993 = loadC3D(filename_59993)
    >>> data = result_59993[0]
    >>> dataunlabeled = result_59993[1]
    >>> markers = result_59993[2]
    >>> roboFilename = getfilenames(x=3)[1]
    >>> result_roboFilename = loadC3D(roboFilename)
    >>> roboDataUnlabeled = result_roboFilename[1]

    Testing for some values from 59993_Frame_Static.c3d.

    >>> around(data[0]['RHNO'], 8) #doctest: +NORMALIZE_WHITESPACE
    array([ 555.46948242, -559.36499023, 1252.84216309])
    >>> around(data[0]['C7'], 8) #doctest: +NORMALIZE_WHITESPACE
    array([ -29.57296562, -9.34280109, 1300.86730957])
    >>> dataunlabeled[4] #doctest: +NORMALIZE_WHITESPACE  
    {'*113': array([-172.66630554,  167.2040863 , 1273.71594238]), 
     '*114': array([ 169.18231201, -227.13475037, 1264.34912109])}
    >>> markers
    ['LFHD', 'RFHD', 'LBHD', ...]

    Frame 0 in RoboStatic.c3d has no unlabeled data.

    >>> roboDataUnlabeled[0]
    {}
    """
    if useEZC3D == True:
        print("Using EZC3D")
        return loadEZC3D(filename)

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
        mydictunlabeled ={}
    return [data,dataunlabeled,markers]

def loadCSV(filename):
    """Open and load a CSV file of motion capture data.

    Keys in the returned data dictionaries are marker names, and 
    the corresponding values are a numpy array with the associated
    value. ``data[marker] = array([x, y, z])``

    Parameters
    ----------
    filename : str
        File name of the CSV file to be loaded.
 
    Returns
    -------
    [motionData, unlabeledMotionData, labels] : array
        `motionData` is a list of dict. Each dict represents one frame in 
        the trial. `unlabeledMotionData` contains a list of dictionaries
        of the same form as in `motionData`, but for unlabeled points. 
        `labels` is a list of marker names.  

    Examples
    --------
    Sample_Static.csv in SampleData is used to test the output.

    >>> filename = 'SampleData/ROM/Sample_Static.csv' 
    >>> result = loadCSV(filename)
    >>> motionData = result[0]
    >>> unlabeledMotionData = result[1]
    >>> labels = result[2]

    Testing for some values from data.

    >>> motionData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
    array([ 811.9591064, 677.3413696, 1055.390991 ])
    >>> motionData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE 
    array([ 250.765976, 165.616333, 1528.094116])
    >>> unlabeledMotionData[0] #doctest: +NORMALIZE_WHITESPACE
    {'*111': array([ 692.8970947, 423.9462585, 1240.289063 ]), 
     '*112': array([-225.5265198, 405.5791321, 1214.458618 ]), 
     '*113': array([ -82.65164185, 232.3781891 , 1361.853638 ]), 
     '*114': array([ 568.5736694, 260.4929504, 1361.799805 ])}
    >>> labels
    ['LFHD', 'RFHD', 'LBHD', ...]
    """
    if filename == '':
        self.returnedData.emit(None)
    import numpy as np
    from numpy.compat import asbytes #probably not needed

    fh = open(filename,'r')

    fh=iter(fh)
    delimiter=','

    def rowToDict(row,labels):
        """Convert a row and labels to a dictionary.

        This function is only in scope from within `loadCSV`.

        Parameters
        ----------
        row : array
            List of marker data.
        labels : array
            List of marker names.

        Returns
        -------
        dic, unlabeleddic : tuple
            `dic` is a dictionary where keys are marker names and values
            are the corresponding marker value. `unlabeleddic` holds
            all unlabeled marker values in the same format as `dic`.

        Examples
        --------
        This example uses a loop and numpy.array_equal to test the equality
        of individual dictionary elements since python does not guarantee 
        the order of dictionary elements. 

        >>> from numpy import array, array_equal
        >>> row = ['-1003.583618', '81.007614', '1522.236938', 
        ...        '-1022.270447', '-47.190071', '1519.680420', 
        ...        '-833.953979', '40.892181', '1550.325562']
        >>> labels = ['LFHD', 'RFHD', 'LBHD']
        >>> dict, unlabeleddict = rowToDict(row, labels)
        >>> expectedDict = {'LFHD': array([-1003.583618, 81.007614, 1522.236938]), 
        ...                 'RFHD': array([-1022.270447, -47.190071, 1519.68042]), 
        ...                 'LBHD': array([-833.953979, 40.892181, 1550.325562])}
        >>> unlabeleddict #No unlabeled values are expected for this example
        {}
        >>> flag = True #False if any values are not equal
        >>> for marker in dict:
        ...     if (not array_equal(dict[marker], expectedDict[marker])):
        ...         flag = False
        >>> flag
        True
        """
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
        """Split a line in a csv file into an array

        This function is only in scope from within `loadCSV`.

        Parameters
        ----------
        line : str
            String form of the line to be split

        Returns
        -------
        array
            Array form of `line`, split on the predefined delimiter ','.

        Examples
        --------
        >>> line = '-772.184937, -312.352295, 589.815308'
        >>> split_line(line)
        ['-772.184937', ' -312.352295', ' 589.815308']
        """
        if pyver == 2: line = asbytes(line).strip(asbytes('\r\n'))
        elif pyver == 3: line = line.strip('\r\n')
        if line:
            return line.split(delimiter)
        else:
            return []

    def parseTrajectories(fh,framesNumber):
        r"""Converts rows of motion capture data into a dictionary

        This function is only in scope from within `loadCSV`.

        Parameters
        ----------
        fh : list iterator object
            Iterator for rows of motion capture data. The first 3 rows
            in `fh` contain the frequency, labels, and field headers 
            respectively. All elements of the rows in `fh` are strings.
            See Examples.
        framesNumber : int
            Number of rows iterated over in `fh`.

        Returns
        -------
        labels, rows, rowsUnlabeled, freq : tuple
            `labels` is a list of marker names.
            `rows` is a list of dict of motion capture data.
            `rowsUnlabeled` is of the same type as `rows`, but for
            unlabeled data.
            `freq` is the frequency in Hz.

        Examples
        --------
        This example uses a loop and numpy.array_equal to test the equality
        of individual dictionary elements since python does not guarantee 
        the order of dictionary elements. 

        Example for 2 markers, LFHD and RFHD, and one frame of trial. 
        >>> from numpy import array, array_equal

        # Rows will hold frequency, headers, fields, and one row of data
        >>> rows = [None, None, None, None] 
        >>> rows[0] = '240.000000,Hz\n'
        >>> rows[1] = ',LFHD,,,RFHD\n'
        >>> rows[2] = 'Field #,X,Y,Z,X,Y,Z\n'
        >>> rows[3] = '1,-1003.583618,81.007614,1522.236938,-1022.270447,-47.190071,1519.680420\n'
        >>> fh = iter(rows)
        >>> framesNumber = 1 #Indicates one row of data
        >>> labels, rows, rowsUnlabeled, freq = parseTrajectories(fh, framesNumber)
        >>> labels
        ['LFHD', 'RFHD']
        >>> expectedRows = [{'LFHD': array([-1003.583618,  81.007614, 1522.236938]), 
        ...                  'RFHD': array([-1022.270447, -47.190071, 1519.68042 ])}]

        >>> flag = True #False if any values are not equal
        >>> for i in range(len(expectedRows)):
        ...     for key in rows[i]:
        ...         if (not array_equal(rows[i][key], expectedRows[i][key])):
        ...             flag = False
        >>> flag
        True
        >>> rowsUnlabeled
        [{}]
        >>> freq
        240.0
        """
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
        """Loads motion capture data from a csv or c3d file.

        Either a csv or c3d file of motion capture data can be used.
        `loadCSV` or `loadC3D` will be called accordingly.

        Parameters
        ----------
        filename : str
            Path of the csv or c3d file to be loaded.

        Returns
        -------
        data : array
            `data` is a list of dict. Each dict represents one frame in 
            the trial.

        Examples
        --------
        RoboResults.csv and RoboResults.c3d in SampleData are used to 
        test the output.

        >>> csvFile = 'SampleData/Sample_2/RoboResults.csv' 
        >>> c3dFile = 'SampleData/Sample_2/RoboStatic.c3d'
        >>> csvData = loadData(csvFile)
        SampleData/Sample_2/RoboResults.csv
        >>> c3dData = loadData(c3dFile)
        SampleData/Sample_2/RoboStatic.c3d

        Testing for some values from the loaded csv file.

        >>> csvData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
        array([-772.184937, -312.352295, 589.815308])
        >>> csvData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE 
        array([-1010.098999, 3.508968, 1336.794434])

        Testing for some values from the loaded c3d file.

        >>> c3dData[0]['RHNO'] #doctest: +NORMALIZE_WHITESPACE
        array([-259.45016479, -844.99560547, 1464.26330566])
        >>> c3dData[0]['C7'] #doctest: +NORMALIZE_WHITESPACE
        array([-2.20681717e+02, -1.07236075e+00, 1.45551550e+03])    
        """        
        print(filename)
        if str(filename).endswith('.c3d'):
                
                data = loadC3D(filename)[0]
                #add any missing keys
                keys = markerKeys()
                for frame in data:
                    for key in keys:
                        frame.setdefault(key,[np.nan,np.nan,np.nan])
                return data
                
        elif str(filename).endswith('.csv'):
                return loadCSV(filename)[0]		

def dataAsArray(data):
    """Converts a dictionary of markers with xyz data to an array
    of dictionaries. 

    Assumes all markers have the same length of data.

    Parameters
    ----------
    data : dict
        Dictionary of marker data. Keys are marker names. Values are 
        arrays of 3 elements, each of which is an array of x, y, and z
        coordinate values respectively. ``data[marker] = array([x, y, z])``

    Returns
    -------
    dataArray : array
        List of dictionaries.

    Examples
    --------
    This example uses a loop and ``numpy.array_equal`` to test the equality
    of individual dictionary elements since python does not guarantee 
    the order of dictionary elements. 

    Example for motion capture data for 3 markers, each with data for 
    one frame of trial.

    >>> from numpy import array, array_equal
    >>> data = {'RFHD': [array([325.82985131]), array([402.55452959]), array([1722.49816649])], 
    ...         'LFHD': [array([184.55160796]), array([409.68716101]), array([1721.34289625])],
    ...         'LBHD': [array([197.8621642]) , array([251.28892152]), array([1696.90197756])]}
    >>> result = dataAsArray(data)
    >>> expected = [{'RFHD': array([325.82985131, 402.55452959, 1722.49816649]), 
    ...              'LFHD': array([184.55160796, 409.68716101, 1721.34289625]), 
    ...              'LBHD': array([197.8621642, 251.28892152, 1696.90197756])}]
    >>> flag = True #False if any values are not equal
    >>> for i in range(len(result)):
    ...     for key in result[i]:
    ...         if (not array_equal(result[i][key], expected[i][key])):
    ...             flag = False
    >>> flag
    True
    """
    names = list(data.keys())
    dataArray = []

    #make the marker arrays a better format
    for marker in data:
        #Turn multi array into single
        xyz = [ np.array(x) for x in zip( data[marker][0],data[marker][1],data[marker][2] ) ]
        data[marker] = xyz

    #use the first marker to get the length of frames
    datalen = len( data[names[0]] )

    for i in range(datalen):

        frameDict = {}

        for marker in data:
            frameDict[marker] = data[marker][i]

        dataArray.append(frameDict)

    return dataArray

def dataAsDict(data,npArray=False):
    """Converts the frame by frame based data to a dictionary of keys 
    with all motion data as an array per key.

    Parameters
    ----------
    data : array
        List of dict. Indices of `data` correspond to frames
        in the trial. Keys in the dictionary are marker names and 
        values are xyz coordinates of the corresponding marker.
    npArray : bool, optional
        False by default. If set to true, the function will return
        a numpy array for each key instead of a list.
    
    Returns
    -------
    dataDict : dict
        Dictionary of the motion capture data from `data`.

    Examples
    --------
    This example uses a loop and ``numpy.array_equal`` to test the equality
    of individual dictionary elements since python does not guarantee 
    the order of dictionary elements.
 
    >>> from numpy import array, array_equal
    >>> data = [{'RFHD': array([325.82985131, 402.55452959, 1722.49816649]), 
    ...          'LFHD': array([184.55160796, 409.68716101, 1721.34289625]),
    ...          'LBHD': array([197.8621642 , 251.28892152, 1696.90197756])}, 
    ...         {'RFHD': array([326.82985131, 403.55452959, 1723.49816649]), 
    ...          'LFHD': array([185.55160796, 408.68716101, 1722.34289625]),
    ...          'LBHD': array([198.8621642 , 252.28892152, 1697.90197756])}]
    >>> result = dataAsDict(data, True) #return as numpy array
    >>> expected = {'RFHD': array([[ 325.82985131,  402.55452959, 1722.49816649],
    ...                            [ 326.82985131,  403.55452959, 1723.49816649]]), 
    ...             'LFHD': array([[ 184.55160796,  409.68716101, 1721.34289625],
    ...                            [ 185.55160796,  408.68716101, 1722.34289625]]), 
    ...             'LBHD': array([[ 197.8621642 ,  251.28892152, 1696.90197756],
    ...                            [ 198.8621642 ,  252.28892152, 1697.90197756]])}
    >>> flag = True #False if any values are not equal
    >>> for marker in result:
    ...     if (not array_equal(result[marker], expected[marker])):
    ...         flag = False
    >>> flag
    True
    """
    dataDict = {}
    
    for frame in data:
        for key in frame:
            dataDict.setdefault(key,[])
            dataDict[key].append(frame[key])
    
    if npArray == True:
        for key in dataDict:
            dataDict[key] = np.array(dataDict[key])
        
    return dataDict

def writeKinetics(CoM_output,kinetics):
    """Uses numpy.save to write kinetics data as an .npy file.

    Parameters
    ----------
    CoM_output : file, str, or Path
        Full path of the file to be saved to or a file object 
        or a filename.
    kinetics : array_like
        Array data to be saved.

    Examples
    --------
    >>> import tempfile
    >>> pyver = sys.version_info.major
    >>> if pyver == 2:
    ...     tmpdirName = tempfile.mkdtemp()
    ... else:
    ...     tmpdir = tempfile.TemporaryDirectory()
    ...     tmpdirName = tmpdir.name
    >>> from numpy import load
    >>> import os
    >>> from shutil import rmtree
    >>> CoM_output = os.path.join(tmpdirName, 'CoM')
    >>> kinetics = [[246.57466721, 313.55662383, 1026.56323492],
    ...             [246.59137623, 313.6216639, 1026.56440096],
    ...             [246.60850798, 313.6856272, 1026.56531282]]
    >>> writeKinetics(CoM_output, kinetics)
    >>> load(CoM_output + '.npy') #doctest: +NORMALIZE_WHITESPACE
    array([[ 246.57466721, 313.55662383, 1026.56323492],
           [ 246.59137623, 313.6216639 , 1026.56440096],
           [ 246.60850798, 313.6856272 , 1026.56531282]])
    >>> rmtree(tmpdirName)
    """
    np.save(CoM_output,kinetics)
        
def writeResult(data,filename,**kargs):
        """Writes the result of the calculation into a csv file.
 
        Lines 0-6 of the output csv are headers. Lines 7 and onwards
        are angle or axis calculations for each frame. For example,
        line 7 of the csv is output for frame 0 of the motion capture.
        The first element of each row of ouput is the frame number. 
        
        Parameters
        ----------
        data : array_like
            Motion capture data as a matrix of frames as rows.
            Each row is a numpy array of length 273.
            Indices 0-56 correspond to the values for angles.  
            Indices 57-272 correspond to the values for axes.
            See Examples.
        filename : str
            Full path of the csv to be saved. Do not include '.csv'. 
        **kargs : dict
            Dictionary of keyword arguments as follows.

            delimiter : str, optional
                String to be used as the delimiter for the csv file. The
                default delimiter is ','.
            angles : bool or array, optional
                True or false to save angles, or a list of angles to save.
                True by default.
            axis : bool or array, optional
                True or false to save axis, or a list of axis to save.
                True by default.
                
        Examples
        --------
        >>> from numpy import array, zeros
        >>> import os
        >>> from shutil import rmtree
        >>> import tempfile
        >>> pyver = sys.version_info.major
        >>> if pyver == 2:
        ...     tmpdirName = tempfile.mkdtemp()
        ... else:
        ...     tmpdir = tempfile.TemporaryDirectory()
        ...     tmpdirName = tmpdir.name

        Prepare a frame of data to write to csv. This example writes joint angle values
        for the first joint, the pelvis, and axis values for the pelvis origin, PELO.

        >>> frame = zeros(273)
        >>> angles = array([-0.308494914509454, -6.12129279337001, 7.57143110215171])
        >>> for i in range(len(angles)):
        ...     frame[i] = angles[i]
        >>> axis = array([-934.314880371094, -4.44443511962891, 852.837829589844])
        >>> for i in range(len(axis)):
        ...     frame[i+57] = axis[i]
        >>> data = [frame]
        >>> outfile = os.path.join(tmpdirName, 'output')

        Writing angles only.

        >>> writeResult(data, outfile, angles=True, axis=False)
        >>> with open(outfile + '.csv') as file:
        ...     lines = file.readlines()
        >>> result = lines[7].strip().split(',') 
        >>> result #doctest: +NORMALIZE_WHITESPACE
        ['0.000000000000000', 
         '-0.308494914509454', '-6.121292793370010', '7.571431102151710',...]

        Writing axis only.

        >>> writeResult(data, outfile, angles=False, axis=True) 
        (1, 273)...
        >>> with open(outfile + '.csv') as file:
        ...     lines = file.readlines()
        >>> result = lines[7].strip().split(',') 
        >>> result #doctest: +NORMALIZE_WHITESPACE
        ['0.000000000000000', 
         '-934.314880371093977', '-4.444435119628910', '852.837829589843977',...]
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
                if type(dataFilter)==type(None):
                        dataFilter=np.delete(data, filterData, 1)
                else:
                        dataFilter=np.delete(dataFilter, filterData, 1)
        if type(dataFilter)==type(None):
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

def smKeys():
    """A list of segment labels.
 
    Returns
    -------
    keys : array
        List of segment labels.

    Examples
    --------
    >>> smKeys() #doctest: +NORMALIZE_WHITESPACE
    ['Bodymass', 'Height', 'HeadOffset', 'InterAsisDistance', 'LeftAnkleWidth', 
     'LeftAsisTrocanterDistance', 'LeftClavicleLength', 'LeftElbowWidth', 
     'LeftFemurLength', 'LeftFootLength', 'LeftHandLength', 'LeftHandThickness', 
     'LeftHumerusLength', 'LeftKneeWidth', 'LeftLegLength', 'LeftRadiusLength', 
     'LeftShoulderOffset', 'LeftTibiaLength', 'LeftWristWidth', 'RightAnkleWidth', 
     'RightClavicleLength', 'RightElbowWidth', 'RightFemurLength', 'RightFootLength', 
     'RightHandLength', 'RightHandThickness', 'RightHumerusLength', 'RightKneeWidth', 
     'RightLegLength', 'RightRadiusLength', 'RightShoulderOffset', 'RightTibiaLength', 
     'RightWristWidth']
    """
    keys = ['Bodymass', 'Height', 'HeadOffset', 'InterAsisDistance', 'LeftAnkleWidth', 'LeftAsisTrocanterDistance',
            'LeftClavicleLength',
            'LeftElbowWidth', 'LeftFemurLength', 'LeftFootLength', 'LeftHandLength', 'LeftHandThickness',
            'LeftHumerusLength', 'LeftKneeWidth',
            'LeftLegLength', 'LeftRadiusLength', 'LeftShoulderOffset', 'LeftTibiaLength', 'LeftWristWidth',
            'RightAnkleWidth',
            'RightClavicleLength', 'RightElbowWidth', 'RightFemurLength', 'RightFootLength', 'RightHandLength',
            'RightHandThickness', 'RightHumerusLength',
            'RightKneeWidth', 'RightLegLength', 'RightRadiusLength', 'RightShoulderOffset', 'RightTibiaLength',
            'RightWristWidth',
            ]
    return keys
        
def loadVSK(filename,dict=True):
        """Open and load a vsk file.
  
        Parameters
        ----------
        filename : str
            Path to the vsk file to be loaded
        dict : bool, optional
            Returns loaded vsk file values as a dictionary if False. 
            Otherwise, return as an array.
 
        Returns
        -------
        [vsk_keys, vsk_data] : array
            `vsk_keys` is a list of labels. `vsk_data` is a list of values
            corresponding to the labels in `vsk_keys`. 

        Examples
        --------
        RoboSM.vsk in SampleData is used to test the output.

        >>> from .pyCGM_Helpers import getfilenames
        >>> filename = getfilenames(x=3)[2]
        >>> filename
        'SampleData/Sample_2/RoboSM.vsk'
        >>> result = loadVSK(filename)
        >>> vsk_keys = result[0]
        >>> vsk_data = result[1]
        >>> vsk_keys
        ['Bodymass', 'Height', 'InterAsisDistance',...]
        >>> vsk_data
        [72.0, 1730.0, 281.118011474609,...]

        Return as a dictionary.

        >>> result = loadVSK(filename, False)
        >>> type(result)
        <...'dict'>
 
        Testing for some dictionary values.

        >>> result['Bodymass']
        72.0
        >>> result['RightStaticPlantFlex']
        0.17637075483799
        """
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
        #print vsk_keys
        if dict==False: return createVskDataDict(vsk_keys,vsk_data) 
        
        return [vsk_keys,vsk_data]


def splitDataDict(motionData):       
    """Splits an array of motion capture data into separate labels and data.

    Parameters
    ----------
    motionData : array
        List of dict. Indices of `motionData` correspond to frames
        in the trial. Keys in the dictionary are marker names and 
        values are xyz coordinates of the corresponding marker.

    Returns
    -------
    labels, data : tuple
        `labels` is a list of marker position names from the dictionary
        keys in `motiondata`. `data` is a list of xyz coordinate 
        positions corresponding to the marker names in `labels`. 
        Indices of `data` correspond to frames in the trial.

    Examples
    --------
    Example for three markers and two frames of trial.

    >>> from numpy import array
    >>> motionData = [{'RFHD': array([325.82985131, 402.55452959, 1722.49816649]), 
    ...                'LFHD': array([184.55160796, 409.68716101, 1721.34289625]),
    ...                'LBHD': array([197.8621642 , 251.28892152, 1696.90197756])}, 
    ...               {'RFHD': array([326.82985131, 403.55452959, 1723.49816649]), 
    ...                'LFHD': array([185.55160796, 408.68716101, 1722.34289625]),
    ...                'LBHD': array([198.8621642 , 252.28892152, 1697.90197756])}]
    >>> values, labels = splitDataDict(motionData)
    >>> labels
    ['RFHD', 'LFHD', 'LBHD']
    >>> values #doctest: +NORMALIZE_WHITESPACE
    [array([[ 325.82985131,  402.55452959, 1722.49816649],
            [ 184.55160796,  409.68716101, 1721.34289625],
            [ 197.8621642 ,  251.28892152, 1696.90197756]]), 
     array([[ 326.82985131,  403.55452959, 1723.49816649],
            [ 185.55160796,  408.68716101, 1722.34289625],
            [ 198.8621642 ,  252.28892152, 1697.90197756]])]
    """ 
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
    """Converts two lists of values and labels to a dictionary.

    Parameters
    ----------
    values : array
        Array of motion data values. Indices of `values` correspond to
        frames in the trial. Each element is an array of xyz coordinates. 
    labels : array
        List of marker names.
  
    Returns
    -------
    data : array
        Array of dictionaries of motion capture data. Keys are marker 
        names and values are arrays of xyz values. [x, y, z]. Array
        indices correspond to frames of the trial.
   
    Examples
    --------
    Example for three markers and two frames of trial.

    >>> from numpy import array_equal
    >>> labels = ['RFHD', 'LFHD', 'LBHD']
    >>> values = [[[ 325.82985131,  402.55452959, 1722.49816649],
    ...            [ 184.55160796,  409.68716101, 1721.34289625],
    ...            [ 197.8621642 ,  251.28892152, 1696.90197756]],
    ...           [[ 326.82985131,  403.55452959, 1723.49816649],
    ...            [ 185.55160796,  408.68716101, 1722.34289625],
    ...            [ 198.8621642 ,  252.28892152, 1697.90197756]]]
    >>> result = combineDataDict(values, labels)
    >>> expected = [{'RFHD': [325.82985131, 402.55452959, 1722.49816649], 
    ...              'LFHD': [184.55160796, 409.68716101, 1721.34289625], 
    ...              'LBHD': [197.8621642, 251.28892152, 1696.90197756]}, 
    ...              {'RFHD': [326.82985131, 403.55452959, 1723.49816649], 
    ...              'LFHD': [185.55160796, 408.68716101, 1722.34289625], 
    ...              'LBHD': [198.8621642, 252.28892152, 1697.90197756]}]
    >>> flag = True #False if any values are not equal
    >>> for i in range(len(result)):
    ...     for key in result[i]:
    ...         if (not array_equal(result[i][key], expected[i][key])):
    ...             flag = False
    >>> flag
    True
    """
    data = []
    tmp_dict = {}
    for i in range (len(values)):
        for j in range (len(values[i])):
            tmp_dict[labels[j]]=values[i][j]
        data.append(tmp_dict)
        tmp_dict = {}
        
    return data
        

def make_sure_path_exists(path):
    """Creates a file path.
    
    Parameters
    ----------
    path : str
        String of the full file path of the directory to be created.

    Raises
    ------
    Exception
        Raised if the path was unable to be created for any reason 
        other than the directory already existing.

    Examples
    --------
    >>> import os
    >>> from shutil import rmtree
    >>> import tempfile
    >>> pyver = sys.version_info.major
    >>> if pyver == 2:
    ...     tmpdirName = tempfile.mkdtemp()
    ... else:
    ...     tmpdir = tempfile.TemporaryDirectory()
    ...     tmpdirName = tmpdir.name
    >>> newDirectory = os.path.join(tmpdirName, 'newDirectory')
    >>> make_sure_path_exists(newDirectory)
    >>> os.path.isdir(newDirectory)
    True
    >>> rmtree(tmpdirName)
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
