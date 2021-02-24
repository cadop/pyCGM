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

"""
This file is an example of how to call the pycgm code without a console, or 
more likely, as a way to integrate the code into your own system/software
"""

import sys
import os
# Temporary try/except to deal with relative import issue
# This file should eventually be rewritten anyway
try:
    import pycgmStatic
    import pycgmIO
    import pycgmCalc
except:
    import pyCGM_Single.pycgmStatic as pycgmStatic
    import pyCGM_Single.pycgmIO as pycgmIO
    import pyCGM_Single.pycgmCalc as pycgmCalc

def getfilenames():
    """Gets the filenames of the dynamic trial, static trial, vsk file, and
    output file and returns their file paths. The example filenames below
    are found in the SampleData folder in the git repository.

    Returns
    -------
    tuple
        Returns a tuple of four strings that includes the paths to
        dynamic_trial, static_trial, vsk_file, and outputFile
    """
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir( scriptdir )
    os.chdir( ".." ) #relative to github
    os.chdir( "./SampleData/59993_Frame/" ) #Directory from github
    dir = os.getcwd() + os.sep
    dynamic_trial = dir+'59993_Frame_Dynamic.c3d'
    static_trial = dir+'59993_Frame_Static.c3d'
    vsk_file = dir+'59993_Frame_SM.vsk'
    outputfile = dir+'pycgm_results.csv'
    os.chdir( scriptdir )

    return dynamic_trial,static_trial,vsk_file,outputfile

def loadData(dynamic_trial,static_trial,vsk_file):
    """Loads the data given the paths to the dynamic trial, the static
    trial, and the vsk file. Calls to pycgmIO are made to load in the
    data.

    Returns
    -------
    tuple
        Returns a tuple of three arrays that pertain to the motion data,
        the vsk file, and the static data. These arrays are then directly
        used to compute the joint angles. To see how these arrays are
        made, please refer to pycgmIO.
    """
    #load the data, usually there is some checks in here to make sure we loaded
    # correctly, but for now we assume its loaded
    motionData  = pycgmIO.loadData(dynamic_trial)
    vskdata = pycgmIO.loadVSK(vsk_file)
    staticData = pycgmIO.loadData(static_trial)
    #The vsk is loaded, but for some reasons the return is split, so we combine
    vsk = pycgmIO.createVskDataDict(vskdata[0],vskdata[1])
    print("Motion Data Length:",len(motionData))

    return motionData,vsk,staticData

def main():
    """Take in the paths to the trials, the vsk file, and the output file,
    and load the data in. Calibrate the measurements using pycgmStatic's
    getStatic() function, and calculate the angles using pycgmCalc's
    calcAngles() function.

    The result of the calcAngles() function are written through pycgmIO's
    writeResult() function.
    """
    #Load the filenames
    dynamic_trial,static_trial,vsk_file,outputfile = getfilenames()
    #Load a dynamic trial, static trial, and vsk (subject measurements)
    motionData,vskData,staticData = loadData(dynamic_trial,static_trial,vsk_file)

    #Calculate the static offsets
    flat_foot = False
    calibratedMeasurements = pycgmStatic.getStatic(staticData,vskData,flat_foot)
	#Calculate the dynamic trial
    # passing the calibrated subject measurements
    motionData = motionData[:500] #temporary, just to speed up the calculation
    result=pycgmCalc.calcAngles(motionData,start=None,end=None,vsk=calibratedMeasurements,splitAnglesAxis=False,formatData=False)

    #Write the results to a csv file, if wanted,
    # otherwise could just return the angles/axis to some other function
    pycgmIO.writeResult(result,outputfile)

    return

main()
