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

###########
#This file is an example of how to call the pycgm code 
# useful to integrate the code into your own system/software
#Now it imports the singlecorecode version as this is the only one 
# working python 2 and 3
##########

import sys
import os
from SingleCoreCode import pycgmStatic
from SingleCoreCode import pycgmIO
from SingleCoreCode import pycgmCalc

def getfilenames():
   
    dir = 'SampleData\\ROM'
    #Change this to the subject 
    dynamic_trial = dir+'\\'+'Sample_Dynamic.c3d' 
    static_trial = dir+'\\'+'Sample_Static.c3d' 
    vsk_file = dir+'\\'+'Sample_SM.vsk'     
    outputfile = dir+'\\'+'pycgm_updates_results.csv'    
    
    return dynamic_trial,static_trial,vsk_file,outputfile
    
def loadData(dynamic_trial,static_trial,vsk_file):
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
    #Load the filenames
    dynamic_trial,static_trial,vsk_file,outputfile = getfilenames()
    #Load a dynamic trial, static trial, and vsk (subject measurements)
    motionData,vskData,staticData = loadData(dynamic_trial,static_trial,vsk_file)
    
    #Calculate the static offsets
    # flat_foot = False
    calibratedMeasurements = pycgmStatic.getStatic(staticData,vskData,flat_foot=False)
	#Calculate the dynamic trial
    # passing the calibrated subject measurements 
    
    result=pycgmCalc.calcAngles(motionData,start=None,end=None,vsk=calibratedMeasurements,splitAnglesAxis=False,formatData=False)

    #Write the results to a csv file, if wanted, 
    # otherwise could just return the angles/axis to some other function
    pycgmIO.writeResult(result,outputfile)

    return

main()
