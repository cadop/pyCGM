#pyCGM

###########
#This file is an example of how to call the pycgm code without a console, or more likely, as a 
# way to integrate the code into your own system/software
#There are a few commented parts that show how to use some pipeline functions such as filtering. 
##########

import sys
import os

try: from pyCGM_Single.Pipelines import rigid_fill, filtering, prep,clearMarker
except: print("Could not import Pipelines.py, possibly missing scipy\n Otherwise check the directory locations")

from pyCGM_Single import pycgmStatic
from pyCGM_Single import pycgmIO
from pyCGM_Single import pycgmCalc   
from pyCGM_Single import pyCGM_Helpers
   
def loadData(dynamic_trial,static_trial,vsk_file):
    #load the data, usually there is some checks in here to make sure we loaded
    # correctly, but for now we assume its loaded
    motionData  = pycgmIO.loadData(dynamic_trial) 
    vsk = pycgmIO.loadVSK(vsk_file,dict=False)
    staticData = pycgmIO.loadData(static_trial)
    #The vsk is loaded, if dict=True (default), we combine
    #vsk = pycgmIO.createVskDataDict(vsk[0],vsk[1]) 
    
    return motionData,vsk,staticData

def main():
    #Load the filenames
    #pyCGM_Helpers.py contains some sample directory data based on github directories
    dynamic_trial,static_trial,vsk_file,outputfile,CoM_output = pyCGM_Helpers.getfilenames(x=2) #change x to use different files
    
    #Load a dynamic trial, static trial, and vsk (subject measurements)
    motionData,vskData,staticData = loadData(dynamic_trial,static_trial,vsk_file)
    
    #Calibrate the static offsets and subject measurements
    calSM = pycgmStatic.getStatic(staticData,vskData,flat_foot=False)

    # #Load data as a dictionary instead of a frame-by-frame array of dictionary
    # staticDataDict = pycgmIO.dataAsDict(staticData,npArray=True)
    # motionDataDict = pycgmIO.dataAsDict(motionData,npArray=True)    

    # ####  Start Pipeline oeprations 
    # movementFilled = rigid_fill(motionDataDict,staticDataDict) 
    # movementFiltered = filtering(motionDataDict)
    # movementFinal = prep(movementFiltered)
    # motionData = movementFinal
    # ### End pipeline operations

    #hack for changing the global coordinates until finding a proper way    
    # this impacts the global angles, such as pelvis, but not the anatomical angles (e.g., hip)
    #calSM['GCS'] = pycgmStatic.rotmat(x=0,y=0,z=180) 
    #calSM['HeadOffset'] = 0  #example of manually modifying a subject measurement
    
    kinematics,joint_centers=pycgmCalc.calcAngles(motionData,start=None,end=None,vsk=calSM,splitAnglesAxis=False,formatData=False,returnjoints=True)
    kinetics=pycgmCalc.calcKinetics(joint_centers, calSM['Bodymass'])
    
    #Write the results to a csv file, if wanted, 
    # otherwise could just return the angles/axis to some other function
    pycgmIO.writeResult(kinematics,outputfile,angles=True,axis=False)    
    pycgmIO.writeKinetics(CoM_output,kinetics) #quick save of CoM

    return

main()
