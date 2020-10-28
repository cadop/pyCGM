import unittest
import numpy as np
from tempfile import TemporaryDirectory
from pyCGM_Single.pyCGM import pelvisJointCenter
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmCalc import calcAngles
from pyCGM_Single.pycgmIO import dataAsDict, loadData, loadVSK, writeResult
from pyCGM_Single.pycgmStatic import getStatic, rotmat
from pyCGM_Single.Pipelines import rigid_fill, filtering, prep
import os

class TestCSVOutput(unittest.TestCase):
    rounding_precision = 8
    tmpdir = TemporaryDirectory()
    cwd = os.getcwd()
    if (cwd.split(os.sep)[-1]=="pyCGM_Single"):
        parent = os.path.dirname(cwd)
        os.chdir(parent)
    cwd = os.getcwd()
    sampleDataDir = os.path.join(cwd, "SampleData")

    def loadFiles(self, x):
        cur_dir = self.cwd
        dynamic_trial,static_trial,vsk_file,_,_ = getfilenames(x)
        dynamic_trial = os.path.join(cur_dir, dynamic_trial)
        static_trial = os.path.join(cur_dir, static_trial)
        vsk_file = os.path.join(cur_dir, vsk_file)
        motionData = loadData(dynamic_trial)
        staticData = loadData(static_trial)
        vsk = loadVSK(vsk_file, dict=False)
        return motionData, staticData, vsk

    def performPipelineOperations(self, staticData, motionData):
        for frame in motionData:
            frame['SACR'] = pelvisJointCenter(frame)[2]
        staticDataDict = dataAsDict(staticData,npArray=True)
        motionDataDict = dataAsDict(motionData,npArray=True)
        movementFilled = rigid_fill(motionDataDict,staticDataDict) 
        movementFiltered = filtering(motionDataDict)
        movementFinal = prep(movementFiltered)
        return movementFinal
    
    def test_ROM(self):
        motionData,staticData,vskData = self.loadFiles(2)
        calSM = getStatic(staticData,vskData,flat_foot=False)
        kinematics,joint_centers=calcAngles(motionData,start=None,end=None,vsk=calSM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmpdir.name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expectedFile = os.path.join(os.path.join(self.sampleDataDir, 'ROM'), 'pycgm_results.csv.csv')
        expectedLines = []
        resultLines = []
        with open(expectedFile) as expectedHandle:
            expectedLines = expectedHandle.readlines()
        with open(outfile + '.csv') as resultHandle:
            resultLines = resultHandle.readlines()

        for i in range(7, len(expectedLines)):
            expectedLine = np.array(expectedLines[i].strip().split(','))
            resultLine = np.array(resultLines[i].strip().split(','))
            for j in range(len(expectedLine)):
            	np.testing.assert_almost_equal(float(expectedLine[j]), float(resultLine[j]), self.rounding_precision)







            
        
        
