import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
from pyCGM_Single.pyCGM import pelvisJointCenter
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmCalc import calcAngles
from pyCGM_Single.pycgmIO import dataAsDict, loadData, loadVSK, writeResult
from pyCGM_Single.pycgmStatic import getStatic, rotmat
from pyCGM_Single.Pipelines import rigid_fill, filtering, prep

    
class TestCSVOutput:
    @classmethod
    def setup_class(cls):
        cls.rounding_precision = 8
        cwd = os.getcwd()
        if (cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()
        cls.pyver = sys.version_info.major
        cls.sample_data_directory = os.path.join(cls.cwd, "SampleData")
        cls.directory_59993_Frame = os.path.join(cls.sample_data_directory, '59993_Frame')
        cls.directory_ROM = os.path.join(cls.sample_data_directory, 'ROM')
        cls.directory_vicon = os.path.join(cls.sample_data_directory, 'Vicon_Test_Files')

        cls.filename_59993_Frame_dynamic = os.path.join(cls.directory_59993_Frame, '59993_Frame_Dynamic.c3d')
        cls.filename_59993_Frame_static = os.path.join(cls.directory_59993_Frame, '59993_Frame_Static.c3d')
        cls.filename_59993_Frame_vsk = os.path.join(cls.directory_59993_Frame, '59993_Frame_SM.vsk')

        cls.filename_ROM_dynamic = os.path.join(cls.directory_ROM, 'Sample_Dynamic.c3d')
        cls.filename_ROM_static = os.path.join(cls.directory_ROM, 'Sample_Static.c3d')
        cls.filename_ROM_vsk = os.path.join(cls.directory_ROM, 'Sample_SM.vsk')

        cls.filename_vicon_dynamic = os.path.join(cls.directory_vicon, 'Movement_trial.c3d')
        cls.filename_vicon_static = os.path.join(cls.directory_vicon, 'Static_trial.c3d')
        cls.filename_vicon_vsk = os.path.join(cls.directory_vicon, 'Test.vsk')

    def setup_method(self):
        if (self.pyver == 2):
            self.tmp_dir_name = tempfile.mkdtemp()
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.tmp_dir_name = self.tmp_dir.name
    
    def teardown_method(self):
        if (self.pyver == 2):
            rmtree(self.tmp_dir_name)

    def compare_csv(self, result, expected, rounding_precision, has_GCS=True):
        """Compares two output CSV angle files and determines if they are equal.

        Files are in the format written by pycgmIO.writeResult().

        Parameters
        ----------
        result : string
            Filename of the result file.
        expected : string
            Filename of the expected file.
        rounding_precision : int
            Number of decimal places to compare output values to.
        has_GCS : boolean
            True if the outputs were calculated with an accurate Global Coordinate System axis.
            If False, Pelvis, R Foot, L Foot, Head, and Thorax angles will be ignored since they
            are dependent on the GCS value.
        """
        expected_file = open(expected, 'r')
        result_file  = open(result, 'r')
        expected_lines = expected_file.readlines()
        result_lines = result_file.readlines()

        expected_angles = []
        result_angles = []
        for i in range(7, len(expected_lines)):
            expected_line = np.array(expected_lines[i].strip().split(','))
            expected_angles.append(expected_line.astype(np.float64))
            result_line = np.array(result_lines[i].strip().split(','))
            result_angles.append(result_line.astype(np.float64))
        
        if (not has_GCS):
            #List of angles to ignore because they are affected by the global coordinate system.
            #Ignored angles are Pelvis, R Foot, L Foot, Head, Thorax
            ignore = [i for i in range(22, 34)]
            ignore.extend([1,2,3])

            #Delete
            expected_angles = np.delete(expected_angles, ignore, axis = 1)
            result_angles = np.delete(result_angles, ignore, axis = 1)
        
        np.testing.assert_almost_equal(expected_angles, result_angles, rounding_precision)
    
    def load_files(self, dynamic_trial, static_trial, vsk_file):
        motion_data = loadData(dynamic_trial)
        static_data = loadData(static_trial)
        vsk_data = loadVSK(vsk_file, dict=False)
        return motion_data, static_data, vsk_data
    
    def test_ROM(self):
        motion_data,static_data,vsk_data = self.load_files(self.filename_ROM_dynamic, self.filename_ROM_static, self.filename_ROM_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expected_file = os.path.join(self.directorpycgmile, self.rounding_precision)
    
    def test_59993_Frame(self):
        motion_data,static_data,vsk_data = self.load_files(self.filename_59993_Frame_dynamic, self.filename_59993_Frame_static, self.filename_59993_Frame_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,start=0, end=500,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=True)
        expected_file = os.path.join(self.directory_59993_Frame,'pycgm_results.csv.csv')
        self.compare_csv(outfile + '.csv', expected_file, self.rounding_precision, has_GCS=False)
    
    '''
    def test_vicon(self):
        motion_data,static_data,vsk_data = self.load_files(self.filename_vicon_dynamic, self.filename_vicon_static, self.filename_vicon_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expected_file = os.path.join(self.directory_vicon,'vicon_results.csv')
        self.compare_csv(outfile + '.csv', expected_file, 3, has_GCS=False)
    '''

        
        
