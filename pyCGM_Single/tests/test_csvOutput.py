import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
from pyCGM_Single.pyCGM import pelvisJointCenter
from pyCGM_Single.pyCGM_Helpers import getfilenames
from pyCGM_Single.pycgmCalc import calcAngles, calcKinetics
from pyCGM_Single.pycgmIO import dataAsDict, loadData, loadVSK, writeResult
from pyCGM_Single.pycgmStatic import getStatic, rotmat
from pyCGM_Single.Pipelines import rigid_fill, filtering, prep

class TestCSVOutput:
    @classmethod
    def setup_class(cls):
        """
        Called once for all tests in TestCSVOutput.
        Sets rounding precision, and sets the current working
        directory to the pyCGM folder. Also sets the current python version
        and loads filenames used for testing.
        """
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

        cls.known_labels = ['Pelvis','R Hip','L Hip','R Knee','L Knee','R Ankle',
                        'L Ankle','R Foot','L Foot',
                        'Head','Thorax','Neck','Spine','R Shoulder','L Shoulder',
                        'R Elbow','L Elbow','R Wrist','L Wrist']

    def setup_method(self):
        """
        Called once before every test method runs.
        Creates up a temporary directory to be used for testing
        functions that write to disk.
        """
        if (self.pyver == 2):
            self.tmp_dir_name = tempfile.mkdtemp()
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.tmp_dir_name = self.tmp_dir.name
    
    def teardown_method(self):
        """
        Called once after every test method is finished running.
        If using Python 2, perform cleanup of previously created
        temporary directory in setup_method(). Cleanup is done
        automatically in Python 3.
        """
        if (self.pyver == 2):
            rmtree(self.tmp_dir_name)

    def convert_to_pycgm_label(self, label):
        """Convert angle label name to known pycgm angle label.

        Since the output from vicon nexus and mokka uses slightly 
        different angle labels, we convert them to the pycgm format 
        to make it easier to compare csv outputs across different
        formats.

        Parameters
        ----------
        label : string
            String of the label name.

        Returns
        -------
        string
            String of the known pycgm label corresponding to `label`.
        """
        known_labels = set(self.known_labels)
        
        label_aliases = {
            #Angle names used in vicon and mokka
            'RPelvisAngles': 'Pelvis',
            'RHipAngles' : 'R Hip',
            'LHipAngles' : 'L Hip',
            'RKneeAngles' : 'R Knee',
            'LKneeAngles' : 'L Knee',
            'RAnkleAngles' : 'R Ankle',
            'LAnkleAngles' : 'L Ankle',
            'RFootProgressAngles' : 'R Foot',
            'LFootProgressAngles' : 'L Foot',
            'RHeadAngles' : 'Head',
            'RThoraxAngles' : 'Thorax',
            'RNeckAngles' : 'Neck',
            'RSpineAngles' : 'Spine',
            'RShoulderAngles' : 'R Shoulder',
            'LShoulderAngles' : 'L Shoulder',
            'RElbowAngles' : 'R Elbow',
            'LElbowAngles' : 'L Elbow',
            'RWristAngles' : 'R Wrist',
            'LWristAngles' : 'L Wrist'
        }

        if label in known_labels:
            return label
        elif label in label_aliases:
            return label_aliases[label]
        else:
            return None

    def load_output_csv(self, csv_file, header_line_number=5, first_output_row=7, first_output_col=1, label_prefix_len=0):
        """
        Loads an output csv of angles or markers into a 2d array where each index
        represents a row in the csv.

        This function tests for equality of the 19 angles that pycgm outputs, but allows
        loading of files of different formats. Assumes that each angle has exactly three
        values associated with it (x, y, z).

        Parameters
        ----------
        csv_file : string
            String of the path of the filename to be loaded.
        header_line_number : int
            Index of the line number in which the angle or marker labels are written.
        first_output_row : int
            Index of the line number in which the first row of output begins.
        first_output_col : int
            Index of the column number in which the first column of output begins.
        label_prefix_len : int
            Length of the prefix on each label, if it exists. 0 by default.

        Returns
        -------
        output : 2darray
            2d matrix where each index represents a row of angle data loaded from
            the csv.
        """
        output = []
        infile = open(csv_file, 'r')
        lines = infile.readlines()
        #Create a dict of index to known pycgm label:
        index_to_header = {}
        headers = lines[header_line_number].strip().split(',')[first_output_col:]
        for i in range(len(headers)):
            header = headers[i]
            if header != "":
                #Find which known pycgm header this header corresponds to, trimming prefix length if needed
                header = header.strip()[label_prefix_len:]
                header = self.convert_to_pycgm_label(header)
                #Record that index i corresponds to this header
                index_to_header[i] = header
        
        #Loop over all lines starting from the first line of output
        for line in lines[first_output_row:]:
            arr = [0 for i in range(19*3)]
            #Convert line in the csv to an array of floats
            formatted_line = line.strip().split(',')[first_output_col:]
            l = []
            for num in formatted_line:
                try:
                    l.append(float(num))
                except:
                    l.append(0)
            #Loop over the array of floats, knowing which indices 
            #corresponds to which angles from the index_to_header dictionary
            for i in range(len(l)):
                if i in index_to_header:
                    label = index_to_header[i]
                    if (label != None):
                        index = self.known_labels.index(label) * 3
                        arr[index] = l[i]
                        arr[index+1] = l[i+1]
                        arr[index+2] = l[i+2]
            output.append(arr)

        infile.close()
        return np.array(output)

    def load_center_of_mass(self, csv_file, row_start, col_start):
        """Load center of mass values into an array, where each index
        has the center of mass coordinates for a frame.

        Parameters
        ----------
        csv_file : string
            Filename of the csv file to be loaded.
        row_start : int
            Index of the first row in which center of mass data begins.
        col_start : int
            Index of the first column in which center of mass data begins.

        Returns
        -------
        center_of_mass : 2darray
            Array representation of the center of mass data.
        """
        infile = open(csv_file, 'r')
        center_of_mass = []
        lines = infile.readlines()
        for line in lines[row_start:]:
            formatted_line = line.strip().split(',')
            coordinates = formatted_line[col_start:col_start+3]
            coordinates = [float(x) for x in coordinates]
            center_of_mass.append(coordinates)
        infile.close()
        return center_of_mass

    def compare_output(self, result, expected, rounding_precision, has_GCS=True):
        """Compares two output CSV angle files and determines if they are equal.

        Files are in the format written by pycgmIO.writeResult().

        Parameters
        ----------
        result : list
            2d array of output angles from a result calculation.
        expected : string
            2d array of output angles of expected values.
        rounding_precision : int
            Number of decimal places to compare output values to.
        has_GCS : boolean
            True if the outputs were calculated with an accurate Global Coordinate System axis.
            If False, Pelvis, R Foot, L Foot, Head, and Thorax angles will be ignored since they
            are dependent on the GCS value. True by default.
        """
        if (not has_GCS):
            #List of angles to ignore because they are affected by the global coordinate system.
            #Ignored angles are Pelvis, R Foot, L Foot, Head, Thorax
            ignore = [i for i in range(21, 33)]
            ignore.extend([0,1,2])

            #Delete those angles
            expected = np.delete(expected, ignore, axis = 1)
            result = np.delete(result, ignore, axis = 1)
        
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    def compare_center_of_mass(self, result, expected, tolerance):
        """Asserts that two arrays of center of mass coordinates
        are equal with a certain tolerance.

        Assumes that center of mass coordinates are in mm.
        
        Result and expected must be the same length.

        Parameters
        ----------
        result : array
            Array of result center of mass coordinates.
        expected : array
            Array of expected center of mass coordinates.
        tolerance : int
            Sets how big the difference between center of mass coordinates
            can be.
        """
        for i in range(len(expected)):
            for j in range(len(expected[i])):
                assert abs(result[i][j] - expected[i][j] < tolerance)
        
    def load_files(self, dynamic_trial, static_trial, vsk_file):
        """
        Uses load functions from pycgmIO to load data from c3d and
        vsk files.
        """
        motion_data = loadData(dynamic_trial)
        static_data = loadData(static_trial)
        vsk_data = loadVSK(vsk_file, dict=False)
        return motion_data, static_data, vsk_data
    
    def test_ROM(self):
        """
        Tests pycgm output csv files using files in SampleData/ROM/.
        """
        motion_data,static_data,vsk_data = self.load_files(self.filename_ROM_dynamic, self.filename_ROM_static, self.filename_ROM_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expected_file = os.path.join(self.directory_ROM,'pycgm_results.csv.csv')

        result = self.load_output_csv(outfile + '.csv')
        expected = self.load_output_csv(expected_file)
        self.compare_output(result, expected, self.rounding_precision)

    def test_59993_Frame(self):
        """
        Tests pycgm output csv files using files in SampleData/59993_Frame/.
        """
        motion_data,static_data,vsk_data = self.load_files(self.filename_59993_Frame_dynamic, self.filename_59993_Frame_static, self.filename_59993_Frame_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,start=0, end=500,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expected_file = os.path.join(self.directory_59993_Frame,'pycgm_results.csv.csv')

        result = self.load_output_csv(outfile + '.csv')
        expected = self.load_output_csv(expected_file)
        self.compare_output(result, expected, self.rounding_precision, has_GCS=False)

    '''
    def test_vicon(self):
        """
        Tests pycgm output csv files using files in SampleData/Vicon_Test_Files/.
        """
        motion_data,static_data,vsk_data = self.load_files(self.filename_vicon_dynamic, self.filename_vicon_static, self.filename_vicon_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        
        outfile = os.path.join(self.tmp_dir_name, 'output')
        writeResult(kinematics,outfile,angles=True,axis=False)
        expected_file = os.path.join(self.directory_vicon,'Movement_trial.csv')

        result = self.load_output_csv(outfile + '.csv')
        expected = self.load_output_csv(expected_file, header_line_number=2, first_output_row=5, first_output_col=2, label_prefix_len=5)
        self.compare_output(result, expected, 3, has_GCS=False)

    def test_vicon_center_of_mass(self):
        """
        Test center of mass output values using sample files in SampleData/Vicon_Test_Files/.
        """
        motion_data,static_data,vsk_data = self.load_files(self.filename_vicon_dynamic, self.filename_vicon_static, self.filename_vicon_vsk)
        cal_SM = getStatic(static_data,vsk_data,flat_foot=False)
        kinematics,joint_centers=calcAngles(motion_data,vsk=cal_SM,splitAnglesAxis=False,formatData=False,returnjoints=True)
        kinetics = calcKinetics(joint_centers, cal_SM['Bodymass'])
        expected = self.load_center_of_mass(os.path.join(self.directory_vicon,'Movement_trial.csv'), 5, 2)
        self.compare_center_of_mass(kinetics, expected, 30)
    
    '''
