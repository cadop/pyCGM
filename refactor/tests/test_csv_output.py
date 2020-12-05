import pytest
import numpy as np
import os
import sys
import tempfile
from shutil import rmtree
from refactor.pycgm import CGM

#Define several helper functions used in loading and comparing output CSV files
def convert_to_pycgm_label(label):
    """Convert angle label name to known pycgm angle label.

    Since output from other programs can use slightly 
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
    known_labels = set(['Pelvis','R Hip','L Hip','R Knee','L Knee','R Ankle',
                        'L Ankle','R Foot','L Foot',
                        'Head','Thorax','Neck','Spine','R Shoulder','L Shoulder',
                        'R Elbow','L Elbow','R Wrist','L Wrist'])
    
    label_aliases = {
        #Angle names commonly used to pycgm angle names
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

def load_output_csv(csv_file, header_line_number=5, first_output_row=7, first_output_col=1, label_prefix_len=0):
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
        The default header_line_number of 5 represents the output from pycgmIO.writeResult().
    first_output_row : int
        Index of the line number in which the first row of output begins.
        The default first_output_row of 7 represents the output from pycgmIO.writeResult().
    first_output_col : int
        Index of the column number in which the first column of output begins.
        The default first_output_col of 1 represents the output from pycgmIO.writeResult().
    label_prefix_len : int
        Length of the prefix on each label, if it exists. 0 by default.

    Returns
    -------
    output : 2darray
        2d matrix where each index represents a row of angle data loaded from
        the csv.
    """
    known_labels = ['Pelvis','R Hip','L Hip','R Knee','L Knee','R Ankle',
                        'L Ankle','R Foot','L Foot',
                        'Head','Thorax','Neck','Spine','R Shoulder','L Shoulder',
                        'R Elbow','L Elbow','R Wrist','L Wrist']
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
            header = convert_to_pycgm_label(header)
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
                    index = known_labels.index(label) * 3
                    arr[index] = l[i]
                    arr[index+1] = l[i+1]
                    arr[index+2] = l[i+2]
        output.append(arr)

    infile.close()
    return np.array(output)

def load_center_of_mass(csv_file, row_start, col_start):
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

def compare_center_of_mass(result, expected, tolerance):
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
        Sets how large the difference between any two center of mass coordinates
        can be.
    """
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert abs(result[i][j] - expected[i][j] < tolerance)

def get_columns_to_compare(test_folder):
    """
    Helper function to test the files in SampleData. Gets
    indices of angles that can be compared for equality, depending
    on which file is being compared.

    There are 57 angle coordinates to be compared, with 3 coordinates
    for each of 19 angles.

    If the global coordinate system is unknown for a given file,
    angles affected by the GCS are ignored.
    Ignored angles are Pelvis, R Foot, L Foot, Head, Thorax, with corresponding
    indices 0, 1, 2 and 21 - 32.

    The files in Test_Files also ignore the Neck X coordinate, at 
    index 33.
    """
    gcs_ignore = [i for i in range(21, 33)]
    gcs_ignore.extend([0,1,2])
    columns = [i for i in range(57)]
    if (test_folder == 'ROM'):
        return columns
    if (test_folder == '59993_Frame'):
        for i in gcs_ignore:
            columns.remove(i)
        return columns
    if (test_folder == 'Test_Files'):
        for i in gcs_ignore:
            columns.remove(i)
        columns.remove(33)
        return columns

class TestCSVOutput:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in TestCSVOutput.
        Sets rounding precision, and sets the current working
        directory to the pyCGM folder. Sets the current python version
        and loads filenames used for testing.

        We also use the pycgm functions to generate and load output CSV data
        and load them into the class.
        """
        self.rounding_precision = 8
        cwd = os.getcwd()
        if (cwd.split(os.sep)[-1]=="refactor"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        self.cwd = os.getcwd()
        self.pyver = sys.version_info.major

        #Create a temporary directory used for writing CSVs to
        if (self.pyver == 2):
            self.tmp_dir_name = tempfile.mkdtemp()
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.tmp_dir_name = self.tmp_dir.name
        
        #Create file path names for the files being tested
        self.sample_data_directory = os.path.join(self.cwd, "SampleData")
        self.directory_59993_Frame = os.path.join(self.sample_data_directory, '59993_Frame')
        self.directory_ROM = os.path.join(self.sample_data_directory, 'ROM')
        self.directory_test = os.path.join(self.sample_data_directory, 'Test_Files')

        #Load outputs to be tested for SampleData/59993_Frame/

        self.filename_59993_Frame_dynamic = os.path.join(self.directory_59993_Frame, '59993_Frame_Dynamic.c3d')
        self.filename_59993_Frame_static = os.path.join(self.directory_59993_Frame, '59993_Frame_Static.c3d')
        self.filename_59993_Frame_vsk = os.path.join(self.directory_59993_Frame, '59993_Frame_SM.vsk')
        self.subject_59993_Frame = CGM(self.filename_59993_Frame_static, self.filename_59993_Frame_dynamic, self.filename_59993_Frame_vsk, start=0, end=500)
        self.subject_59993_Frame.run()
        outfile = os.path.join(self.tmp_dir_name, 'output_59993_Frame.csv')
        self.subject_59993_Frame.write_results(outfile, write_axes=False, write_com=False)
        expected_file = os.path.join(self.directory_59993_Frame,'pycgm_results.csv.csv')
        self.result_59993_Frame = load_output_csv(outfile)
        self.expected_59993_Frame = load_output_csv(expected_file)
        
        #Load outputs to be tested for SampleData/ROM/

        self.filename_ROM_dynamic = os.path.join(self.directory_ROM, 'Sample_Dynamic.c3d')
        self.filename_ROM_static = os.path.join(self.directory_ROM, 'Sample_Static.c3d')
        self.filename_ROM_vsk = os.path.join(self.directory_ROM, 'Sample_SM.vsk')
        self.subject_ROM = CGM(self.filename_ROM_static, self.filename_ROM_dynamic, self.filename_ROM_vsk)
        self.subject_ROM.run()        
        outfile = os.path.join(self.tmp_dir_name, 'output_ROM')
        self.subject_ROM.write_results(outfile, write_axes=False, write_com=False)
        expected_file = os.path.join(self.directory_ROM,'pycgm_results.csv.csv')
        self.result_ROM = load_output_csv(outfile)
        self.expected_ROM = load_output_csv(expected_file)
        
        #Load outputs to be tested for SampleData/Test_Files/

        self.filename_test_dynamic = os.path.join(self.directory_test, 'Movement_trial.c3d')
        self.filename_test_static = os.path.join(self.directory_test, 'Static_trial.c3d')
        self.filename_test_vsk = os.path.join(self.directory_test, 'Test.vsk')
        self.subject_test = CGM(self.filename_test_static, self.filename_test_dynamic, self.filename_test_vsk)
        self.subject_test.run()
        outfile = os.path.join(self.tmp_dir_name, 'output_Test_Files')
        self.subject_test.write_results(outfile, write_axes=False, write_com=False)
        expected_file = os.path.join(self.directory_test,'Movement_trial.csv')
        self.result_Test_Files = load_output_csv(outfile)
        self.expected_Test_Files = load_output_csv(expected_file, header_line_number=2, first_output_row=5, first_output_col=2, label_prefix_len=5)
        

    @classmethod
    def teardown_class(self):
        """
        Called once after all tests in TestCSVOutput are finished running.
        If using Python 2, perform cleanup of the previously created
        temporary directory in setup_class(). Cleanup is done automatically in 
        Python 3. 
        """
        if (self.pyver == 2):
            rmtree(self.tmp_dir_name)

    @pytest.fixture
    def angles_ROM(self, request):
        column = request.param
        return self.result_ROM[:,column], self.expected_ROM[:,column]

    @pytest.mark.parametrize("angles_ROM", get_columns_to_compare("ROM"), indirect=True)
    def test_ROM(self, angles_ROM):
        """
        Tests pycgm output csv files using input files from SampleData/ROM/.
        """
        result_angles, expected_angles = angles_ROM
        np.testing.assert_almost_equal(result_angles, expected_angles, self.rounding_precision)
    
    
    @pytest.fixture
    def angles_59993_Frame(self, request):
        column = request.param
        return self.result_59993_Frame[:,column], self.expected_59993_Frame[:,column]

    @pytest.mark.parametrize("angles_59993_Frame", get_columns_to_compare("59993_Frame"), indirect=True)
    def test_59993_Frame(self, angles_59993_Frame):
        """
        Tests pycgm output csv files using input files from SampleData/ROM/.
        """
        result_angles, expected_angles = angles_59993_Frame
        np.testing.assert_almost_equal(result_angles, expected_angles, self.rounding_precision)

    @pytest.fixture
    def angles_Test_Files(self, request):
        column = request.param
        return self.result_Test_Files[:,column], self.expected_Test_Files[:,column]

    @pytest.mark.parametrize("angles_Test_Files", get_columns_to_compare("Test_Files"), indirect=True)
    def test_Test_Files(self, angles_Test_Files):
        """
        Tests pycgm output csv files using input files from SampleData/ROM/.
        """
        result_angles, expected_angles = angles_Test_Files
        np.testing.assert_almost_equal(result_angles, expected_angles, 3)

    def test_Test_Files_center_of_mass(self):
        """
        Test center of mass output values using sample files in SampleData/Test_Files/.
        """
        kinetics = self.subject_test.com_results
        expected = load_center_of_mass(os.path.join(self.directory_test,'Movement_trial.csv'), 5, 2)
        compare_center_of_mass(kinetics, expected, 30)
    