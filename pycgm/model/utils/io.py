import re
import time
import xml.etree.ElementTree as ET

import numpy as np

from . import c3d
from .constants import POINT_DTYPE
from .dataset import Dataset

class IO():
    def __init__(self, static_filename, dynamic_filenames, measurement_filename):
        self.static_filename = static_filename
        self.dynamic_filenames = [dynamic_filenames] if not isinstance(dynamic_filenames, list) else dynamic_filenames
        self.dynamics = []
        { 'RoboWalk' : ['SACR', 'RMKN'] }
        self.measurement_filename = measurement_filename

    def load(self):
        static, num_frames = self.load_c3d(self.static_filename, return_frame_count=True)
        self.static = Dataset(static, num_frames)

        for filename in self.dynamic_filenames:
            dynamic, num_frames = self.load_c3d(filename, return_frame_count=True)
            trial_name = re.findall(r'[^\/]+(?=\.)', filename)[0]
            self.dynamics.append(Dataset(dynamic, num_frames, trial_name))

        self.measurements = self.load_vsk(self.measurement_filename)

        return self.static, self.dynamics, self.measurements

    def load_c3d(self, filename, return_frame_count=False):
        """Loads motion capture data from a c3d file into a numpy structured array.

        Parameters
        ----------
        filename : str
            Path of the c3d file to be loaded.

        return_frame_count : bool, optional
            Set to True to return the number of frames as well

        Returns
        -------
        dynamic_struct : array
            A structured array of the file's marker data
        """

        reader = c3d.Reader(open(filename, 'rb'))
        labels = reader.get('POINT:LABELS').string_array
        frames_list = np.array(list(reader.read_frames(True, True, yield_frame_no=False)), dtype=object)

        marker_names = [str(label.rstrip()) for label in labels]
        num_markers = len(frames_list[0][0])
        num_frames = len(frames_list)
        frame_numbers = np.arange(num_frames)
        float_arr = np.column_stack(frames_list[:, 0]).astype(float).reshape(num_markers, num_frames, 3)

        marker_xyz = [(key, (POINT_DTYPE, (num_frames,))) for key in marker_names]
        marker_positions = np.insert(float_arr, 0, frame_numbers, axis=2)
        marker_positions.dtype = POINT_DTYPE

        dynamic_struct = np.empty((1), dtype=marker_xyz)
        for i, name in enumerate(dynamic_struct.dtype.names):
            dynamic_struct[name][0][:, np.newaxis] = marker_positions[i]

        if return_frame_count:
            return dynamic_struct, num_frames

        return dynamic_struct


    def load_vsk(self, filename, dict=True):
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

        >>> filename = 'SampleData/Sample_2/RoboSM.vsk'
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
        # Check if the filename is valid
        # if not, return None
        if filename == '':
            return None

        # Create an XML tree from file
        tree = ET.parse(filename)

        # Get the root of the file
        # <KinematicModel>
        root = tree.getroot()

        # Store the values of each parameter in a dictionary
        # the format is (NAME,VALUE)
        vsk_keys = [r.get('NAME') for r in root[0]]
        vsk_data = []
        for R in root[0]:
            val = (R.get('VALUE'))
            if val == None:
                val = 0
            vsk_data.append(float(val))

        # print vsk_keys
        if dict == False:
            vsk = {}
            for key, data in zip(vsk_keys, vsk_data):
                vsk[key] = data
            return vsk

        sm_dtype = []
        for key in vsk_keys:
            if key == "GCS":
                sm_dtype.append((key, 'f8', (3,3)))
            else:
                sm_dtype.append((key, 'f8'))
        sm_dtype.append(("FlatFoot", '?'))
        vsk_data.append(0)
        measurements_struct = np.array(tuple(vsk_data), dtype=sm_dtype)
        return measurements_struct
