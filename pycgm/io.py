#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import xml.etree.ElementTree as ET

if sys.version_info[0] == 2:
    import c3d

    pyver = 2
    # print("Using python 2 c3d loader")

else:
    from pycgm import c3dpy3 as c3d

    pyver = 3
    # print("Using python 3 c3d loader - c3dpy3")

try:
    from ezc3d import c3d as ezc

    useEZC3D = True
    # print("EZC3D Found, using instead of Python c3d")
except ImportError:
    useEZC3D = False


class IO:
    # Utility Functions
    @staticmethod
    def marker_keys():
        """Returns a list of marker names that pycgm uses.

        Returns
        -------
        markers : list
            List of marker names.
        """
        return ['RASI', 'LASI', 'RPSI', 'LPSI', 'RTHI', 'LTHI', 'RKNE', 'LKNE', 'RTIB',
                'LTIB', 'RANK', 'LANK', 'RTOE', 'LTOE', 'LFHD', 'RFHD', 'LBHD', 'RBHD',
                'RHEE', 'LHEE', 'CLAV', 'C7', 'STRN', 'T10', 'RSHO', 'LSHO', 'RELB', 'LELB',
                'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']

    # Reading Functions
    @staticmethod
    def load_scaling_table():
        """Loads PlugInGait scaling table from segments.csv.

        Returns
        -------
        seg_scale : dict
            Dictionary containing segment scaling factors.

        Examples
        --------
        >>> from pycgm.io import IO
        >>> IO.load_scaling_table()
        {...

        Notes
        -----
        The PiG scaling factors are taken from Dempster -- they are available at:
        http://www.c-motion.com/download/IORGaitFiles/pigmanualver1.pdf
        """
        seg_scale = {}
        with open(os.path.dirname(os.path.abspath(__file__)) + '/segments.csv',
                  'r') as f:  # TODO: use __init__ file for this
            header = False
            for line in f:
                if not header:
                    header = True
                else:
                    row = line.rstrip('\n').split(',')
                    seg_scale[row[0]] = {'com': float(row[1]), 'mass': float(row[2]), 'x': float(row[3]),
                                         'y': float(row[4]), 'z': float(row[5])}
        return seg_scale

    @staticmethod
    def load_marker_data(filename):
        """Open and load a c3d or csv file of motion capture data.

        `filename` can be either a c3d or csv file. Depending on the file
        extension, `load_csv` or `load_c3d` will be called.

        Parameters
        ----------
        filename : str
            Path of the csv or c3d file to be loaded.

        Returns
        -------
        data, mappings : tuple
            `data` is a 3d numpy array. Each index `i` corresponds to frame `i`
            of trial. `data[i]` contains a list of coordinate values for each marker.
            Each coordinate value is a 1x3 list: [X, Y, Z].
            `mappings` is a dictionary that indicates which marker corresponds to which index
            in `data[i]`.

        Examples
        --------
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> csv_file = pycgm.get_robo_results()
        >>> c3d_file = pycgm.get_robo_data()[0]
        >>> csv_data, csv_mappings = IO.load_marker_data(csv_file)
        >>> c3d_data, c3d_mappings = IO.load_marker_data(c3d_file)

        Testing for some values from the loaded csv file.

        >>> csv_data[0][csv_mappings['RHNO']] #doctest: +NORMALIZE_WHITESPACE
        array([-772.184937, -312.352295, 589.815308])
        >>> csv_data[0][csv_mappings['C7']] #doctest: +NORMALIZE_WHITESPACE
        array([-1010.098999, 3.508968, 1336.794434])

        Testing for some values from the loaded c3d file.

        >>> c3d_data[0][c3d_mappings['RHNO']] #doctest: +NORMALIZE_WHITESPACE
        array([-259.45016479, -844.99560547, 1464.26330566])
        >>> c3d_data[0][c3d_mappings['C7']] #doctest: +NORMALIZE_WHITESPACE
        array([-2.20681717e+02, -1.07236075e+00, 1.45551550e+03])
        """
        # print(filename)
        if str(filename).endswith('.c3d'):
            return IO.load_c3d(filename)
        elif str(filename).endswith('.csv'):
            return IO.load_csv(filename)

    @staticmethod
    def load_csv(filename):
        """Open and load a csv file of motion capture data.

        Parameters
        ----------
        filename : str
            Path of the csv file to be loaded.

        Returns
        -------
        data, mappings : tuple
            `data` is a 3d numpy array. Each index `i` corresponds to frame `i`
            of trial. `data[i]` contains a list of coordinate values for each marker.
            Each coordinate value is a 1x3 list: [X, Y, Z].
            `mappings` is a dictionary that indicates which marker corresponds to which index
            in `data[i]`.

        Examples
        --------
        >>> from numpy import around, array, shape
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> filename = pycgm.get_rom_csv()
        >>> data, mappings = IO.load_csv(filename)

        Test for the shape of data.

        >>> shape(data)  # Indicates 275 frames, 141 points of data, 3 coordinates per point
        (275, 141, 3)

        Testing for some values from 59993_Frame_Static.c3d.

        >>> around(data[0][mappings['RHNO']], 8)
        array([ 811.9591064,  677.3413696, 1055.390991 ])
        >>> around(data[0][mappings['C7']], 8)
        array([ 250.765976,  165.616333, 1528.094116])
        >>> around(data[0][mappings['*113']], 8)
        array([ -82.65164185,  232.3781891 , 1361.853638  ])

        Testing for correct mappings.

        >>> mappings['RFHD']
        1
        >>> mappings['*113'] #unlabeled marker
        113
        """
        expected_markers = IO.marker_keys()
        fh = open(filename, 'r')
        fh = iter(fh)
        delimiter = ','

        def row_to_array(row):
            frame = []
            if pyver == 2:
                row = zip(row[0::3], row[1::3], row[2::3])
            elif pyver == 3:
                row = list(zip(row[0::3], row[1::3], row[2::3]))
            empty = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
            for coordinates in row:
                try:
                    frame.append(np.float64(coordinates))
                except ValueError:
                    frame.append(empty.copy())
            return np.array(frame)

        def split_line(line):
            if pyver == 2:
                line = np.compat.asbytes(line).strip(np.compat.asbytes('\r\n'))
            elif pyver == 3:
                line = line.strip('\r\n')
            if line:
                return line.split(delimiter)
            else:
                return []

        def parse_trajectories(fh):
            data = []
            mappings = {}

            delimiter = ','
            if pyver == 2:
                freq = np.float64(split_line(fh.next())[0])
                labels = split_line(fh.next())[1::3]
                fields = split_line(fh.next())
            elif pyver == 3:
                freq = np.float64(split_line(next(fh))[0])
                labels = split_line(next(fh))[1::3]
                fields = split_line(next(fh))
            delimiter = np.compat.asbytes(delimiter)
            for i in range(len(labels)):
                label = labels[i]
                if label in expected_markers:
                    expected_markers.remove(label)
                mappings[label] = i

            for row in fh:
                row = split_line(row)[1:]
                frame = row_to_array(row)
                data.append(frame)
            return data, mappings

        # Find the trajectories
        for i in fh:
            if i.startswith("TRAJECTORIES"):
                # First elements with freq,labels,fields
                if pyver == 2:
                    rows = [fh.next(), fh.next(), fh.next()]
                elif pyver == 3:
                    rows = [next(fh), next(fh), next(fh)]
                for j in fh:
                    if j.startswith("\r\n"):
                        break
                    rows.append(j)
                break
        rows = iter(rows)
        data, mappings = parse_trajectories(rows)

        if len(expected_markers) > 0:
            print("The following expected pycgm markers were not found in", filename, ":")
            print(expected_markers)
            print("pycgm functions may not work properly as a result.")
            print("Consider renaming the markers or adding the expected markers to the c3d file if they are missing.")

        data = np.array(data)
        return data, mappings

    @staticmethod
    def load_c3d(filename):
        """Open and load a c3d file of motion capture data.

        Parameters
        ----------
        filename : str
            Path of the c3d file to be loaded.

        Returns
        -------
        data, mappings : tuple
            `data` is a 3d numpy array. Each index `i` corresponds to frame `i`
            of trial. `data[i]` contains a list of coordinate values for each marker.
            Each coordinate value is a 1x3 list: [X, Y, Z].
            `mappings` is a dictionary that indicates which marker corresponds to which index
            in `data[i]`.

        Examples
        --------
        >>> from numpy import around, array, shape
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> filename = pycgm.get_59993_data()[0]
        >>> data, mappings = IO.load_c3d(filename)

        Test for the shape of data.

        >>> shape(data) #Indicates 371 frames, 189 points of data, 3 coordinates per point
        (371, 189, 3)

        Testing for some values from 59993_Frame_Static.c3d.

        >>> around(data[0][mappings['RHNO']], 8)
        array([ 555.46948242, -559.36499023, 1252.84216309])
        >>> around(data[0][mappings['C7']], 8)
        array([ -29.57296562,   -9.34280109, 1300.86730957])
        >>> around(data[0][mappings['*113']], 8)
        array([-173.22341919,  166.87660217, 1273.29980469])

        Testing for correct mappings.

        >>> mappings['RFHD']
        1
        >>> mappings['*113']  # unlabeled marker
        113
        """
        data = []
        mappings = {}
        expected_markers = IO.marker_keys()
        reader = c3d.Reader(open(filename, 'rb'))
        labels = reader.get('POINT:LABELS').string_array
        markers = [str(label.rstrip()) for label in labels]
        for i in range(len(markers)):
            marker = markers[i]
            if marker in expected_markers:
                expected_markers.remove(marker)
            mappings[marker] = i

        for frame_no, points, analog in reader.read_frames(True, True):
            frame = []
            for label, point in zip(markers, points):
                frame.append(point)
            data.append(frame)

        if len(expected_markers) > 0:
            print("The following expected pycgm markers were not found in", filename, ":")
            print(expected_markers)
            print("pycgm functions may not work properly as a result.")
            print("Consider renaming the markers or adding the expected markers to the c3d file if they are missing.")

        data = np.array(data)
        return data, mappings

    @staticmethod
    def load_sm(filename):
        """Open and load a file with subject measurement data.

        Subject measurements can be in a vsk or csv file. Depending on the file
        extension, `load_SM_vsk` or `load_SM_csv` will be called.

        Parameters
        ----------
        filename : str
            Path to the subject measurement file to be loaded

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

        Examples
        --------
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> vsk_filename, csv_filename = pycgm.get_robo_measurements()
        >>> vsk_subject_measurements = IO.load_sm(vsk_filename)
        >>> csv_subject_measurements = IO.load_sm(csv_filename)

        Testing for some values from loaded vsk file.

        >>> vsk_subject_measurements['Bodymass']
        72.0
        >>> vsk_subject_measurements['RightStaticPlantFlex']
        0.17637075483799

        Testing for some values from loaded csv file.

        >>> csv_subject_measurements['Bodymass']
        72.0
        >>> csv_subject_measurements['RightStaticPlantFlex']
        0.17637075483799
        """
        if str(filename).endswith('.vsk'):
            return IO.load_sm_vsk(filename)
        elif str(filename).endswith('.csv'):
            return IO.load_sm_csv(filename)

    @staticmethod
    def load_sm_vsk(filename):
        """Open and load a vsk file with subject measurement data.

        Parameters
        ----------
        filename : str
            Path to the vsk file to be loaded.

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

        Examples
        --------
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> filename = pycgm.get_robo_measurements()[0]
        >>> subject_measurements = IO.load_sm_vsk(filename)
        >>> subject_measurements['Bodymass']
        72.0
        >>> subject_measurements['RightStaticPlantFlex']
        0.17637075483799
        """
        # Check if the filename is valid
        # if not, return None
        if filename == '':
            return None

        # Create Dictionary to store values from VSK file
        subject_measurements = {}
        # Create an XML tree from file
        tree = ET.parse(filename)
        # Get the root of the file
        # <KinematicModel>
        root = tree.getroot()

        # Store the values of each parameter in a dictionary
        # the format is (NAME,VALUE)
        keys = [r.get('NAME') for r in root[0]]
        data = []
        for R in root[0]:
            val = (R.get('VALUE'))
            if val is None:
                val = 0
            data.append(float(val))

        for key, value in zip(keys, data):
            subject_measurements[key] = value

        return subject_measurements

    @staticmethod
    def load_sm_csv(filename):
        """Open and load a csv file with subject measurement data.

        csv files with subject measurements are lines of data in the following
        format:

        `key, value`

        where `key` is a subject measurement label, such as Bodymass, and `value`
        is its value.

        Parameters
        ----------
        filename : str
            Path to the csv file to be loaded.

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

        Examples
        --------
        >>> import pycgm
        >>> from pycgm.io import IO
        >>> filename = pycgm.get_robo_measurements()[1]
        >>> subject_measurements = IO.load_sm_csv(filename)
        >>> subject_measurements['Bodymass']
        72.0
        >>> subject_measurements['RightStaticPlantFlex']
        0.17637075483799
        """
        subject_measurements = {}
        with open(filename, 'r') as infile:
            for row in infile:
                line = row.strip().split(',')
                subject_measurements[line[0]] = np.float64(line[1])
        return subject_measurements

    # Writing Functions
    @staticmethod
    def write_result(filename, angle_output, angle_mapping, axis_output, axis_mapping, center_of_mass_output,
                     angles=None, axis=None, center_of_mass=True):
        """Writes outputs from pycgm to a CSV file.

        Lines 0-6 of the output csv are headers. Lines 7 and onwards
        are angle, axis, or center of mass results for each frame. For example,
        line 7 of the csv is output for frame 0 of the motion capture.
        The first element of each row of output is the frame number.

        Assumes that angle_output, axis_output, and center_of_mass_output are all
        of the same length - the number of frames in the trial.

        Parameters
        ----------
        filename : str
            Path of the csv filename to write to.
        angle_output, axis_output : 3darray
            3d numpy arrays, where each index in the array corresponds to a frame
            of trial to write output for. Each index contains an array of 1x3 arrays of
            XYZ coordinate data to write for an angle or axis.
        center_of_mass_output : 2darray
            2d numpy array where each index in the array corresponds to a frame of trial.
            Each index contains a 1x3 array of the XYZ coordinate of the center of mass for
            that frame.
        angle_mapping, axis_mapping : dict
            Dictionary where keys are angle or axis output labels, such as 'R Hip' or 'PELO', and values
            indicate which index in `angle_output` or `axis_output` corresponds to the data for that label.
        angles, axis : bool or list or tuple, optional
            Indicates whether or not to include the corresponding output in the
            written csv, or a list or tuple of angles/axes to write.
            If `angles` or `axis` is not specified or is True, all angles or axes will be written in
            the default order.
            If `angles` or `axis` is False, no angles or axes will be written.
            If `angles` or `axis` is a list or tuple, the angles or axes will be written in the order
            given by the labels in the list or tuple. The labels must exist in the respective mapping
            dictionaries `angle_mapping` or `axis_mapping`.
        center_of_mass : bool, optional
            Indicates whether or not to include center of mass output in the written
            csv. True by default.

        Examples
        --------
        >>> from numpy import array
        >>> import os
        >>> from shutil import rmtree
        >>> import tempfile
        >>> tmp_dir_name = tempfile.mkdtemp()
        >>> import pycgm
        >>> from pycgm.io import IO

        Prepare data to write to csv.

        >>> angle_output = [[[1, 2, 3], [4, 5, 6]]]
        >>> angle_mapping = {'Pelvis' : 0, 'R Hip' : 1}
        >>> axis_output = [[[1, 1, 1], [2, 2, 2]]]
        >>> axis_mapping = {'PELO': 0, 'HIPX' : 1}
        >>> center_of_mass_output = [[0, 1, 2]]
        >>> filename = os.path.join(tmp_dir_name, 'output.csv')

        Writing angles only.

        >>> IO.write_result(filename, angle_output, angle_mapping, axis_output, axis_mapping, center_of_mass_output,
        ...                 axis=False, center_of_mass = False)
        >>> with open(filename) as file:
        ...    lines = file.readlines()
        >>> result = lines[7].strip().split(',')
        >>> result = [float(i) for i in result]
        >>> result
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        Writing axis only.

        >>> IO.write_result(filename, angle_output, angle_mapping, axis_output, axis_mapping, center_of_mass_output,
        ...                 angles=False, center_of_mass = False)
        >>> with open(filename) as file:
        ...    lines = file.readlines()
        >>> result = lines[7].strip().split(',')
        >>> result = [float(i) for i in result]
        >>> result
        [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

        Writing center of mass only.
        >>> IO.write_result(filename, angle_output, angle_mapping, axis_output, axis_mapping, center_of_mass_output,
        ...                 angles=False, axis=False)
        >>> with open(filename) as file:
        ...    lines = file.readlines()
        >>> result = lines[7].strip().split(',')
        >>> result = [float(i) for i in result]
        >>> result
        [0.0, 0.0, 1.0, 2.0]

        >>> rmtree(tmp_dir_name)
        """
        # Default orders to write angles or axes:
        default_angle_labels = ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                                'L Ankle', 'R Foot', 'L Foot',
                                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']

        default_axis_labels = ["PELO", "PELX", "PELY", "PELZ", "HIPO", "HIPX", "HIPY", "HIPZ", "R KNEO",
                               "R KNEX", "R KNEY", "R KNEZ", "L KNEO", "L KNEX", "L KNEY", "L KNEZ", "R ANKO", "R ANKX",
                               "R ANKY", "R ANKZ", "L ANKO", "L ANKX", "L ANKY", "L ANKZ", "R FOOO", "R FOOX", "R FOOY",
                               "R FOOZ", "L FOOO", "L FOOX", "L FOOY", "L FOOZ", "HEAO", "HEAX", "HEAY", "HEAZ", "THOO",
                               "THOX", "THOY", "THOZ", "R CLAO", "R CLAX", "R CLAY", "R CLAZ", "L CLAO", "L CLAX",
                               "L CLAY", "L CLAZ", "R HUMO", "R HUMX", "R HUMY", "R HUMZ", "L HUMO", "L HUMX", "L HUMY",
                               "L HUMZ", "R RADO", "R RADX", "R RADY", "R RADZ", "L RADO", "L RADX", "L RADY", "L RADZ",
                               "R HANO", "R HANX", "R HANY", "R HANZ", "L HANO", "L HANX", "L HANY", "L HANZ"]

        if angles is False and axis is False and center_of_mass is False:
            return

        # Populate data_to_write
        data_to_write = []
        num_frames = len(angle_output)
        for i in range(num_frames):
            temp = [i]  # Frame number
            # Determine whether or not to write center_of_mass
            if center_of_mass:
                temp.extend(center_of_mass_output[i])
            # Get which angles to write
            if isinstance(angles, (list, tuple)):
                # Loop over angles
                for label in angles:
                    if label in angle_mapping:
                        temp.extend(angle_output[i][angle_mapping[label]])
            elif angles is None or angles is True:
                # Write all keys in default_angle_labels
                for label in default_angle_labels:
                    if label in angle_mapping:
                        temp.extend(angle_output[i][angle_mapping[label]])

            # Get which axes to write
            if isinstance(axis, (list, tuple)):
                # Loop over axis
                for label in axis:
                    if label in axis_mapping:
                        temp.extend(axis_output[i][axis_mapping[label]])
            elif axis is None or axis is True:
                # Write all keys in default_axis_labels
                for label in default_axis_labels:
                    if label in axis_mapping:
                        temp.extend(axis_output[i][axis_mapping[label]])
            data_to_write.append(temp)

        # Convert to numpy array
        data_to_write = np.array(data_to_write)

        # Determine header
        # Find which angles were actually written
        output_angle_labels = []
        if isinstance(angles, (list, tuple)):
            for label in angles:
                if label in angle_mapping:
                    output_angle_labels.append(label)
        elif angles is None or angles is True:
            for label in default_angle_labels:
                if label in angle_mapping:
                    output_angle_labels.append(label)

        output_axis_labels = []
        if isinstance(axis, (list, tuple)):
            for label in axis:
                if label in axis_mapping:
                    output_axis_labels.append(label)
        elif axis is None or axis is True:
            for label in default_axis_labels:
                if label in axis_mapping:
                    output_axis_labels.append(label)

        header = " ,"
        header_angles = ["Joint Angle,,,", ",x = flexion/extension angle", ",y = abduction/adduction angle",
                         ",z = external/internal rotation angle", ","]
        header_axis = ["Joint Coordinate", ",,,O = Origin", ",,,X = X axis orientation",
                       ",,,Y = Y axis orientation", ",,,Z = Z axis orientation"]

        has_angle_output = len(output_angle_labels) > 0
        has_axis_output = len(output_axis_labels) > 0
        has_com = center_of_mass

        for i in range(len(header_angles)):
            if has_com:
                header += ",,,"
            if has_angle_output:
                header += header_angles[i]
            if has_axis_output:
                header += ',' * 3 * (len(output_angle_labels) - 1)
                header += header_axis[i]
            header += '\n'

        header += ','
        if center_of_mass:
            header += "Center of Mass,,,"
        for label in output_angle_labels:
            header += label + ",,,"
        for label in output_axis_labels:
            header += label + ",,,"
        header += "\n"
        xyz = " Frame Number," + "X,Y,Z," * int(len(data_to_write[0][1:]) / 3)
        np.savetxt(filename, data_to_write, header=header + xyz, delimiter=',', fmt="%.15f")
