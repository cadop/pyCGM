import numpy as np
import sys
import xml.etree.ElementTree as ET
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
except ModuleNotFoundError:
    useEZC3D = False

class IO:
    # Utility Functions
    @property
    def marker_keys(self):
        """Returns a list of marker names that pycgm uses.

        Returns
        -------
        markers : list
            List of marker names.
        """
        return ['RASI','LASI','RPSI','LPSI','RTHI','LTHI','RKNE','LKNE','RTIB',
                'LTIB','RANK','LANK','RTOE','LTOE','LFHD','RFHD','LBHD','RBHD',
                'RHEE','LHEE','CLAV','C7','STRN','T10','RSHO','LSHO','RELB','LELB',
                'RWRA','RWRB','LWRA','LWRB','RFIN','LFIN']
    
    # Reading Functions
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
        """
        print(filename)
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
        >>> from refactor import io
        >>> import os
        >>> filename = 'SampleData' + os.sep + 'ROM' + os.sep + 'Sample_Static.csv'
        >>> data, mappings = io.IO.load_csv(filename)
        
        Test for the shape of data.
        
        >>> shape(data) #Indicates 275 frames, 141 points of data, 3 coordinates per point
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
        expected_markers = IO.marker_keys
        fh = open(filename,'r')
        fh = iter(fh)
        delimiter=','

        def row_to_array(row):
            frame = []
            if pyver == 2: row=zip(row[0::3],row[1::3],row[2::3])
            if pyver == 3: row=list(zip(row[0::3],row[1::3],row[2::3]))
            empty=np.asarray([np.nan,np.nan,np.nan],dtype=np.float64)
            for coordinates in row:
                try:
                    frame.append(np.float64(coordinates))
                except:
                    frame.append(empty.copy())
            return np.array(frame)

        def split_line(line):
            if pyver == 2: line = np.compat.asbytes(line).strip(np.compat.asbytes('\r\n'))
            elif pyver == 3: line = line.strip('\r\n')
            if line:
                return line.split(delimiter)
            else:
                return []

        def parse_trajectories(fh):
            data = []
            mappings = {}

            delimiter=','
            if pyver == 2:
                freq=np.float64(split_line(fh.next())[0])
                labels=split_line(fh.next())[1::3]
                fields=split_line(fh.next())
            elif pyver == 3:
                freq=np.float64(split_line(next(fh))[0])
                labels=split_line(next(fh))[1::3]
                fields=split_line(next(fh))
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
                #First elements with freq,labels,fields
                if pyver == 2: rows=[fh.next(),fh.next(),fh.next()]
                if pyver == 3: rows=[next(fh),next(fh),next(fh)]
                for j in fh:
                    if j.startswith("\r\n"):
                        break
                    rows.append(j)
                break
        rows=iter(rows)
        data, mappings = parse_trajectories(rows)

        if (len(expected_markers) > 0):
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
        >>> from refactor import io
        >>> import os
        >>> filename = 'SampleData' + os.sep + '59993_Frame' + os.sep + '59993_Frame_Static.c3d'
        >>> data, mappings = io.IO.load_c3d(filename)
        
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
        >>> mappings['*113'] #unlabeled marker
        113
        """
        data = []
        mappings = {}
        expected_markers = IO.marker_keys
        reader = c3d.Reader(open(filename, 'rb'))
        labels = reader.get('POINT:LABELS').string_array
        markers = [str(label.rstrip()) for label in labels]
        num_markers = len(markers)
        for i in range(len(markers)):
            marker = markers[i]
            if (marker in expected_markers):
                expected_markers.remove(marker)
            mappings[marker] = i

        for frame_no, points, analog in reader.read_frames(True, True):
            frame = []
            for label, point in zip(markers, points):
                frame.append(point)
            data.append(frame)
        
        if (len(expected_markers) > 0):
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
        dict : bool, optional
            Returns loaded subject measurement values as a dictionary if True.
            Otherwise, return as an array `[vsk_keys, vsk_data]`.
            True by default.

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

            If `dict` is False, return as an array `[keys, data]`, where keys is a
            list of the subject measurement labels, and data is a list of the
            corresponding values.
        """
        pass

    @staticmethod
    def load_sm_vsk(filename:
        """Open and load a vsk file with subject measurement data.

        Parameters
        ----------
        filename : str
            Path to the vsk file to be loaded.
        dict : bool, optional
            Returns loaded subject measurement values as a dictionary if True.
            Otherwise, return as an array `[vsk_keys, vsk_data]`.
            True by default.

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

            If `dict` is False, return as an array `[keys, data]`, where keys is a
            list of the subject measurement labels, and data is a list of the
            corresponding values.
        """
        pass

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
        dict : bool, optional
            Returns loaded subject measurement values as a dictionary if True.
            Otherwise, return as an array `[vsk_keys, vsk_data]`.
            True by default.

        Returns
        -------
        subject_measurements : dict
            Dictionary where keys are subject measurement labels, such as
            `Bodymass`, and values are the corresponding value.

            If `dict` is False, return as an array `[keys, data]`, where keys is a
            list of the subject measurement labels, and data is a list of the
            corresponding values.
        """
        pass

    # Writing Functions
    def write_result(filename, output_data, output_mapping, angles=True, axis=True, center_of_mass=True):
        """Writes outputs from pycgm to a CSV file.

        Lines 0-6 of the output csv are headers. Lines 7 and onwards
        are angle, axis, or center of mass results for each frame. For example,
        line 7 of the csv is output for frame 0 of the motion capture.
        The first element of each row of output is the frame number.

        Parameters
        ----------
        filename : str
            Path of the csv filename to write to.
        output_data : 3darray
            3d numpy array, where each index in the array corresponds to a frame
            of trial to write output for. Each index contains a list of 1x3 lists of
            XYZ coordinate data to write.s
        output_mapping : dict
            Dictionary where keys are output labels, such as `R Hip`, and values
            indicate which index in `output_data` corresponds to the data for that label.
        angles, axis : bool or list, optional
            Indicates whether or not to include the corresponding output in the
            written csv, or a list of angles/axes to write. All are True by default.
        center_of_mass : bool, optional
            Indicates wheter or not to include center of mass output in the written
            csv. True by default.
        """
        # The function will include list of 19 angles, 24 axes, and center of mass with the same
        # consistent naming scheme that already exists. See labelsAngs and labelsAxis in existing writeResult.
        pass
