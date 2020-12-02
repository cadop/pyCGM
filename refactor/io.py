# Utility Functions
def marker_keys():
    """Returns a list of marker names that pycgm uses.

    Returns
    -------
    markers : list
        List of marker names.
    """
    pass


# Reading Functions
def load_data(filename):
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
    pass


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
    """
    pass


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
    """
    pass


def load_sm(filename, dict=True):
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


def load_sm_vsk(filename, dict=True):
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


def load_sm_csv(filename, dict=True):
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
