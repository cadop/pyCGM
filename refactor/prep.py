#!/usr/bin/python
# -*- coding: utf-8 -*-

# Utility
def normalize(v):
    """Normalizes an input vector

    Parameters
    ----------
    v : ndarray
        Input vector. 1-D list of floating point numbers.

    Returns
    -------
    ndarray
        Normalized form of input vector `v`. Returns `v` if its norm is 0.
    """


# Gap Filling
def target_name():
    """Creates a list of marker names.

    Returns
    -------
    target_names : array
        Empty list of marker names.
    """

def target_dict():
    """Creates a dictionary of marker to segment.

    Returns
    -------
    targetDict : dict
        Dict of marker to segment.
    """


def segment_dict():
    """Creates a dictionary of segments to marker names.

    Returns
    -------
    segmentDict : dict
        Dictionary of segments to marker names.
    """


def get_marker_location(pm, c):
    """Finds the location of the missing marker in the world frame.

    Parameters
    ----------
    pm : ndarray or list
        Location of the missing marker in the cluster frame. 1x3 list or
        numpy array.
    c : 2darray or list
        2d array or list indicating locations of other markers in the same
        cluster and same frame as the missing marker.
        Contains 3 1x3 lists or numpy arrays: `[X, Y, Z]`.

    Returns
    -------
    pw : array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    """


def get_static_transform(p, c):
    """Finds the location of the missing marker in the cluster frame.

    Parameters
    ----------
    p : ndarray or list
        1x3 numpy array or list: `[X, Y, Z]` indicating the location of
        the marker at a previously visible frame.
    c : 2darray or list
        2d array or list indicating locations of other markers in the same
        cluster as the missing marker. Contains 3 1x3 lists or numpy arrays:
        `[X, Y, Z]`.

    Returns
    -------
    pm : array
        Location of the missing marker in the cluster frame. List of
        3 elements `[X, Y, Z]`.
    """


def transform_from_static(data, data_mapping, static, static_mapping, key, useables, s):
    """Performs gap filling using static data.

    Uses static data to create an inverse transformation matrix that is stored
    between a 4 marker cluster. The matrix is then applied to estimate the position
    of the missing marker `key`.

    Parameters
    ----------
    data, static : 3darray
        3d numpy array of dynamic or static trial data respectively. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping, static_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data` or `static` respectively.
    key : str
        String indicating the missing marker.
    useables : list
        List of other markers in the same cluster as the missing
        marker `key`.
    s : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    """


def transform_from_mov(data, data_mapping, key, clust, last_time, i):
    """Performs gap filling using previous frames of motion capture data.

    Uses previous frames of motion capture data to create an inverse
    transformation matrix that is stored between a 4 marker cluster.
    The matrix is then applied to estimate the positionof the missing
    marker `key`.

    Parameters
    ----------
    data : 3darray
        3d numpy array of dynamic trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data`.
    key : str
        String indicating the missing marker.
    clust : list
        List of other markers in the same cluster as the missing
        marker `key`.
    last_time : int
        Frame number of the last frame in which all markers in `clust`
        and `key` were visible.
    i : int
        Frame number that the marker data is missing for.

    Returns
    -------
    array
        Location of the missing marker in the world frame. List of
        3 elements `[X, Y, Z]`.
    """


def segment_finder(key, data, data_mapping, target_dict, segment_dict, j, missings):
    """Find markers in the same cluster as `key` to use for gap filling.

    Finds markers in the same cluster as the marker `key` that have visible
    data that can be used to perform gap filling.

    Parameters
    ----------
    key : str
        String representing the missing marker.
    data : 3darray
        3d numpy array of dynamic trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data`.
    target_dict : dict
        Dict of marker to segment.
    segment_dict : dict
        Dictionary of segments to marker names.
    j : int
        Frame number that the marker data is missing for.
    missings : dict
        Dicionary of marker to list representing which other frames
        the marker is missing for.

    Returns
    -------
    useables : array
        List of marker names in the same cluster as the marker `key` that
        can be used for gap filling.
    """


def rigid_fill(data, data_mapping, static, static_mapping):
    """Fills in gaps in motion capture data.

    Estimates marker positions from previous marker positions
    or static data to fill in gaps in `data`. Calls either
    `transform_from_static` or `transform_from_mov` where
    appropriate to estimate positions of markers with missing data.

    Parameters
    ----------
    data, static : 3darray
        3d numpy array of dynamic or static trial data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    data_mapping, static_mapping : dict
        Dictionary that indicates which marker corresponds to which
        index in `data` or `static`.

    Returns
    -------
    3darray
        3d numpy array of the same format as `data` after gap filling
        has been performed.
    """


# Filtering
def butter_filter(data, cutoff_frequency, sampling_frequency):
    """Applies a fourth order Butterworth filter.

    Fourth order Butterworth filter to be used in filt() and filter_mask_nans()
    functions, which are in Utilities. Filter is applied forward and backwards
    with the filtfilt() function -- see Notes for more details.

    Parameters
    ----------
    data : 1darray or list
        Data to be filtered.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
        Sampling frequency signal was acquired at.

    Returns
    -------
    1darray
        1D numpy array of the signal after applying the filter.

    Notes
    -----
    Applying the filter one way will create a phase shift of the output
    signal compared to the input signal. For a 2nd order filter, this will
    be 90 degrees. Thus, filtfilt applies the signal once forward and once
    backward, which is referred to as phase correction. Whilst this brings
    the net phase shift to zero, it also means the cutoff of the filter will
    be twice as sharp when compared to a single filtering. In effect, a 2nd
    order filter applied twice will be a 4th order filter. We can apply a
    correction factor to the cuttoff frequency to compensate. Correction
    factor C = square root of 2 to the power of 1/n - 1, where n is equal to
    the number of passes.
    """


def filt(data, cutoff_frequency, sampling_frequency):
    """Applies a Butterworth filter to `data`.

    Takes in XYZ time series for one marker and loops over
    all 3 columns of `data`, applying `prep.butter_filter()` to each
    of them.

    Parameters
    ----------
    data : 2darray
        2d numpy array where each index of `data` contains a
        1x3 list of coordinate values: `[X, Y, Z]`.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
        Sampling frequency signal was acquired at.

    Returns
    -------
    filtered_data : 2darray
        2d numpy array of the same format as `data` after the Butterworth
        filter is applied.
    """


def filtering(data, cutoff_frequency, sampling_frequency):
    """Applies a Butterworth filter to motion capture data.

    This function takes in motion capture data for several markers,
    loops over all of them and calls `prep.filt()` on each one.

    Parameters
    ----------
    data : 3darray
        3d numpy array of motion capture data. Each index
        corresponds to a frame of trial. Each index holds a list of
        coordinate values for each marker in the trial.
        Each coordinate value is a 1x3 list: `[X, Y, Z]`.
    cutoff_frequency : int
        Desired cutoff frequency.
    sampling_frequency : int
        Sampling frequency signal was acquired at.

    Returns
    -------
    filtered_data : 3darray
        3d numpy array of the same format as `data` after the Butterworth
        filter is applied.
    """
