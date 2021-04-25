"""
This file is used to get sample data.
"""

import os


def getfilenames(filename):
    """ Get Filenames for sample data.

    The `59993_Frame` directory includes data of a human walking around while
    also performing other movements using all joints together.

    The `ROM` directory includes data of a human circling all joints
    individually while standing in one place.

    The `Sample_2` directory includes data of a robowalk.

    Parameters
    ----------
    filename : int, optional


    Returns
    -------
    String
        Returns the path for the specified file.

    Example
    -------
    >>> import os
    >>> from .helpers import getfilenames
    >>> getfilenames('59993_Frame_SM.vsk')
    'SampleData/59993_Frame/59993_Frame_SM.vsk'
    >>> getfilenames('Sample_Static.c3d')
    'SampleData/ROM/Sample_Static.c3d'
    >>> getfilenames('RoboWalk.c3d')
    'SampleData/Sample_2/RoboWalk.c3d'
    """

    frame = ['59993_Frame_Dynamic.c3d', '59993_Frame_SM.vsk',
             '59993_Frame_Static.c3d', 'pycgm_results.csv']
    ROM = ['Sample_Dynamic.c3d', 'Sample_Static.c3d',
           'Sample_SM.vsk', 'pycgm_results.csv']
    Sample = ['RoboWalk.c3d', 'RoboStatic.c3d',
              'RoboSM.vsk', 'pycgm_results.csv']

    if filename in frame:
        data_dir = 'SampleData/59993_Frame/'
    elif filename in ROM:
        data_dir = 'SampleData/ROM/'
    elif filename in Sample:
        data_dir = 'SampleData/Sample_2/'

    return os.path.join(data_dir, filename)


# print(getfilenames('RoboWalk.c3d'))
