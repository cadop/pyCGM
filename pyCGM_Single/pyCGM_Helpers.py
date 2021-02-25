"""
This file is used to get sample data.
"""

import os
def getfilenames(x=1):
    """ Get Filenames for sample data.

    Parameters
    ----------
    x : int, optional
        A flag (1, 2, or 3) that denotes which variation of files to retrieve.
        Default is 1 if not given.
        1 denotes the files in the `59993_Frame` directory.
        2 denotes the files in the `ROM` directory.
        3 denotes the files in the `Sample_2` directory.

    Returns
    -------
    dynamic_trial, static_trial, vsk_file, outputfile, CoM_output : tuple
        Returns the respective filenames in the relative path.
    
    Example
    -------
    >>> import os 
    >>> from .pyCGM_Helpers import getfilenames
    >>> import os
    >>> getfilenames() #doctest: +NORMALIZE_WHITESPACE
    ('SampleData/59993_Frame/59993_Frame_Dynamic.c3d', 
    'SampleData/59993_Frame/59993_Frame_Static.c3d', 
    'SampleData/59993_Frame/59993_Frame_SM.vsk', 
    'SampleData/59993_Frame/pycgm_results.csv', 
    'SampleData/59993_Frame/CoM')
    >>> getfilenames(2) #doctest: +NORMALIZE_WHITESPACE
    ('SampleData/ROM/Sample_Dynamic.c3d',
    'SampleData/ROM/Sample_Static.c3d',
    'SampleData/ROM/Sample_SM.vsk',
    'SampleData/ROM/pycgm_results.csv',
    'SampleData/ROM/CoM')
    >>> getfilenames(3) #doctest: +NORMALIZE_WHITESPACE
    ('SampleData/Sample_2/RoboWalk.c3d',
    'SampleData/Sample_2/RoboStatic.c3d',
    'SampleData/Sample_2/RoboSM.vsk',
    'SampleData/Sample_2/pycgm_results.csv',
    'SampleData/Sample_2/CoM')
    """
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir( scriptdir )
    os.chdir( ".." ) #move current path one up to the directory of pycgm_embed
    
    if x == 1:
        dir = 'SampleData/59993_Frame/'
        dynamic_trial = dir+'59993_Frame_Dynamic.c3d'
        static_trial =  dir+'59993_Frame_Static.c3d'
        vsk_file =      dir+'59993_Frame_SM.vsk'
        outputfile =    dir+'pycgm_results.csv'
        CoM_output =    dir+'CoM'
        
    if x == 2:
        dir = 'SampleData/ROM/'
        dynamic_trial = dir+'Sample_Dynamic.c3d' 
        static_trial =  dir+'Sample_Static.c3d' 
        vsk_file =      dir+'Sample_SM.vsk'     
        outputfile =    dir+'pycgm_results.csv'
        CoM_output =    dir+'CoM'
        
    if x == 3:
        dir = 'SampleData/Sample_2/'
        dynamic_trial = dir+'RoboWalk.c3d' 
        static_trial =  dir+'RoboStatic.c3d' 
        vsk_file =      dir+'RoboSM.vsk'     
        outputfile =    dir+'pycgm_results.csv'
        CoM_output =    dir+'CoM'
    
    return dynamic_trial,static_trial,vsk_file,outputfile,CoM_output