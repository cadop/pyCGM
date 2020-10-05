import os
def getfilenames(x=1):
    """ Get Filenames function

    Parameters
    ----------
    x : int, optional
        A flag that denotes which variation of files to retrieve.
        Only works in the inclusive range of 1-3.
        Default is 1 if not given.

    Returns
    -------
    dynamic_trial, static_trial, vsk_file, outputfile, CoM_output : tuple
        Returns the respective filenames in the relative path.
    
    Example
    -------
    >>> from .pyCGM_Helpers import getfilenames
    >>> import os
    
    # Default is x = 1.
    >>> getfilenames() #doctest: +NORMALIZE_WHITESPACE
    ('SampleData/59993_Frame/59993_Frame_Dynamic.c3d', 
    'SampleData/59993_Frame/59993_Frame_Static.c3d', 
    'SampleData/59993_Frame/59993_Frame_SM.vsk', 
    'SampleData/59993_Frame/pycgm_results.csv', 
    'SampleData/59993_Frame/CoM')

    >>> x = 2
    >>> getfilenames(x) #doctest: +NORMALIZE_WHITESPACE
    ('SampleData/ROM/Sample_Dynamic.c3d', 
    'SampleData/ROM/Sample_Static.c3d', 
    'SampleData/ROM/Sample_SM.vsk', 
    'SampleData/ROM/pycgm_results.csv', 
    'SampleData/ROM/CoM')

    >>> x = 3
    >>> getfilenames(x) #doctest: +NORMALIZE_WHITESPACE
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