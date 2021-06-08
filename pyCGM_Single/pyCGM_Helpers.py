import os
def getfilenames(x=1):

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