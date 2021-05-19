========
Overview
========

Getting Started
===============
**Ensure the code is running on your system**
    1. Clone or download the repository
    2. Run the *pycgm_embed.py* file
    3. Check the *SampleData* folder for a newly created *.csv* results file

**Change the file to calculate**
    1. Open the *pyCGM_Helpers.py* file
    2. Modify/Add a file directory and name
    3. Change the loading value in *main()* function of the *pycgm_embed.py* file

Setting up the calculation
##########################
Besides the pyCGM code, the three types of data needed are:
    1. A dictionary of subject measurements that correspond to the PlugInGait naming conventions from a vsk file
    2. An ordered list of data for a static trial where each index is a frame, and each frame contains a dictionary of marker names with a xyz value.
    3. An ordered list of data for a dynamic trial where each index is a frame, and each frame contains a dictionary of marker names with a xyz value.

While this data can be passed directly to the ``calcAngles()`` function in *pycgmCalc.py*, there are some easy helper functions that will load the data from a *.vsk* file, formated *.csv* file, and *.c3d* file through the *pycgmIO.py* module.

The global axis is defined differently in each file. Therefore you can change the rotation of the global axis before calling the ``calcAngles()`` function like so:

``calSM['GCS'] = pycgmStatic.rotmat(x=0,y=0,z=180)``

Static Calibration
##################
There are some automatic switches that will happen based on the data used.
    #. If the ASIS to Trocanter distances are not 0, the values will be used. Otherwise they will be calculated.
    #. If the Inter ASIS distance is not 0 it will be used, otherwise it is averaged from the static trial.
    #. If there is no Knee/Ankle width value or it is set to 0, it will be calculated by the MKN and MMA markers.


.. raw:: html

    <embed>
        <a href="../../source/UML.html">Interactive UML Diagram</a>
        <iframe src="../../source/UML.html" width="100%" height="400"></iframe> 
    </embed>