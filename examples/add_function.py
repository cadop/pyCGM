import os
from pycgm.CGMs.additional_function import Model_NewFunction
from pycgm.pyCGM import PyCGM

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir("..") # Move current path up

# Model with additional function
model_extended = Model_NewFunction('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
                                  ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
                                   'pycgm/SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM(model_extended)
cgm.run_all()

# Access model_extended output
print(f"{model_extended.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{model_extended.data.dynamic.Sample_Dynamic.angles.RHip.shape=}")

# Indexing cgm
print(f"{cgm[0].data.dynamic.RoboWalk.axes.REye.shape=}")
