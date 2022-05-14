import os
from pycgm.CGMs.modified_function import Model_CustomPelvis
from pycgm.pyCGM import PyCGM

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir("..") # Move current path up

# Model with an overridden function
model_modified = Model_CustomPelvis('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
                                   ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
                                    'pycgm/SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM(model_modified)
cgm.run_all()

# Access model_modified output
print(f"{model_modified.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{model_modified.data.dynamic.Sample_Dynamic.angles.RHip.shape=}")

# Indexing cgm
print(f"{cgm[0].data.dynamic.RoboWalk.axes.REye.shape=}")
