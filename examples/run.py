import os
from pycgm.model.model import Model
from pycgm.pyCGM import PyCGM

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir("..") # Move current path up

# Standard Model, 2 dynamic trials
model = Model('pycgm/SampleData/Sample_2/RoboStatic.c3d', \
             ['pycgm/SampleData/Sample_2/RoboWalk.c3d', 'pycgm/SampleData/ROM/Sample_Dynamic.c3d'], \
              'pycgm/SampleData/Sample_2/RoboSM.vsk')

cgm = PyCGM(model)
cgm.run_all()

# Access model output
print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
print(f"{model.data.dynamic.Sample_Dynamic.angles.RHip.shape=}")

# Indexing cgm
print(f"{cgm[0].data.dynamic.RoboWalk.axes.LKnee.shape=}")
