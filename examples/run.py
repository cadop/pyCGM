import os

import pycgm

script_dir = pycgm.get_data_dir()

# Standard Model, 2 dynamic trials
model = pycgm.Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                   [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
                    os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

cgm = pycgm.PyCGM(model)
cgm.run_all()

# Access model output
print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")

# Indexing cgm
print(f"{cgm[0].data.dynamic.RoboWalk.axes.LKnee.shape=}")
