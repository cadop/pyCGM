import os

import numpy as np
from csv_diff import diff_pycgm_csv
from model.model import Model

import pyCGM


def get_data_dir():
    """
    Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'SampleData')

script_dir = get_data_dir()

# Standard Model, 2 dynamic trials
# model = Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
#              [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
#               os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

model = Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
              os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), \
              os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

# Run model
model.run()

# Verify RoboWalk results
# diff_pycgm_csv(model, 'RoboWalk', os.path.join(script_dir, 'Sample_2/pycgm_results.csv'))

# Run set of models
# cgm = pyCGM.PyCGM([model, model, model, model])
# cgm.run_all()


# Access model output
# print(f"{model.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
