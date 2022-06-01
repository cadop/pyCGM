import os
from tkinter import W

import pycgm
import numpy as np

script_dir = pycgm.get_data_dir()

class Model_CustomPelvis(pycgm.Model):
    def __init__(self, static_trial, dynamic_trials, measurements):
        super().__init__(static_trial, dynamic_trials, measurements)

        # Override the calc_axis_pelvis function in Model
        self.modify_function('calc_axis_pelvis', measurements=["Bodymass", "ImaginaryMeasurement"],
                                                      markers=["RASI", "LASI", "RPSI", "LPSI", "SACR"],
                                                 returns_axes=['Pelvis'])


    def calc_axis_pelvis(self, bodymass, imaginary_measurement, rasi, lasi, rpsi, lpsi, sacr):
        """
        Make the Pelvis Axis.
        """

        num_frames = rasi.shape[0]
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = np.zeros((num_frames, 3))

        pel_axis_stack = np.column_stack([x,y,z,o])
        pel_axis_matrix = pel_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        # [ xx yx zx ox ] = [r/l]_eye_axis_matrix[0]
        # [ xy yy zy oy ] = [r/l]_eye_axis_matrix[1]
        # [ xz yz zz oz ] = [r/l]_eye_axis_matrix[2]

        return pel_axis_matrix



# Model with an overridden function
model_modified = Model_CustomPelvis(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                                         [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
                                          os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

cgm = pycgm.PyCGM(model_modified)
cgm.run_all()

# Access model_modified output
print(f"{model_modified.data.dynamic.RoboWalk.axes.Pelvis.shape=}")
