import os

import pycgm
import numpy as np

script_dir = pycgm.get_data_dir()

class Model_NewFunction(pycgm.Model):
    def __init__(self, static_trial, dynamic_trials, measurements):
        super().__init__(static_trial, dynamic_trials, measurements)

        # Add a custom function to the Model
        self.add_function('calc_axis_eye', measurements=["Bodymass", "HeadOffset"],
                                                markers=["RFHD", "LFHD", "RBHD", "LBHD"],
                                                   axes=["Head"],
                                           returns_axes=['REye', 'LEye'],
                                                  order=['calc_axis_head', 1]) 

    def calc_axis_eye(self, bodymass, head_offset, rfhd, lfhd, rbhd, lbhd, head_axis):
        """
        Make the Eye Axis.
        """

        num_frames = rfhd.shape[0]
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = np.zeros((num_frames, 3))

        r_eye_axis_stack = np.column_stack([x,y,z,o])
        l_eye_axis_stack = np.column_stack([x,y,z,o])

        r_eye_axis_matrix = r_eye_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        l_eye_axis_matrix = l_eye_axis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        # [ xx yx zx ox ] = [r/l]_eye_axis_matrix[0]
        # [ xy yy zy oy ] = [r/l]_eye_axis_matrix[1]
        # [ xz yz zz oz ] = [r/l]_eye_axis_matrix[2]

        return np.array([r_eye_axis_matrix, l_eye_axis_matrix])


# Model with additional function
model_extended = Model_NewFunction(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                                  [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
                                   os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

cgm = pycgm.PyCGM(model_extended)
cgm.run_all()

# Access model_extended output
print(f"{model_extended.data.dynamic.RoboWalk.axes.LEye.shape=}")
