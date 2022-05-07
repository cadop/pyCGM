import numpy as np

from ..model.model import Model


class Model_NewFunction(Model):
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

