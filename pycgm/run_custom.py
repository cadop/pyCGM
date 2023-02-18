import os 

from model.model import Model
from model.calc.function import Function

import numpy as np

def get_data_dir():
    """
    Returns the directory of the package.
    """
    return os.path.join(os.path.dirname(__file__), 'SampleData')

script_dir = get_data_dir()


@Function.info(markers=["RFHD", "LFHD", "RBHD", "LBHD"],
          measurements=["Bodymass", "HeadOffset"],
                  axes=["Head"],
          returns_axes=['REye', 'LEye'])
def calc_axis_eye(bodymass, head_offset, rfhd, lfhd, rbhd, lbhd, head_axis):
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

@Function.info(markers=["RFHD", "LFHD", "RBHD", "LBHD"],
          measurements=["Bodymass", "HeadOffset"],
                  axes=["Head"],
        returns_angles=['REye', 'LEye'])
def calc_angle_eye(bodymass, head_offset, rfhd, lfhd, rbhd, lbhd, head_axis):
        """
        Make the Eye Angle.
        """

        num_frames = rfhd.shape[0]
        r = np.zeros((num_frames, 3))
        l = np.zeros((num_frames, 3))

        eye_angle= np.array([r, l])

        return eye_angle


model = Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
             [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
              os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))
# Add function to existing set
model.insert_axis_function(calc_axis_eye, before='calc_axis_pelvis')
model.run()
#
# # Create model with predefined axis and angle function or function list
# model = Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
#              [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
#               os.path.join(script_dir, 'Sample_2/RoboSM.vsk'),
#               calc_axis_eye,
#               calc_angle_eye)
# model.run()
#
