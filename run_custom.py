import os 

import numpy as np

import pycgm
from pycgm.model.utils.csv_diff import diff_pycgm_csv


script_dir = pycgm.get_data_dir()


@pycgm.Function.info(markers=["RFHD", "LFHD", "RBHD", "LBHD"],
                measurements=["HeadOffset"],
                        axes=["Head"],
        returns_measurements=['REyeDiameter', 'LEyeDiameter'])
def calibrate_eye_diameter(rfhd, lfhd, rbhd, lbhd, head_offset, head_axis):
        """
        Calibrate the eye diameter.
        Example of a function that returns a custom measurement.
        """

        r_diameter = np.average(rfhd - rbhd) + head_offset
        l_diameter = np.average(lfhd - lbhd) + head_offset

        return np.array([r_diameter, l_diameter])


@pycgm.Function.info(markers=["RFHD", "LFHD", "RBHD", "LBHD"],
                measurements=["Bodymass", "HeadOffset"],
                        axes=["Head"],
                returns_axes=['REye', 'LEye'])
def calc_axis_eye(bodymass, head_offset, rfhd, lfhd, rbhd, lbhd, head_axis):
        """
        Make the Eye Axis.
        Example of a function that returns a custom axis.
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


# Create a model with 2 dynamic trials
extended_model = pycgm.Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                            [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
                             os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

# Extend default CGM with custom functions
extended_model.insert_static_function(calibrate_eye_diameter, after='calc_static_head')
extended_model.insert_dynamic_function(calc_axis_eye, after='calc_axis_head')
extended_model.run()
diff_pycgm_csv(extended_model, 'RoboWalk', os.path.join(script_dir, 'Sample_2/pycgm_results.csv'))

# Access extended model outputs
print(f'{extended_model.data.static.calibrated.measurements.REyeDiameter=}')
print(f'{extended_model.data.dynamic.RoboWalk.axes.Pelvis.shape=}')
print(f'{extended_model.data.dynamic.RoboWalk.axes.REye.shape=}')

# Create the same model with predefined static and dynamic function sets
custom_model = pycgm.Model(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                          [os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), os.path.join(script_dir, 'ROM/Sample_Dynamic.c3d')], \
                           os.path.join(script_dir, 'Sample_2/RoboSM.vsk'),
                           static_functions=[calibrate_eye_diameter],
                           dynamic_functions=[calc_axis_eye])
custom_model.run()

# Access custom model outputs
print(f'{custom_model.data.static.calibrated.measurements.REyeDiameter=}')
print(f'{custom_model.data.dynamic.RoboWalk.axes.REye.shape=}')
