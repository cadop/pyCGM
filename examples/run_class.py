import os 

import numpy as np

import pycgm
from pycgm.model.utils.csv_diff import diff_pycgm_csv


script_dir = pycgm.get_data_dir()


class CustomStatic(pycgm.CalcStatic):
    def __init__(self):
        super().__init__()
        self.funcs.append(self.calibrate_double_leg_length)
        
    @pycgm.Function.info(measurements=['MeanLegLength'],
                 returns_measurements=['DoubleLegLength'])
    def calibrate_double_leg_length(mean_leg_length):
        return mean_leg_length * 2.0
        

class CustomDynamic(pycgm.CalcDynamic):

    @pycgm.Function.info(markers=['RASI', 'LASI', 'RPSI', 'LPSI', 'SACR'],
                    measurements=['DoubleLegLength'],
                    returns_axes=['Pelvis'])
    def calc_axis_pelvis(rasi, lasi, rpsi, lpsi, sacr, double_leg_length):
        """
        Make the Pelvis Axis.
        """
        print(f'{double_leg_length=}')
        if sacr is None:
            sacr = (rpsi + lpsi) / 2.0
        o = (rasi+lasi)/2.0
        b1 = o - sacr
        b2 = lasi - rasi
        y = b2 / np.linalg.norm(b2,axis=1)[:, np.newaxis]
        b3 = b1 - ( y * np.sum(b1*y,axis=1)[:, np.newaxis] )
        x = b3/np.linalg.norm(b3,axis=1)[:, np.newaxis]
        z = np.cross(x, y)
        num_frames = rasi.shape[0]
        pelvis_stack = np.column_stack([x,y,z,o])
        pelvis_matrix = pelvis_stack.reshape(num_frames,4,3).transpose(0,2,1)
        return pelvis_matrix


class Model_CustomPelvis(pycgm.Model):
    def __init__(self, static_filename, dynamic_filenames, measurement_filename, static_functions=None, dynamic_functions=None):
        super().__init__(static_filename, dynamic_filenames, measurement_filename, CustomStatic(), CustomDynamic())


model = Model_CustomPelvis(os.path.join(script_dir, 'Sample_2/RoboStatic.c3d'), \
                           os.path.join(script_dir, 'Sample_2/RoboWalk.c3d'), \
                           os.path.join(script_dir, 'Sample_2/RoboSM.vsk'))

model.run()
diff_pycgm_csv(model, 'RoboWalk', os.path.join(script_dir, 'Sample_2/pycgm_results.csv'))
