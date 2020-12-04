import pytest
import numpy as np
from refactor.pycgm import CGM
import os

rounding_precision = 8

class TestKineticsCGM:
    @classmethod
    def setup_class(self):
        """
        Called once for all tests in TestKineticsCGM.
        """
    
    @pytest.mark.parametrize("lhjc, rhjc, pelvis_axis, expected", [
        (np.array([282.57097863, 139.43231855, 435.52900012]),
         np.array([208.38050472, 122.80342417, 437.98979061]),
         np.array([[151.60830688, 291.74131775, 832.89349365], 
                   [251.74063624, 392.72694721, 1032.78850073], 
                   [250.61711554, 391.87232862, 1032.8741063], 
                   [251.60295336, 391.84795134, 1033.88777762]]),
         ([[245.47574167, 131.11787136, 436.75939536],
          [261.0890402, 155.43411628, 500.91761881]])),
        (np.array([984.96369008, 161.72241084, 714.78280362]),
         np.array([-570.727107, 409.48579719, 387.17336605]),
         np.array([[-553.90052549, -327.14438741, -4.58459872], 
                   [586.81782059, 994.852335, -164.15032491], 
                   [367.53692416, -193.11814502, 141.95648112], 
                   [814.64795266, 681.51439276, 87.63894117]]),
         ([[ 207.11829154,  285.60410402,  550.97808484],
          [1344.7944079 , 1237.35589455,  673.3680447 ]])),
        (np.array([624.42435686, 746.90148656, -603.71552902]),
         np.array([-651.87182756, -493.94862894, 640.38910712]),
         np.array([[691.47208667, 395.90359428, 273.23978111], 
                   [711.02920886, -701.16459687, 532.55441473], 
                   [-229.76970935, -650.15236712, 359.70108366], 
                   [222.81186893, 536.56366268, 386.21334066]]),
         ([[ -13.72373535,  126.47642881,   18.33678905],
          [ 627.86028973, 1671.50486946, 1130.43333413]]))
    ])
    def test_find_l5_pelvis(self, lhjc, rhjc, pelvis_axis, expected):
        """
        This function tests CGM.find_l5_pelvis(lhjc, rhjc, pelvis_axis),
        where lhjc and rhjc are numpy arrays that give the position of the
        LHJC and RHJC markers, and pelvis_axis gives the arrays of the pelvis origin,
        x-axis, y-axis, and z-axis.
        """
        result = CGM.find_l5_pelvis(lhjc, rhjc, pelvis_axis)
        np.testing.assert_almost_equal(result, expected, rounding_precision)
    
    def test_find_l5_thorax(self, lhjc, rhjc, thorax_axis, expected):
        """
        """
        pass