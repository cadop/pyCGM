#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM, StaticCGM
from math import sin, cos
import numpy as np


class CustomCGM1(CGM):
    """Sample custom class that doubles the values of all the pelvis axes"""

    @staticmethod
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        pelvis_axis = super(CustomCGM1, CustomCGM1).pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)
        return pelvis_axis * 2.0


class CustomCGM2(CGM):
    """Sample custom class that uses a custom static class to get a new value in measurements
    and use it within a customized calculation method"""

    def __init__(self, path_static, path_dynamic, path_measurements, cores=1, start=0, end=-1):
        static = CustomStaticCGM(path_static, path_measurements)
        super().__init__(path_static, path_dynamic, path_measurements, static, cores, start, end)

    @staticmethod
    def hip_axis_calc(pelvis_axis, measurements):
        hip_axis = super(CustomCGM2, CustomCGM2).hip_axis_calc(pelvis_axis, measurements)
        multiplier = measurements['multiplier']
        return hip_axis * multiplier


class CustomStaticCGM(StaticCGM):
    """Sample static class that gives back one extra subject measurement called 'multiplier'"""

    def get_static(self, flat_foot=False, gcs=None):
        cal_sm = super().get_static(flat_foot, gcs)
        cal_sm['multiplier'] = 2.0
        return cal_sm
