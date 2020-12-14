#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM, StaticCGM
from pycgm.io import IO
import numpy as np


class CustomCGM1(CGM):
    """Sample custom class that doubles the values of all the pelvis axes."""

    @staticmethod
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        pelvis_axis = super(CustomCGM1, CustomCGM1).pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)
        return pelvis_axis * 2.0


class CustomCGM2(CGM):
    """Sample custom class that uses a custom static class to get a new value in measurements
    and use it within a customized calculation method."""

    def __init__(self, path_static, path_dynamic, path_measurements, cores=1, start=0, end=-1):
        static = CustomStaticCGM1(path_static, path_measurements)
        super().__init__(path_static, path_dynamic, path_measurements, static, cores, start, end)

    @staticmethod
    def hip_axis_calc(pelvis_axis, measurements):
        hip_axis = super(CustomCGM2, CustomCGM2).hip_axis_calc(pelvis_axis, measurements)
        multiplier = measurements['multiplier']
        return hip_axis * multiplier


class CustomStaticCGM1(StaticCGM):
    """Sample static class that gives back one extra subject measurement called 'multiplier'."""

    def get_static(self, flat_foot=False, gcs=None):
        cal_sm = super().get_static(flat_foot, gcs)
        cal_sm['multiplier'] = 2.0
        return cal_sm


class CustomCGM3(CGM):
    """Sample custom class that utilizes two extra imaginary markers, RWRI and LWRI.
    These replace the calculation that by default averages RWRA and RWRB, and LWRA and LWRB.

    They are only present in the dynamic trial."""

    @property
    def joint_marker_names(self):
        names = super().joint_marker_names
        names['Wrist'].extend('RWRI LWRI'.split())
        return names

    @staticmethod
    def wrist_axis_calc(markers, elbow_axis, measurements):
        rwra, rwrb, lwra, lwrb, rwri, lwri = markers

        mm = 7.0

        # Retrieve elbow joint centers
        rejc, lejc = elbow_axis[0], elbow_axis[4]

        # Calculate wrist joint centers again
        r_elbow_axis = elbow_axis[1:4]
        l_elbow_axis = elbow_axis[5:]
        r_elbow_flex = r_elbow_axis[1] - rejc
        l_elbow_flex = l_elbow_axis[1] - lejc

        # right
        x_axis = np.subtract(rwra, rwrb)
        x_axis = x_axis / np.linalg.norm(x_axis)

        z_axis = np.subtract(rejc, rwri)
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        r_wri_y_axis = y_axis

        # left
        x_axis = np.subtract(lwra, lwrb)
        x_axis = x_axis / np.linalg.norm(x_axis)

        z_axis = np.subtract(lejc, lwri)
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        l_wri_y_axis = y_axis

        r_wrist_thickness = measurements['RightWristWidth']
        l_wrist_thickness = measurements['LeftWristWidth']
        r_wrist_thickness = r_wrist_thickness / 2.0 + mm
        l_wrist_thickness = l_wrist_thickness / 2.0 + mm

        rwjc = rwri + r_wrist_thickness * r_wri_y_axis
        lwjc = lwri - l_wrist_thickness * l_wri_y_axis

        # Calculate wrist axis
        # right
        y_axis = r_elbow_flex
        y_axis = y_axis / np.array([np.linalg.norm(y_axis)])

        z_axis = np.subtract(rejc, rwjc)
        z_axis = z_axis / np.array([np.linalg.norm(z_axis)])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.array([np.linalg.norm(x_axis)])

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.array([np.linalg.norm(z_axis)])

        # Attach all the axes to wrist joint center.
        r_x_axis = x_axis + rwjc
        r_y_axis = y_axis + rwjc
        r_z_axis = z_axis + rwjc

        # left
        y_axis = l_elbow_flex
        y_axis = y_axis / np.array([np.linalg.norm(y_axis)])

        z_axis = np.subtract(lejc, lwjc)
        z_axis = z_axis / np.array([np.linalg.norm(z_axis)])

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.array([np.linalg.norm(x_axis)])

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.array([np.linalg.norm(z_axis)])

        # Attach all the axes to wrist joint center.
        l_x_axis = x_axis + lwjc
        l_y_axis = y_axis + lwjc
        l_z_axis = z_axis + lwjc

        return np.array([rwjc, r_x_axis, r_y_axis, r_z_axis, lwjc, l_x_axis, l_y_axis, l_z_axis])
