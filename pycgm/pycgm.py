#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from pycgm.io import IO
from math import cos, sin, acos, degrees, radians, sqrt, pi
import numpy as np
import sys

if sys.version_info[0] == 2:
    pyver = 2
else:
    pyver = 3

__all__ = ['CGM', 'StaticCGM', 'SubjectManager']


class CGM:

    def __init__(self, path_static, path_dynamic, path_measurements, static=None, cores=1, start=0, end=-1):
        """Initialization of CGM object function

        Instantiates various class attributes based on parameters and default values.

        Parameters
        ----------
        path_static : str
            File path of the static trial in csv or c3d form
        path_dynamic : str
            File path of the dynamic trial in csv or c3d form
        path_measurements : str
            File path of the subject measurements in csv or vsk form

        Examples
        --------
        >>> from pycgm.pycgm import CGM
        >>> import pycgm
        >>> static_trial, dynamic_trial, measurements = pycgm.get_rom_data()
        >>> subject1 = CGM(static_trial, dynamic_trial, measurements)
        """
        self.path_static = path_static
        self.path_dynamic = path_dynamic
        self.path_measurements = path_measurements
        self.static = static if static else StaticCGM(path_static, path_measurements)
        self.cores = cores
        self.angle_results = None
        self.axis_results = None
        self.com_results = None
        self.joint_centers = None
        self.marker_map = {marker: marker for marker in IO.marker_keys()}
        self.marker_data = None
        self.marker_idx = None
        self.measurements = None
        self.jc_idx = {}
        self.axis_idx = {}
        self.angle_idx = {}
        self.start = start
        self.end = end

        jc_labels = ['pelvis_origin', 'pelvis_x', 'pelvis_y', 'pelvis_z',
                     'thorax_origin', 'thorax_x', 'thorax_y', 'thorax_z',
                     'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RFoot', 'LFoot',
                     'RHEE', 'LHEE', 'C7', 'CLAV', 'STRN', 'T10', 'Front_Head', 'Back_Head',
                     'RShoulder', 'LShoulder', 'RHumerus', 'LHumerus', 'RRadius', 'LRadius',
                     'RHand', 'LHand']
        for i, label in enumerate(jc_labels):
            self.jc_idx[label] = i

        default_axis_labels = ["PELO", "PELX", "PELY", "PELZ", "HIPO", "HIPX", "HIPY", "HIPZ", "R KNEO",
                               "R KNEX", "R KNEY", "R KNEZ", "L KNEO", "L KNEX", "L KNEY", "L KNEZ", "R ANKO", "R ANKX",
                               "R ANKY", "R ANKZ", "L ANKO", "L ANKX", "L ANKY", "L ANKZ", "R FOOO", "R FOOX", "R FOOY",
                               "R FOOZ", "L FOOO", "L FOOX", "L FOOY", "L FOOZ", "HEAO", "HEAX", "HEAY", "HEAZ", "THOO",
                               "THOX", "THOY", "THOZ", "R CLAO", "R CLAX", "R CLAY", "R CLAZ", "L CLAO", "L CLAX",
                               "L CLAY", "L CLAZ", "R HUMO", "R HUMX", "R HUMY", "R HUMZ", "L HUMO", "L HUMX", "L HUMY",
                               "L HUMZ", "R RADO", "R RADX", "R RADY", "R RADZ", "L RADO", "L RADX", "L RADY", "L RADZ",
                               "R HANO", "R HANX", "R HANY", "R HANZ", "L HANO", "L HANX", "L HANY", "L HANZ"]
        for i, label in enumerate(default_axis_labels):
            self.axis_idx[label] = i

        default_angle_labels = ['Pelvis', 'R Hip', 'L Hip', 'R Knee', 'L Knee', 'R Ankle',
                                'L Ankle', 'R Foot', 'L Foot',
                                'Head', 'Thorax', 'Neck', 'Spine', 'R Shoulder', 'L Shoulder',
                                'R Elbow', 'L Elbow', 'R Wrist', 'L Wrist']
        for i, label in enumerate(default_angle_labels):
            self.angle_idx[label] = i

    # Customisation functions
    def remap(self, old, new):
        """Remap marker function

        Remaps a single marker from the expected name in CGM to a new one, using `old` and `new`.

        Parameters
        ----------
        old : str
            String containing the marker name that pycgm currently expects.
        new : str
            String containing the marker name to map `old` to.
        """
        self.marker_map[old] = new

    def full_remap(self, mapping):
        """Remap all markers function

        Uses the passed dictionary as the mapping for all markers.

        Parameters
        ----------
        mapping: dict
            Dictionary where each key is a string of pycgm's expected marker
            name and each value is a string of the new marker name.
        """
        self.marker_map = mapping

    # Input and output handlers
    def run(self):
        """Execute the CGM calculations function

        Loads in appropriate data from IO using paths.
        Performs any necessary prep on data.
        Runs the static calibration trial.
        Runs the dynamic trial to calculate all axes and angles.
        """

        # Get PlugInGait scaling table from segments.csv for use in center of mass calculations
        seg_scale = IO.load_scaling_table()

        self.marker_data, self.marker_idx = IO.load_marker_data(self.path_dynamic)
        self.end = self.end if self.end != -1 else len(self.marker_data)
        self.marker_data = self.marker_data[self.start:self.end]

        self.measurements = self.static.get_static()

        methods = [self.pelvis_axis_calc, self.hip_axis_calc, self.knee_axis_calc,
                   self.ankle_axis_calc, self.foot_axis_calc,
                   self.head_axis_calc, self.thorax_axis_calc, self.shoulder_axis_calc,
                   self.elbow_axis_calc, self.wrist_axis_calc, self.hand_axis_calc,
                   self.pelvis_angle_calc, self.hip_angle_calc, self.knee_angle_calc,
                   self.ankle_angle_calc, self.foot_angle_calc,
                   self.head_angle_calc, self.thorax_angle_calc, self.neck_angle_calc,
                   self.spine_angle_calc, self.shoulder_angle_calc, self.elbow_angle_calc, self.wrist_angle_calc]
        mappings = [self.marker_map, self.marker_idx, self.axis_idx, self.angle_idx, self.jc_idx]
        results = self.multi_calc(self.marker_data, methods, mappings, self.measurements, seg_scale)
        self.axis_results, self.angle_results, self.com_results, self.joint_centers = results

    def write_results(self, path_results, write_axes=None, write_angles=None, write_com=True):
        """Write CGM results to a csv file.

        Uses IO.write_result().

        Parameters
        ----------
        path_results : str, optional
            File path of the output file in csv or c3d form
        write_axes : bool or list, optional
            Boolean option to enable or disable writing of axis results to output file, or list
            of axis names to write. If not specified or True, all axes will be written.
        write_angles : bool or list, optional
            Boolean option to enable or disable writing of angle results to output file, or list
            of angle names to write. If not specified or True, all angles will be written.
        write_com : bool, optional
            Boolean option to enable or disable writing of center of mass results to output file.
            True by default.
        """
        # Ensure that write_results() is not being called before run().
        if self.angle_results is None or self.axis_results is None or self.com_results is None:
            print("Results cannot be saved before run() is called.")
        else:
            IO.write_result(path_results, self.angle_results, self.angle_idx, self.axis_results, self.axis_idx,
                            self.com_results, write_angles, write_axes, write_com)

    @staticmethod
    def multi_calc(data, methods, mappings, measurements, seg_scale, cores=1):
        """Multiprocessing calculation handler function

        Takes in the necessary information for performing each frame's calculation as parameters
        and distributes frames along available cores.

        Parameters
        ----------
        data : ndarray
            3d ndarray consisting of each frame by each marker by x, y, and z positions.
        methods : list
            List containing the calculation methods to be used.
        mappings : list
            List containing dictionary mappings for marker names and input and output indices.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        seg_scale : dict
            Segment scaling factors loaded in from the segments.csv file for use in center
            of mass calculations.
        cores : int, optional
            Number of cores to perform multiprocessing with, defaulting to 1 if not specified.

        Returns
        -------
        results : tuple
            A tuple consisting of the angle results and axis results. Angle results are
            stored as a 3d ndarray of each frame by each angle by x, y, and z. Axis results
            are stored as a 4d ndarray of each frame by each joint by origin and xyz unit vectors
            by x, y, and z location.
        """

        markers, marker_idx, axis_idx, angle_idx, jc_idx = mappings

        axis_results = np.empty((len(data), len(axis_idx), 3), dtype=float)
        axis_results.fill(np.nan)
        angle_results = np.empty((len(data), len(angle_idx), 3), dtype=float)
        angle_results.fill(np.nan)
        joint_centers = np.empty((len(data), len(jc_idx), 3), dtype=float)
        joint_centers.fill(np.nan)
        com_results = np.empty((len(data), 3), dtype=float)
        com_results.fill(np.nan)

        for i, frame in enumerate(data):
            frame_axes, frame_angles, frame_com, frame_jc = CGM.calc(frame, methods, mappings, measurements, seg_scale)
            axis_results[i] = frame_axes
            angle_results[i] = frame_angles
            com_results[i] = frame_com
            joint_centers[i] = frame_jc

        return axis_results, angle_results, com_results, joint_centers

    @staticmethod
    def calc(frame, methods, mappings, measurements, seg_scale):
        """Overall axis and angle calculation function

        Uses the data and methods passed in to distribute the appropriate inputs to each
        axis and angle calculation function (generally markers and axis results) and
        store and return their output, all in the context of a single frame.

        Parameters
        ----------
        frame : ndarray
            An nx3 ndarray consisting of each marker in the current frame and their x, y, and z positions,
            with n being the number of markers expected from the input.
        methods : list
            List containing the calculation methods to be used.
        mappings : list
            List containing dictionary mappings for marker names and input and output indices.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        seg_scale : dict
            Segment scaling factors loaded in from the segments.csv file for use in center
            of mass calculations.

        Returns
        -------
        results : tuple
            A tuple consisting of the axis results, angle results, center of mass results, and joint centers.
            Axis results are stored as a 3d ndarray of each joint by origin and xyz unit vectors
            by x, y, and z location. Angle results are stored as a 2d ndarray of each angle by x, y, and z.
        """

        pel_ax, hip_ax, kne_ax, ank_ax, foo_ax, \
        hea_ax, tho_ax, sho_ax, elb_ax, wri_ax, han_ax, \
        pel_an, hip_an, kne_an, ank_an, foo_an, \
        hea_an, tho_an, nec_an, spi_an, sho_an, elb_an, wri_an = methods  # Add upper when impl

        # markers maps expected marker name to its actual name in the input
        # marker_idx maps actual marker name from input to its index in the input
        # For example, if the input's first marker is RASIS, equivalent of RASI,
        # markers would translate RASI to RASIS and marker_idx would translate RASIS to 0
        markers, marker_idx, axis_idx, angle_idx, jc_idx = mappings

        joint_centers = np.empty((len(jc_idx), 3), dtype=float)
        joint_centers.fill(np.nan)
        axis_results = np.empty((len(axis_idx), 3), dtype=float)
        axis_results.fill(np.nan)
        angle_results = np.empty((len(angle_idx), 3), dtype=float)
        angle_results.fill(np.nan)
        com_results = np.empty(3, dtype=float)
        com_results.fill(np.nan)

        # Axis calculations

        rasi = frame[marker_idx[markers["RASI"]]]  # TODO: use this to figure out arbitrary marker counts
        lasi = frame[marker_idx[markers["LASI"]]]
        if "SACR" in markers:
            sacr = frame[marker_idx[markers["SACR"]]]
            pelvis_axis = pel_ax(rasi, lasi, sacr=sacr)
        elif "RPSI" in markers and "LPSI" in markers:
            rpsi = frame[marker_idx[markers["RPSI"]]]
            lpsi = frame[marker_idx[markers["LPSI"]]]
            pelvis_axis = pel_ax(rasi, lasi, rpsi=rpsi, lpsi=lpsi)
        else:
            raise ValueError("Required marker RPSI and LPSI, or SACR, missing")

        # Populate pelvis axis results
        pelo, pelx, pely, pelz = pelvis_axis
        axis_results[axis_idx["PELO"]] = pelo
        axis_results[axis_idx["PELX"]] = pelx
        axis_results[axis_idx["PELY"]] = pely
        axis_results[axis_idx["PELZ"]] = pelz

        # Populate pelvis_origin, pelvis_x, pelvis_y, pelvis_z in joint_centers
        joint_centers[jc_idx["pelvis_origin"]] = pelo
        joint_centers[jc_idx["pelvis_x"]] = pelx
        joint_centers[jc_idx["pelvis_y"]] = pely
        joint_centers[jc_idx["pelvis_z"]] = pelz

        hip_axis = hip_ax(pelvis_axis, measurements)

        # Populate hip axis results
        hipo, hipx, hipy, hipz = hip_axis[2:]
        axis_results[axis_idx["HIPO"]] = hipo
        axis_results[axis_idx["HIPX"]] = hipx
        axis_results[axis_idx["HIPY"]] = hipy
        axis_results[axis_idx["HIPZ"]] = hipz

        rthi = frame[marker_idx[markers["RTHI"]]]
        lthi = frame[marker_idx[markers["LTHI"]]]
        rkne = frame[marker_idx[markers["RKNE"]]]
        lkne = frame[marker_idx[markers["LKNE"]]]
        hip_origin = hip_axis[:2]

        # Populate LHip and RHip in joint_centers
        rhjc, lhjc = hip_origin
        joint_centers[jc_idx['RHip']] = rhjc
        joint_centers[jc_idx['LHip']] = lhjc

        knee_axis = kne_ax(rthi, lthi, rkne, lkne, hip_origin, measurements)

        # Populate left and right knee axis results
        r_kneo, r_knex, r_kney, r_knez = knee_axis[:4]
        axis_results[axis_idx["R KNEO"]] = r_kneo
        axis_results[axis_idx["R KNEX"]] = r_knex
        axis_results[axis_idx["R KNEY"]] = r_kney
        axis_results[axis_idx["R KNEZ"]] = r_knez

        l_kneo, l_knex, l_kney, l_knez = knee_axis[4:]
        axis_results[axis_idx["L KNEO"]] = l_kneo
        axis_results[axis_idx["L KNEX"]] = l_knex
        axis_results[axis_idx["L KNEY"]] = l_kney
        axis_results[axis_idx["L KNEZ"]] = l_knez

        # Populate RKnee and LKnee in joint_centers
        joint_centers[jc_idx['RKnee']] = r_kneo
        joint_centers[jc_idx['LKnee']] = l_kneo

        rtib = frame[marker_idx[markers["RTIB"]]]
        ltib = frame[marker_idx[markers["LTIB"]]]
        rank = frame[marker_idx[markers["RANK"]]]
        lank = frame[marker_idx[markers["LANK"]]]
        knee_origin = np.array([knee_axis[0], knee_axis[4]])

        ankle_axis = ank_ax(rtib, ltib, rank, lank, knee_origin, measurements)

        # Populate left and right ankle axis results
        r_anko, r_ankx, r_anky, r_ankz = ankle_axis[:4]
        axis_results[axis_idx["R ANKO"]] = r_anko
        axis_results[axis_idx["R ANKX"]] = r_ankx
        axis_results[axis_idx["R ANKY"]] = r_anky
        axis_results[axis_idx["R ANKZ"]] = r_ankz

        l_anko, l_ankx, l_anky, l_ankz = ankle_axis[4:]
        axis_results[axis_idx["L ANKO"]] = l_anko
        axis_results[axis_idx["L ANKX"]] = l_ankx
        axis_results[axis_idx["L ANKY"]] = l_anky
        axis_results[axis_idx["L ANKZ"]] = l_ankz

        # Populate RAnkle and LAnkle in joint_centers
        joint_centers[jc_idx['RAnkle']] = r_anko
        joint_centers[jc_idx['LAnkle']] = l_anko

        rtoe = frame[marker_idx[markers["RTOE"]]]
        ltoe = frame[marker_idx[markers["LTOE"]]]
        rhee = frame[marker_idx[markers["RHEE"]]]
        lhee = frame[marker_idx[markers["LHEE"]]]

        # Populate RHEE and LHEE in joint_centers
        joint_centers[jc_idx['RHEE']] = rhee
        joint_centers[jc_idx['LHEE']] = lhee

        foot_axis = foo_ax(rtoe, ltoe, ankle_axis, measurements)

        # Populate left and right foot axis results
        r_fooo, r_foox, r_fooy, r_fooz = foot_axis[:4]
        axis_results[axis_idx["R FOOO"]] = r_fooo
        axis_results[axis_idx["R FOOX"]] = r_foox
        axis_results[axis_idx["R FOOY"]] = r_fooy
        axis_results[axis_idx["R FOOZ"]] = r_fooz

        l_fooo, l_foox, l_fooy, l_fooz = foot_axis[4:]
        axis_results[axis_idx["L FOOO"]] = l_fooo
        axis_results[axis_idx["L FOOX"]] = l_foox
        axis_results[axis_idx["L FOOY"]] = l_fooy
        axis_results[axis_idx["L FOOZ"]] = l_fooz

        # Populate RFoot and LFoot in joint_centers
        joint_centers[jc_idx['RFoot']] = r_fooo
        joint_centers[jc_idx['LFoot']] = l_fooo

        rfhd = frame[marker_idx[markers["RFHD"]]]
        lfhd = frame[marker_idx[markers["LFHD"]]]
        rbhd = frame[marker_idx[markers["RBHD"]]]
        lbhd = frame[marker_idx[markers["LBHD"]]]

        # Populate Front_Head and Back_Head in joint_centers
        joint_centers[jc_idx['Front_Head']] = np.array([(rfhd + lfhd) / 2.0])
        joint_centers[jc_idx['Back_Head']] = np.array([(rbhd + lbhd) / 2.0])

        head_axis = hea_ax(rfhd, lfhd, rbhd, lbhd, measurements)
        heao, heax, heay, heaz = head_axis

        # Populate head axis results
        axis_results[axis_idx["HEAO"]] = heao
        axis_results[axis_idx["HEAX"]] = heax
        axis_results[axis_idx["HEAY"]] = heay
        axis_results[axis_idx["HEAZ"]] = heaz

        clav = frame[marker_idx[markers["CLAV"]]]
        c7 = frame[marker_idx[markers["C7"]]]
        strn = frame[marker_idx[markers["STRN"]]]
        t10 = frame[marker_idx[markers["T10"]]]

        # Populate C7, CLAV, STRN, and T10 in joint_centers
        joint_centers[jc_idx["C7"]] = c7
        joint_centers[jc_idx["CLAV"]] = clav
        joint_centers[jc_idx["STRN"]] = strn
        joint_centers[jc_idx["T10"]] = t10

        thorax_axis = tho_ax(clav, c7, strn, t10)
        thoo, thox, thoy, thoz = thorax_axis

        # Populate thorax axis results
        axis_results[axis_idx["THOO"]] = thoo
        axis_results[axis_idx["THOX"]] = thox
        axis_results[axis_idx["THOY"]] = thoy
        axis_results[axis_idx["THOZ"]] = thoz

        # Populate thorax_origin, thorax_x, thorax_y, and thorax_z in joint_centers
        joint_centers[jc_idx["thorax_origin"]] = thoo
        joint_centers[jc_idx["thorax_x"]] = thox
        joint_centers[jc_idx["thorax_y"]] = thoy
        joint_centers[jc_idx["thorax_z"]] = thoz

        rsho = frame[marker_idx[markers["RSHO"]]]
        lsho = frame[marker_idx[markers["LSHO"]]]

        wand = CGM.wand_marker(rsho, lsho, thorax_axis)

        shoulder_axis = sho_ax(rsho, lsho, thorax_axis[0], wand, measurements)
        shoulder_origin = np.array([shoulder_axis[0], shoulder_axis[4]])
        r_shoo, r_shox, r_shoy, r_shoz, l_shoo, l_shox, l_shoy, l_shoz = shoulder_axis

        # Populate shoulder axis results
        axis_results[axis_idx["R CLAO"]] = r_shoo
        axis_results[axis_idx["R CLAX"]] = r_shox
        axis_results[axis_idx["R CLAY"]] = r_shoy
        axis_results[axis_idx["R CLAZ"]] = r_shoz
        axis_results[axis_idx["L CLAO"]] = l_shoo
        axis_results[axis_idx["L CLAX"]] = l_shox
        axis_results[axis_idx["L CLAY"]] = l_shoy
        axis_results[axis_idx["L CLAZ"]] = l_shoz

        # Populate RShoulder and LShoulder in joint_centers
        joint_centers[jc_idx["RShoulder"]] = r_shoo
        joint_centers[jc_idx["LShoulder"]] = l_shoo

        relb = frame[marker_idx[markers["RELB"]]]
        lelb = frame[marker_idx[markers["LELB"]]]
        rwra = frame[marker_idx[markers["RWRA"]]]
        rwrb = frame[marker_idx[markers["RWRB"]]]
        lwra = frame[marker_idx[markers["LWRA"]]]
        lwrb = frame[marker_idx[markers["LWRB"]]]

        elbow_axis = elb_ax(relb, lelb, rwra, rwrb, lwra, lwrb, shoulder_origin, measurements)
        r_elbo, r_elbx, r_elby, r_elbz, l_elbo, l_elbx, l_elby, l_elbz = elbow_axis

        # Populate elbow axis results
        axis_results[axis_idx["R HUMO"]] = r_elbo
        axis_results[axis_idx["R HUMX"]] = r_elbx
        axis_results[axis_idx["R HUMY"]] = r_elby
        axis_results[axis_idx["R HUMZ"]] = r_elbz
        axis_results[axis_idx["L HUMO"]] = l_elbo
        axis_results[axis_idx["L HUMX"]] = l_elbx
        axis_results[axis_idx["L HUMY"]] = l_elby
        axis_results[axis_idx["L HUMZ"]] = l_elbz

        # Populate RHumerus and LHumers in joint_centers
        joint_centers[jc_idx["RHumerus"]] = r_elbo
        joint_centers[jc_idx["LHumerus"]] = l_elbo

        wrist_axis = wri_ax(rwra, rwrb, lwra, lwrb, elbow_axis, measurements)
        wrist_origin = np.array([wrist_axis[0], wrist_axis[4]])
        r_wrio, r_wrix, r_wriy, r_wriz, l_wrio, l_wrix, l_wriy, l_wriz = wrist_axis

        # Populate wrist axis results
        axis_results[axis_idx["R RADO"]] = r_wrio
        axis_results[axis_idx["R RADX"]] = r_wrix
        axis_results[axis_idx["R RADY"]] = r_wriy
        axis_results[axis_idx["R RADZ"]] = r_wriz
        axis_results[axis_idx["L RADO"]] = l_wrio
        axis_results[axis_idx["L RADX"]] = l_wrix
        axis_results[axis_idx["L RADY"]] = l_wriy
        axis_results[axis_idx["L RADZ"]] = l_wriz

        # Populate RRadius and LRadius in joint_centers
        joint_centers[jc_idx["RRadius"]] = r_wrio
        joint_centers[jc_idx["LRadius"]] = l_wrio

        rfin = frame[marker_idx[markers["RFIN"]]]
        lfin = frame[marker_idx[markers["LFIN"]]]

        hand_axis = han_ax(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_origin, measurements)
        r_hano, r_hanx, r_hany, r_hanz, l_hano, l_hanx, l_hany, l_hanz = hand_axis

        # Populate hand axis results
        axis_results[axis_idx["R HANO"]] = r_hano
        axis_results[axis_idx["R HANX"]] = r_hanx
        axis_results[axis_idx["R HANY"]] = r_hany
        axis_results[axis_idx["R HANZ"]] = r_hanz
        axis_results[axis_idx["L HANO"]] = l_hano
        axis_results[axis_idx["L HANX"]] = l_hanx
        axis_results[axis_idx["L HANY"]] = l_hany
        axis_results[axis_idx["L HANZ"]] = l_hanz

        # Populate RHand and LHand in joint_centers
        joint_centers[jc_idx["RHand"]] = r_hano
        joint_centers[jc_idx["LHand"]] = l_hano

        # Angle calculations
        # Pelvis
        pelvis_angle = pel_an(measurements['GCS'], pelvis_axis)
        angle_results[angle_idx["Pelvis"]] = pelvis_angle

        # Hip
        right_hip_angle, left_hip_angle = hip_an(hip_axis, knee_axis)
        angle_results[angle_idx["R Hip"]] = right_hip_angle
        angle_results[angle_idx["L Hip"]] = left_hip_angle

        # Knee
        right_knee_angle, left_knee_angle = kne_an(knee_axis, ankle_axis)
        angle_results[angle_idx["R Knee"]] = right_knee_angle
        angle_results[angle_idx["L Knee"]] = left_knee_angle

        # Ankle
        right_ankle_angle, left_ankle_angle = ank_an(ankle_axis, foot_axis)
        angle_results[angle_idx["R Ankle"]] = right_ankle_angle
        angle_results[angle_idx["L Ankle"]] = left_ankle_angle

        # Foot
        right_foot_angle, left_foot_angle = foo_an(measurements['GCS'], foot_axis)
        angle_results[angle_idx["R Foot"]] = right_foot_angle
        angle_results[angle_idx["L Foot"]] = left_foot_angle

        # Head
        head_angle = hea_an(measurements['GCS'], head_axis)
        angle_results[angle_idx["Head"]] = head_angle

        # Thorax
        thorax_angle = tho_an(thorax_axis)
        angle_results[angle_idx["Thorax"]] = thorax_angle

        # Neck
        neck_angle = nec_an(head_axis, thorax_axis)
        angle_results[angle_idx["Neck"]] = neck_angle

        # Spine
        spine_angle = spi_an(pelvis_axis, thorax_axis)
        angle_results[angle_idx["Spine"]] = spine_angle

        # Shoulder
        right_shoulder_angle, left_shoulder_angle = sho_an(thorax_axis, elbow_axis)
        angle_results[angle_idx["R Shoulder"]] = right_shoulder_angle
        angle_results[angle_idx["L Shoulder"]] = left_shoulder_angle

        # Elbow
        right_elbow_angle, left_elbow_angle = elb_an(elbow_axis, wrist_axis)
        angle_results[angle_idx["R Elbow"]] = right_elbow_angle
        angle_results[angle_idx["L Elbow"]] = left_elbow_angle

        # Wrist
        right_wrist_angle, left_wrist_angle = wri_an(wrist_axis, hand_axis)
        angle_results[angle_idx["R Wrist"]] = right_wrist_angle
        angle_results[angle_idx["L Wrist"]] = left_wrist_angle

        # Center of Mass calculations
        com_results = np.array(CGM.get_kinetics(joint_centers, jc_idx, seg_scale, measurements["Bodymass"]))

        return axis_results, angle_results, com_results, joint_centers

    # Utility functions
    @staticmethod
    def rotation_matrix(x=0, y=0, z=0):
        """Rotation Matrix function

        This function creates and returns a rotation matrix about axes x, y, z.

        Parameters
        ----------
        x, y, z : float, optional
            Angle, which will be converted to radians, in
            each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.

        Returns
        -------
        rxyz : ndarray
            The product of the matrix multiplication as a 3x3 ndarray.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> np.around(CGM.rotation_matrix(), 8)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> x, y, z = 25, 125, 225
        >>> np.around(CGM.rotation_matrix(x, y, z), 8)
        array([[ 0.40557979, -0.40557979,  0.81915204],
               [-0.8856487 , -0.39606407,  0.24240388],
               [ 0.22612258, -0.82379505, -0.51983679]])
        >>> x = 45
        >>> np.around(CGM.rotation_matrix(x), 8)
        array([[ 1.        ,  0.        ,  0.        ],
               [ 0.        ,  0.70710678, -0.70710678],
               [ 0.        ,  0.70710678,  0.70710678]])
        >>> x, y = -45, -90
        >>> np.around(CGM.rotation_matrix(x, y), 8)
        array([[ 0.        ,  0.        , -1.        ],
               [ 0.70710678,  0.70710678,  0.        ],
               [ 0.70710678, -0.70710678,  0.        ]])
        """
        # Convert the x, y, z rotation angles from degrees to radians
        x = radians(x)
        y = radians(y)
        z = radians(z)

        # Making elemental rotations about each of the x, y, z axes
        rx = np.array([[1, 0, 0], [0, cos(x), sin(x) * -1], [0, sin(x), cos(x)]])
        ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [sin(y) * -1, 0, cos(y)]])
        rz = np.array([[cos(z), sin(z) * -1, 0], [sin(z), cos(z), 0], [0, 0, 1]])

        # Making the rotation matrix around x, y, z axes using matrix multiplication
        rxy = np.matmul(rx, ry)
        rxyz = np.matmul(rxy, rz)

        return rxyz

    @staticmethod
    def subtract_origin(axis_vectors):
        """Subtract origin from axis vectors.

        Parameters
        ----------
        axis_vectors : ndarray
            numpy array containing 4 1x3 arrays - the origin vector, followed by
            the three X, Y, and Z axis vectors, each of which is a 1x3 numpy array
            of the respective X, Y, and Z components.

        Returns
        -------
        array
            numpy array containing 3 1x3 arrays of the X, Y, and Z axis vectors, after
            the origin is subtracted away.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> origin = [1, 2, 3]
        >>> x_axis = [4, 4, 4]
        >>> y_axis = [9, 9, 9]
        >>> z_axis = [-1, 0, 1]
        >>> axis_vectors = np.array([origin, x_axis, y_axis, z_axis])
        >>> CGM.subtract_origin(axis_vectors)
        array([[ 3,  2,  1],
               [ 8,  7,  6],
               [-2, -2, -2]])
        """
        # axis_vectors shouldn't need the array cast around it, but a lot of unit test inputs need to be fixed first
        origin, x_axis, y_axis, z_axis = np.array(axis_vectors)
        return np.array([x_axis - origin, y_axis - origin, z_axis - origin])

    @staticmethod
    def find_joint_center(a, b, c, delta):
        """Calculate the Joint Center function.

        This function is based on physical markers, a, b, and c, and joint center, which will be
        calculated in this function. All are in the same plane.

        Parameters
        ----------
        a, b, c : ndarray
            A 1x3 ndarray representing x, y, and z coordinates of the marker.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        mr : ndarray
            Returns the Joint Center x, y, z positions in a 1x3 ndarray.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> a, b, c = np.array([[468, 325, 673],
        ...                     [355, 365, 940 ],
        ...                     [452, 329 , 524]])
        >>> delta = 59
        >>> CGM.find_joint_center(a, b, c, delta)
        array([396.31601533, 347.48398669, 517.78420665])
        """
        # Make the two vector using 3 markers, which is on the same plane.
        v1 = a - c
        v2 = b - c

        # v3 is np.cross vector of v1, v2
        # and then it is normalized.
        v3 = np.cross(v1, v2)
        v3 = v3 / np.linalg.norm(v3)

        m = (b + c) / 2.0
        length = np.subtract(b, m)
        length = np.linalg.norm(length)

        theta = acos(delta / np.linalg.norm(v2))

        cs = cos(theta * 2)
        sn = sin(theta * 2)

        ux = v3[0]
        uy = v3[1]
        uz = v3[2]

        # This rotation matrix is called Rodrigues' rotation formula.
        # In order to make a plane, at least 3 markers are required which means
        # three physical markers on the segment can make a plane.
        # Then the orthogonal vector of the plane will be rotating axis.
        # Joint center is determined by rotating the one vector of plane around rotating axis.

        rot = np.matrix([[cs + ux ** 2.0 * (1.0 - cs), ux * uy * (1.0 - cs) - uz * sn, ux * uz * (1.0 - cs) + uy * sn],
                         [uy * ux * (1.0 - cs) + uz * sn, cs + uy ** 2.0 * (1.0 - cs), uy * uz * (1.0 - cs) - ux * sn],
                         [uz * ux * (1.0 - cs) - uy * sn, uz * uy * (1.0 - cs) + ux * sn, cs + uz ** 2.0 * (1.0 - cs)]])
        r = rot * (np.matrix(v2).transpose())
        r = r * length / np.linalg.norm(r)

        r = np.array([r[0, 0], r[1, 0], r[2, 0]])
        mr = np.array([r[0] + m[0], r[1] + m[1], r[2] + m[2]])

        return mr

    @staticmethod
    def wand_marker(rsho, lsho, thorax_axis):
        """Wand Marker Calculation function

        Takes in a dictionary of x,y,z positions and marker names.
        and takes the thorax axis.
        Calculates the wand marker for calculating the clavicle.

        Markers used: RSHO, LSHO

        Parameters
        ----------
        rsho, lsho : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_axis : ndarray
            A 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.

        Returns
        -------
        wand : ndarray
            Returns a 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rsho, lsho = np.array([[428, 270, 1500],
        ...                        [68, 269, 1510]])
        >>> thorax_axis = np.array([[256, 364, 1459],
        ...                         [256, 365, 1459],
        ...                         [257, 364, 1459],
        ...                         [256, 354, 1458]])
        >>> CGM.wand_marker(rsho, lsho, thorax_axis)
        array([[ 255.76812462,  364.        , 1459.9727455 ],
               [ 256.26181402,  364.        , 1459.96511834]])
        """

        # REQUIRED MARKERS:
        # RSHO
        # LSHO
        thor_o, thor_x, thor_y, thor_z = thorax_axis

        # Calculate for getting a wand marker

        # bring x axis from thorax axis
        axis_x_vec = thor_x - thor_o
        axis_x_vec = axis_x_vec / np.array(np.linalg.norm(axis_x_vec))

        rsho_vec = rsho - thor_o
        lsho_vec = lsho - thor_o
        rsho_vec = rsho_vec / np.array(np.linalg.norm(rsho_vec))
        lsho_vec = lsho_vec / np.array(np.linalg.norm(lsho_vec))

        r_wand = np.cross(rsho_vec, axis_x_vec)
        r_wand = r_wand / np.array(np.linalg.norm(r_wand))
        r_wand = thor_o + r_wand

        l_wand = np.cross(axis_x_vec, lsho_vec)
        l_wand = l_wand / np.array(np.linalg.norm(l_wand))
        l_wand = thor_o + l_wand
        wand = np.array([r_wand, l_wand])

        return wand

    @staticmethod
    def get_angle(axis_p, axis_d):
        """Normal angle calculation function

        This function takes in two axes, proximal and distal, and returns three angles.
        It uses inverse Euler rotation matrix in YXZ order.
        The output contains the angles in degrees.

        As we use arcsin, we have to care about if the angle is in area between -pi/2 to pi/2

        Parameters
        ----------
        axis_p : nparray
            The unit vectors of axis_p, the position of the proximal axis.
        axis_d : nparray
            The unit vectors of axis_d, the position of the distal axis.

        Returns
        -------
        angle : nparray
            Returns the flexion, abduction, and rotation angles in a 1x3 ndarray.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> axis_p = np.array([[ 0.1 ,   0.5,  0.6],
        ...                    [ 0.9,  -0.5,  -0.6],
        ...                    [-0.1,   0.7,  -0.8]])
        >>> axis_d = np.array([[-0.2,  -0.4,  -0.2],
        ...                    [ 0.8,  -0.1 , -0.6],
        ...                    [ 0.4,  -0.1 ,  0.5]])
        >>> CGM.get_angle(axis_p, axis_d)
        array([150.37625125,  -6.31531557,  82.93739733])
        """
        # This is the angle calculation, in which the order is Y-X-Z

        # Alpha is abduction angle.

        ang = ((-1 * axis_d[2][0] * axis_p[1][0]) +
               (-1 * axis_d[2][1] * axis_p[1][1]) +
               (-1 * axis_d[2][2] * axis_p[1][2]))
        alpha = np.nan
        if -1 <= ang <= 1:
            alpha = np.arcsin(ang)

        # Check the abduction angle is in the area between -pi/2 and pi/2
        # Beta is flexion angle
        # Gamma is rotation angle

        if -1.57079633 < alpha < 1.57079633:
            beta = np.arctan2(
                ((axis_d[2][0] * axis_p[0][0]) + (axis_d[2][1] * axis_p[0][1]) + (axis_d[2][2] * axis_p[0][2])),
                ((axis_d[2][0] * axis_p[2][0]) + (axis_d[2][1] * axis_p[2][1]) + (axis_d[2][2] * axis_p[2][2])))
            gamma = np.arctan2(
                ((axis_d[1][0] * axis_p[1][0]) + (axis_d[1][1] * axis_p[1][1]) + (axis_d[1][2] * axis_p[1][2])),
                ((axis_d[0][0] * axis_p[1][0]) + (axis_d[0][1] * axis_p[1][1]) + (axis_d[0][2] * axis_p[1][2])))
        else:
            beta = np.arctan2(
                -1 * ((axis_d[2][0] * axis_p[0][0]) + (axis_d[2][1] * axis_p[0][1]) + (axis_d[2][2] * axis_p[0][2])),
                ((axis_d[2][0] * axis_p[2][0]) + (axis_d[2][1] * axis_p[2][1]) + (axis_d[2][2] * axis_p[2][2])))
            gamma = np.arctan2(
                -1 * ((axis_d[1][0] * axis_p[1][0]) + (axis_d[1][1] * axis_p[1][1]) + (axis_d[1][2] * axis_p[1][2])),
                ((axis_d[0][0] * axis_p[1][0]) + (axis_d[0][1] * axis_p[1][1]) + (axis_d[0][2] * axis_p[1][2])))

        angle = np.array([beta, alpha, gamma]) * 180.0 / pi

        return angle

    @staticmethod
    def get_head_angle(axis_p, axis_d):
        """Head angle calculation function

        This function takes in two axes, proximal and distal, and returns three angles.
        It uses inverse Euler rotation matrix in YXZ order.
        The output contains the angles in degrees.

        As we use arcsin, we have to care about if the angle is in area between -pi/2 to pi/2

        Parameters
        ----------
        axis_p : nparray
            The unit vectors of axis_p, the position of the proximal axis.
        axis_d : nparray
            The unit vectors of axis_d, the position of the distal axis.

        Returns
        -------
        angle : nparray
            Returns the flexion, abduction, and rotation angles in a 1x3 ndarray.

        Examples
        --------
        >>> import numpy as np 
        >>> from pycgm.pycgm import CGM
        >>> axis_p = np.array([[  0.04,  0.99,  0.06],
        ...                    [ 0.99, -0.04, -0.05],
        ...                    [-0.05,  0.07, -0.99]])
        >>> axis_d = np.array([[-0.18, -0.98, -0.02],
        ...                    [ 0.71,  -0.11, -0.69],
        ...                    [ 0.67,  -0.14,  0.72]])
        >>> CGM.get_head_angle(axis_p, axis_d)
        array([ 185.18418007,  -39.99004074, -190.53848926])
        """
        # This is the angle calculation, in which the order is Y-X-Z

        # Alpha is abdcution angle.
        ang = ((-1 * axis_d[2][0] * axis_p[1][0]) +
               (-1 * axis_d[2][1] * axis_p[1][1]) +
               (-1 * axis_d[2][2] * axis_p[1][2]))
        alpha = np.nan
        if -1 <= ang <= 1:
            alpha = np.arcsin(ang)

        # Check the abduction angle is in the area between -pi/2 and pi/2
        # Beta is flexion angle
        # Gamma is rotation angle

        beta = np.arctan2(((axis_d[2][0] * axis_p[1][0]) +
                           (axis_d[2][1] * axis_p[1][1]) +
                           (axis_d[2][2] * axis_p[1][2])),
                          np.sqrt(pow(axis_d[0][0] * axis_p[1][0] +
                                      axis_d[0][1] * axis_p[1][1] +
                                      axis_d[0][2] * axis_p[1][2], 2) +
                                  pow(axis_d[1][0] * axis_p[1][0] +
                                      axis_d[1][1] * axis_p[1][1] +
                                      axis_d[1][2] * axis_p[1][2], 2)))

        alpha = np.arctan2(((axis_d[2][0] * axis_p[0][0]) +
                            (axis_d[2][1] * axis_p[0][1]) +
                            (axis_d[2][2] * axis_p[0][2])) * -1,
                           ((axis_d[2][0] * axis_p[2][0]) +
                            (axis_d[2][1] * axis_p[2][1]) +
                            (axis_d[2][2] * axis_p[2][2])))
        gamma = np.arctan2(((axis_d[0][0] * axis_p[1][0]) +
                            (axis_d[0][1] * axis_p[1][1]) +
                            (axis_d[0][2] * axis_p[1][2])) * -1,
                           ((axis_d[1][0] * axis_p[1][0]) +
                            (axis_d[1][1] * axis_p[1][1]) +
                            (axis_d[1][2] * axis_p[1][2])))

        alpha = 180.0 * alpha / pi
        beta = 180.0 * beta / pi
        gamma = 180.0 * gamma / pi

        beta *= -1

        if alpha < 0:
            alpha *= -1
        elif 0 < alpha < 180:
            alpha = 360 - alpha

        if gamma > 90.0:
            if gamma > 120:
                gamma = 180 - gamma
            else:
                gamma = -180 - gamma
        else:
            if gamma < 0:
                gamma = -180 - gamma
            else:
                gamma = -180.0 - gamma

        angle = np.array([alpha, beta, gamma])

        return angle

    @staticmethod
    def get_shoulder_angle(axis_p, axis_d):
        """Shoulder angle calculation function

        This function takes in two axes, proximal and distal, and returns three angles.
        It uses inverse Euler rotation matrix in YXZ order.
        The output contains the angles in degrees.

        As we use arcsin, we have to care about if the angle is in area between -pi/2 to pi/2

        Parameters
        ----------
        axis_p : nparray
            The unit vectors of axis_p, the position of the proximal axis.
        axis_d : nparray
            The unit vectors of axis_d, the position of the distal axis.

        Returns
        -------
        angle : nparray
            Returns the flexion, abduction, and rotation angles in a 1x3 ndarray.

        Examples
        --------
        >>> import numpy as np 
        >>> from pycgm.pycgm import CGM
        >>> axis_p = np.array([[  0.04,  0.99,  0.06],
        ...                    [ 0.99, -0.04, -0.05],
        ...                    [-0.05,  0.07, -0.99]])
        >>> axis_d = np.array([[-0.18, -0.98, -0.02],
        ...                    [ 0.71,  -0.11, -0.69],
        ...                    [ 0.67,  -0.14,  0.72]])
        >>> CGM.get_shoulder_angle(axis_p, axis_d)
        array([  -3.93357981, -140.068694  ,  172.89948535])
        """

        # Beta is flexion/extension
        # Gamma is adduction/abduction
        # Alpha is internal/external rotation

        # Shoulder angle calculation
        alpha = np.arcsin(((axis_d[2][0] * axis_p[0][0]) +
                           (axis_d[2][1] * axis_p[0][1]) +
                           (axis_d[2][2] * axis_p[0][2])))
        beta = np.arctan2(((axis_d[2][0] * axis_p[1][0]) +
                           (axis_d[2][1] * axis_p[1][1]) +
                           (axis_d[2][2] * axis_p[1][2])) * -1,
                          ((axis_d[2][0] * axis_p[2][0]) +
                           (axis_d[2][1] * axis_p[2][1]) +
                           (axis_d[2][2] * axis_p[2][2])))
        gamma = np.arctan2(((axis_d[1][0] * axis_p[0][0]) +
                            (axis_d[1][1] * axis_p[0][1]) +
                            (axis_d[1][2] * axis_p[0][2])) * -1,
                           ((axis_d[0][0] * axis_p[0][0]) +
                            (axis_d[0][1] * axis_p[0][1]) +
                            (axis_d[0][2] * axis_p[0][2])))

        angle = np.array([alpha, beta, gamma]) * 180.0 / pi

        return angle

    @staticmethod
    def get_spine_angle(axis_p, axis_d):
        """Spine angle calculation function

        This function takes in two axes, proximal and distal, and returns three angles.
        It uses inverse Euler rotation matrix in YXZ order.
        The output contains the angles in degrees.

        As we use arcsin, we have to care about if the angle is in area between -pi/2 to pi/2

        Parameters
        ----------
        axis_p : nparray
            The unit vectors of axis_p, the position of the proximal axis.
        axis_d : nparray
            The unit vectors of axis_d, the position of the distal axis.

        Returns
        -------
        angle : nparray
            Returns the flexion, abduction, and rotation angles in a 1x3 ndarray.

        Examples
        --------
        >>> import numpy as np 
        >>> from pycgm.pycgm import CGM
        >>> axis_p = np.array([[  0.04,  0.99,  0.06],
        ...                    [ 0.42, -0.04, -0.05],
        ...                    [-0.26,  0.07, -0.99]])
        >>> axis_d = np.array([[-0.18, -0.18, -0.62],
        ...                    [ 0.71,  -0.11, -0.69],
        ...                    [ 0.62,  -0.54,  0.12]])
        >>> CGM.get_spine_angle(axis_p, axis_d)
        array([-48.05098494,   8.04265812,  29.39317682])
        """

        alpha = np.arcsin(((axis_d[1][0] * axis_p[2][0]) +
                           (axis_d[1][1] * axis_p[2][1]) +
                           (axis_d[1][2] * axis_p[2][2])))
        gamma = np.arcsin(((-1 * axis_d[1][0] * axis_p[0][0]) +
                           (-1 * axis_d[1][1] * axis_p[0][1]) +
                           (-1 * axis_d[1][2] * axis_p[0][2])) / np.cos(alpha))
        beta = np.arcsin(((-1 * axis_d[0][0] * axis_p[2][0]) +
                          (-1 * axis_d[0][1] * axis_p[2][1]) +
                          (-1 * axis_d[0][2] * axis_p[2][2])) / np.cos(alpha))

        angle = np.array([beta, gamma, alpha]) * 180.0 / pi

        return angle

    @staticmethod
    def point_to_line(point, start, end):
        """Finds the distance from a point to a line.

        Calculates the distance from the point `point` to the line formed
        by the points `start` and `end`.

        Parameters
        ----------
        point, start, end : ndarray
            1x3 numpy arrays representing the XYZ coordinates of a point.
            `point` is a point not on the line.
            `start` and `end` form a line.

        Returns
        -------
        dist, nearest, point : tuple
            `dist` is the closest distance from the point to the line.
            `nearest` is the closest point on the line from `point`.
            It is represented as a 1x3 array.
            `point` is the original point not on the line.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> point = np.array([1, 2, 3])
        >>> start = np.array([4, 5, 6])
        >>> end = np.array([7, 8, 9])
        >>> dist, nearest, point = CGM.point_to_line(point, start, end)
        >>> np.around(dist, 8)
        5.19615242
        >>> np.around(nearest, 8)
        array([4., 5., 6.])
        >>> np.around(point, 8)
        array([1, 2, 3])
        """
        line_vector = end - start
        point_vector = point - start
        line_length = np.linalg.norm(line_vector)
        line_unit_vector = line_vector / line_length
        point_vector_scaled = point_vector * (1.0 / line_length)
        t = np.dot(line_unit_vector, point_vector_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        nearest = line_vector * t
        dist = np.linalg.norm(point_vector - nearest)
        nearest = nearest + start

        return dist, nearest, point

    @staticmethod
    def find_l5(lhjc, rhjc, axis):
        """Estimates the L5 marker position given the pelvis or thorax axis.

        Markers used : LHJC, RHJC

        Parameters
        ----------
        lhjc, rhjc : ndarray
            1x3 ndarray giving the XYZ coordinates of the LHJC and RHJC
            markers respectively.
        axis : ndarray
            Numpy array containing 4 1x3 arrays of pelvis or thorax origin, x-axis, y-axis,
            and z-axis. Only the z-axis affects the estimated L5 result.

        Returns
        -------
        mid_hip, l5 : tuple
            `mid_hip` is a 1x3 ndarray giving the XYZ coordinates of the middle
            of the LHJC and RHJC markers. `l5` is a 1x3 ndarray giving the estimated
            XYZ coordinates of the L5 marker.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> lhjc = np.array([308, 322, 937])
        >>> rhjc = np.array([182, 339, 935])
        >>> axis = np.array([[251, 391, 1032],
        ...                  [251, 392, 1032],
        ...                  [250, 391, 1032],
        ...                  [251, 391, 1033]])
        >>> np.around(CGM.find_l5(lhjc, rhjc, axis), 8)
        array([[ 245.        ,  330.5       ,  936.        ],
               [ 271.06445299,  371.10239489, 1043.26924277]])
        """
        # The L5 position is estimated as (LHJC + RHJC)/2 +
        # (0.0, 0.0, 0.828) * Length(LHJC - RHJC), where the value 0.828
        # is a ratio of the distance from the hip joint centre level to the
        # top of the lumbar 5: this is calculated as in the vertical (z) axis
        mid_hip = (lhjc + rhjc) / 2.0

        offset = np.linalg.norm(lhjc - rhjc) * 0.925
        origin, x_axis, y_axis, z_axis = axis
        norm_dir = z_axis / np.linalg.norm(z_axis)  # Create unit vector
        l5 = mid_hip + offset * norm_dir

        return mid_hip, l5

    # Axis calculation functions
    @staticmethod
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        """Pelvis Axis Calculation function

        Calculates the pelvis joint center and axis and returns them.

        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: CGM.cross product of x_axis and y_axis.

        Parameters
        ----------
        rasi, lasi : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        sacr, rpsi, lpsi : ndarray, optional
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.

        References
        ----------
        .. [12] Kadaba MP, Ramakrishnan HK, Wootten ME.
           Measurement of lower extremity kinematics during level walking.
           Journal of orthopaedic research: official publication of the Orthopaedic Research Society.
           1990;8(3):383-92.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rasi, lasi, rpsi, lpsi = np.array([[ 395,  428, 1036],
        ...                                    [ 183,  422, 1033],
        ...                                    [ 341,  246, 1055],
        ...                                    [ 255,  241, 1057]])
        >>> CGM.pelvis_axis_calc(rasi, lasi, rpsi=rpsi, lpsi=lpsi)
        array([[ 289.        ,  425.        , 1034.5       ],
               [ 288.97356163,  425.99275625, 1034.38279911],
               [ 288.00050025,  424.97171227, 1034.48585614],
               [ 288.98264324,  425.11676832, 1035.4930075 ]])
        >>> rasi, lasi, sacr = np.array([[ 395,  428, 1036],
        ...                              [ 183,  422, 1033],
        ...                              [ 294,  242, 1049]])
        >>> CGM.pelvis_axis_calc(rasi, lasi, sacr=sacr)
        array([[ 289.        ,  425.        , 1034.5       ],
               [ 288.97291419,  425.99651005, 1034.42104387],
               [ 288.00050025,  424.97171227, 1034.48585614],
               [ 288.98367201,  425.07853354, 1035.49677775]])
        """

        # REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        # If sacrum marker is present, use it
        if sacr is not None:
            sacrum = sacr
        # Otherwise mean of posterior markers is used as the sacrum
        else:
            sacrum = (rpsi + lpsi) / 2.0

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is the midpoint between RASI and LASI
        origin = (rasi + lasi) / 2.0

        # Calculate each axis; beta{n} are arbitrary names
        beta1 = origin - sacrum
        beta2 = lasi - rasi

        # Y_axis is normalized beta2
        y_axis = beta2 / np.linalg.norm(beta2)

        # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
        # and then normalized.
        beta3_cal = np.dot(beta1, y_axis) * y_axis
        beta3 = beta1 - beta3_cal
        x_axis = beta3 / np.array(np.linalg.norm(beta3))

        # Z-axis is np.cross product of x_axis and y_axis
        z_axis = np.cross(x_axis, y_axis)

        # Add the origin back to the vector
        y_axis += origin
        z_axis += origin
        x_axis += origin

        return np.array([origin, x_axis, y_axis, z_axis])

    @staticmethod
    def hip_axis_calc(pelvis_axis, measurements):
        """Hip Axis Calculation function

        Calculates the right and left hip joint center and axis and returns them.

        Other landmarks used: origin, sacrum
        Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure,
        InterAsisDistance, L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

        Parameters
        ----------
        pelvis_axis : ndarray
            A 4x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 6x3 ndarray that contains the right and left hip joint centers,
            hip origin, and hip x, y, and z axis components.

        References
        ----------
        .. [20]  Davis RB, Ounpuu S, Tyburski D, Gage JR.
           A gait analysis data collection and reduction technique. Human Movement Science.
           1991;10(5):575–587.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> pelvis_axis = np.array([[ 251, 391, 1032],
        ...                         [ 251, 392, 1032],
        ...                         [ 250, 391, 1032 ],
        ...                         [ 251, 391, 1033]])
        >>> measurements = {'MeanLegLength': 940, 'R_AsisToTrocanterMeasure': 72,
        ...                 'L_AsisToTrocanterMeasure': 72, 'InterAsisDistance': 215}
        >>> CGM.hip_axis_calc(pelvis_axis, measurements)
        array([[314.00929545, 341.01659272, 930.14188199],
               [187.99070455, 341.01659272, 930.14188199],
               [251.        , 341.01659272, 930.14188199],
               [251.        , 342.01659272, 930.14188199],
               [250.        , 341.01659272, 930.14188199],
               [251.        , 341.01659272, 931.14188199]])
        """

        # Requires
        # pelvis axis

        # Model's eigen value

        # LegLength
        # MeanLegLength
        # mm (marker radius)
        # interAsisMeasure

        # Set the variables needed to calculate the joint angle
        # Half of marker size
        mm = 7.0

        mean_leg_length = measurements['MeanLegLength']
        r_asis_to_trocanter_measure = measurements['R_AsisToTrocanterMeasure']
        l_asis_to_trocanter_measure = measurements['L_AsisToTrocanterMeasure']
        inter_asis_measure = measurements['InterAsisDistance']
        c = (mean_leg_length * 0.115) - 15.3
        theta = 0.500000178813934
        beta = 0.314000427722931
        aa = inter_asis_measure / 2.0
        s = -1

        # Hip Joint Center Calculation (ref. Davis_1991)

        # Calculate the distance to translate along the pelvis axis
        # Left
        l_xh = (-l_asis_to_trocanter_measure - mm) * cos(beta) + c * cos(theta) * sin(beta)
        l_yh = s * (c * sin(theta) - aa)
        l_zh = (-l_asis_to_trocanter_measure - mm) * sin(beta) - c * cos(theta) * cos(beta)

        # Right
        r_xh = (-r_asis_to_trocanter_measure - mm) * cos(beta) + c * cos(theta) * sin(beta)
        r_yh = (c * sin(theta) - aa)
        r_zh = (-r_asis_to_trocanter_measure - mm) * sin(beta) - c * cos(theta) * cos(beta)

        # Get the unit pelvis axis
        pelvis_x_axis, pelvis_y_axis, pelvis_z_axis = CGM.subtract_origin(pelvis_axis)

        # Multiply the distance to the unit pelvis axis
        l_hip_jc = pelvis_x_axis * l_xh + pelvis_y_axis * l_yh + pelvis_z_axis * l_zh
        r_hip_jc = pelvis_x_axis * r_xh + pelvis_y_axis * r_yh + pelvis_z_axis * r_zh

        l_hip_jc += pelvis_axis[0]
        r_hip_jc += pelvis_axis[0]

        # Get shared hip axis, it is inbetween the two hip joint centers
        hip_axis_center = (r_hip_jc + l_hip_jc) / 2.0

        # Translate pelvis axis to shared hip center
        # Add the origin back to the vector
        y_axis = pelvis_y_axis + hip_axis_center
        z_axis = pelvis_z_axis + hip_axis_center
        x_axis = pelvis_x_axis + hip_axis_center

        return np.array([r_hip_jc, l_hip_jc, hip_axis_center, x_axis, y_axis, z_axis])

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements):
        """Knee Axis Calculation function

        Calculates the right and left knee joint center and axis and returns them.

        Markers used: RTHI, LTHI, RKNE, LKNE
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        hip_origin : ndarray
            A 2x3 ndarray of the right and left hip origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right knee origin, right knee x, y, and z
            axis components, left knee origin, and left knee x, y, and z axis components.

        Modifies
        --------
        delta is changed suitably to knee.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rthi, lthi, rkne, lkne = np.array([[426, 262, 673],
        ...                                    [51 , 320, 723],
        ...                                    [416, 266, 524],
        ...                                    [84 , 286, 529]])
        >>> hip_origin = np.array([[309, 322, 937],
        ...                        [182, 339, 935]])
        >>> measurements = {'RightKneeWidth': 105, 'LeftKneeWidth': 105 }
        >>> CGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements)
        array([[363.28036639, 292.19664005, 515.36134954],
               [363.72612681, 293.091773  , 515.35546315],
               [362.39432212, 292.63691972, 515.21616214],
               [363.15299602, 292.26657445, 516.3507362 ],
               [142.89606353, 278.88369813, 524.43251176],
               [143.00280898, 279.86598662, 524.27851584],
               [141.90621372, 279.00329985, 524.50927627],
               [142.98988659, 279.0279367 , 525.41759676]])
        """
        # Get Global Values
        mm = 7.0
        r_knee_width = measurements['RightKneeWidth']
        l_knee_width = measurements['LeftKneeWidth']
        r_delta = (r_knee_width / 2.0) + mm
        l_delta = (l_knee_width / 2.0) + mm

        # REQUIRED MARKERS:
        # RTHI
        # LTHI
        # RKNE
        # LKNE
        # hip_JC

        r_hip_jc, l_hip_jc = hip_origin

        # Determine the position of kneeJointCenter using CGM.find_joint_center function
        r_knee_jc = CGM.find_joint_center(rthi, r_hip_jc, rkne, r_delta)
        l_knee_jc = CGM.find_joint_center(lthi, l_hip_jc, lkne, l_delta)

        # Right axis calculation
        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = r_hip_jc - r_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector np.cross vector.
        # the axis is then normalized.
        axis_x = np.cross(axis_z, rkne - r_hip_jc)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation
        # Z axis is Thigh bone calculated by the hipJC and kneeJC
        # the axis is then normalized
        axis_z = l_hip_jc - l_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector np.cross vector.
        # the axis is then normalized.
        axis_x = np.cross(lkne - l_hip_jc, axis_z)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then normalize it.
        r_knee_x_axis, r_knee_y_axis, r_knee_z_axis = r_axis
        r_knee_x_axis = r_knee_x_axis / np.array([np.linalg.norm(r_knee_x_axis)])
        r_knee_y_axis = r_knee_y_axis / np.array([np.linalg.norm(r_knee_y_axis)])
        r_knee_z_axis = r_knee_z_axis / np.array([np.linalg.norm(r_knee_z_axis)])
        l_knee_x_axis, l_knee_y_axis, l_knee_z_axis = l_axis
        l_knee_x_axis = l_knee_x_axis / np.array([np.linalg.norm(l_knee_x_axis)])
        l_knee_y_axis = l_knee_y_axis / np.array([np.linalg.norm(l_knee_y_axis)])
        l_knee_z_axis = l_knee_z_axis / np.array([np.linalg.norm(l_knee_z_axis)])

        # Put both axis in array
        # Add the origin back to the vector
        ry_axis = r_knee_y_axis + r_knee_jc
        rz_axis = r_knee_z_axis + r_knee_jc
        rx_axis = r_knee_x_axis + r_knee_jc

        # Add the origin back to the vector
        ly_axis = l_knee_y_axis + l_knee_jc
        lz_axis = l_knee_z_axis + l_knee_jc
        lx_axis = l_knee_x_axis + l_knee_jc

        return np.array([r_knee_jc, rx_axis, ry_axis, rz_axis, l_knee_jc, lx_axis, ly_axis, lz_axis])

    @staticmethod
    def ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements):
        """Ankle Axis Calculation function

        Calculates the right and left ankle joint center and axis and returns them.

        Markers used: RTIB, LTIB, RANK, LANK
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

        Parameters
        ----------
        rtib, ltib, rank, lank : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        knee_origin : ndarray
            A 2x3 ndarray of the right and left knee origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rtib, ltib, rank, lank = np.array([[433, 211, 273 ],
        ...                                    [50 , 235, 364],
        ...                                    [422, 217, 92 ],
        ...                                    [58 , 208, 86 ]])
        >>> knee_origin = np.array([[364, 292, 515],
        ...                         [143, 279, 524]])
        >>> measurements = {'RightAnkleWidth': 70, 'LeftAnkleWidth': 70,
        ...                 'RightTibialTorsion': 0, 'LeftTibialTorsion': 0}
        >>> CGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)
        array([[393.36370389, 247.2931075 ,  86.87260464],
               [394.09205433, 247.97797298,  86.85104297],
               [392.68188731, 248.01437196,  86.7505238 ],
               [393.2956466 , 247.39672623,  87.86489057],
               [ 98.11999534, 219.11196211,  80.44029928],
               [ 97.84148846, 220.06709428,  80.33952006],
               [ 97.16475735, 218.84739159,  80.57267311],
               [ 98.21976663, 219.24509728,  81.42636253]])
        """

        # Get Global Values
        r_ankle_width = measurements['RightAnkleWidth']
        l_ankle_width = measurements['LeftAnkleWidth']
        r_torsion = measurements['RightTibialTorsion']
        l_torsion = measurements['LeftTibialTorsion']
        mm = 7.0
        r_delta = (r_ankle_width / 2.0) + mm
        l_delta = (l_ankle_width / 2.0) + mm

        # REQUIRED MARKERS:
        # tib_R
        # tib_L
        # ank_R
        # ank_L
        # knee_JC

        knee_jc_r, knee_jc_l = knee_origin

        # This is Torsioned Tibia and this describes the ankle angles
        # Tibial frontal plane is being defined by ANK, TIB, and KJC

        # Determine the position of ankleJointCenter using CGM.find_joint_center function
        r_ankle_jc = CGM.find_joint_center(rtib, knee_jc_r, rank, r_delta)
        l_ankle_jc = CGM.find_joint_center(ltib, knee_jc_l, lank, l_delta)

        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        # Right axis calculation

        # Z axis is shank bone calculated by the ankleJC and kneeJC
        axis_z = knee_jc_r - r_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK, TIB, and KJC markers.
        # and calculated by each point's vector np.cross vector.
        # tib_ank_r vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_r = rtib - rank
        axis_x = np.cross(axis_z, tib_ank_r)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation

        # Z axis is shank bone calculated by the ankleJC and kneeJC
        axis_z = knee_jc_l - l_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK, TIB, and KJC markers.
        # and calculated by each point's vector np.cross vector.
        # tib_ank_l vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_l = ltib - lank
        axis_x = np.cross(tib_ank_l, axis_z)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then normalize it.
        r_ankle_x_axis, r_ankle_y_axis, r_ankle_z_axis = r_axis

        r_ankle_x_axis = r_ankle_x_axis / np.linalg.norm(r_ankle_x_axis)
        r_ankle_y_axis = r_ankle_y_axis / np.linalg.norm(r_ankle_y_axis)
        r_ankle_z_axis = r_ankle_z_axis / np.linalg.norm(r_ankle_z_axis)

        l_ankle_x_axis, l_ankle_y_axis, l_ankle_z_axis = l_axis

        l_ankle_x_axis = l_ankle_x_axis / np.linalg.norm(l_ankle_x_axis)
        l_ankle_y_axis = l_ankle_y_axis / np.linalg.norm(l_ankle_y_axis)
        l_ankle_z_axis = l_ankle_z_axis / np.linalg.norm(l_ankle_z_axis)

        # Put both axis in array
        r_axis = np.array([r_ankle_x_axis, r_ankle_y_axis, r_ankle_z_axis])
        l_axis = np.array([l_ankle_x_axis, l_ankle_y_axis, l_ankle_z_axis])

        # Rotate the axes about the tibia torsion.
        r_torsion = np.radians(r_torsion)
        l_torsion = np.radians(l_torsion)

        r_axis = np.array([[cos(r_torsion) * r_axis[0][0] - sin(r_torsion) * r_axis[1][0],
                            cos(r_torsion) * r_axis[0][1] - sin(r_torsion) * r_axis[1][1],
                            cos(r_torsion) * r_axis[0][2] - sin(r_torsion) * r_axis[1][2]],
                           [sin(r_torsion) * r_axis[0][0] + cos(r_torsion) * r_axis[1][0],
                            sin(r_torsion) * r_axis[0][1] + cos(r_torsion) * r_axis[1][1],
                            sin(r_torsion) * r_axis[0][2] + cos(r_torsion) * r_axis[1][2]],
                           [r_axis[2][0], r_axis[2][1], r_axis[2][2]]])

        l_axis = np.array([[cos(l_torsion) * l_axis[0][0] - sin(l_torsion) * l_axis[1][0],
                            cos(l_torsion) * l_axis[0][1] - sin(l_torsion) * l_axis[1][1],
                            cos(l_torsion) * l_axis[0][2] - sin(l_torsion) * l_axis[1][2]],
                           [sin(l_torsion) * l_axis[0][0] + cos(l_torsion) * l_axis[1][0],
                            sin(l_torsion) * l_axis[0][1] + cos(l_torsion) * l_axis[1][1],
                            sin(l_torsion) * l_axis[0][2] + cos(l_torsion) * l_axis[1][2]],
                           [l_axis[2][0], l_axis[2][1], l_axis[2][2]]])

        # Add the origin back to the vector
        rx_axis = r_axis[0] + r_ankle_jc
        ry_axis = r_axis[1] + r_ankle_jc
        rz_axis = r_axis[2] + r_ankle_jc

        lx_axis = l_axis[0] + l_ankle_jc
        ly_axis = l_axis[1] + l_ankle_jc
        lz_axis = l_axis[2] + l_ankle_jc

        return np.array([r_ankle_jc, rx_axis, ry_axis, rz_axis, l_ankle_jc, lx_axis, ly_axis, lz_axis])

    @staticmethod
    def foot_axis_calc(rtoe, ltoe, ankle_axis, measurements):
        """Foot Axis Calculation function

        Calculates the right and left foot joint axis by rotating uncorrect foot joint axes about offset angle.
        Returns the foot axis origin and axis.

        In case of foot joint center, we've already make 2 kinds of axis for static offset angle.
        and then, Call this static offset angle as an input of this function for dynamic trial.

        Special Cases:

        (anatomical uncorrect foot axis)
        If foot flat is true, then make the reference markers instead of HEE marker
        which height is as same as TOE marker's height.
        otherwise, foot flat is false, use the HEE marker for making Z axis.

        Markers used: RTOE, LTOE
        Other landmarks used: ANKLE_FLEXION_AXIS
        Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex, LeftStaticRotOff, LeftStaticPlantFlex

        Parameters
        ----------
        rtoe, ltoe : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Modifies
        --------
        Axis changes following to the static info.

        you can set the static_info by the button. and this will calculate the offset angles
        the first setting, the foot axis show foot uncorrected anatomical
        reference axis(Z_axis point to the AJC from TOE)

        if press the static_info button so if static_info is not None,
        and then the static offsets angles are applied to the reference axis.
        the reference axis is Z axis point to HEE from TOE

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rtoe, ltoe = np.array([[442, 381, 42],
        ...                        [39 , 382, 41]])
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394 , 248, 87],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [98 , 219, 80],
        ...                        [98 , 220, 80],
        ...                        [97 , 219, 80],
        ...                        [98 , 219, 81]])
        >>> measurements = {'RightStaticRotOff': 0.1, 'LeftStaticRotOff': 0.1,
        ...                 'RightStaticPlantFlex': 0.3, 'LeftStaticPlantFlex': 0.2}
        >>> CGM.foot_axis_calc(rtoe, ltoe, ankle_axis, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[442.        , 381.        ,  42.        ],
               [442.54940367, 380.73530701,  42.79252333],
               [441.29240336, 381.35704172,  42.60977717],
               [441.55563237, 380.10419933,  42.00886416],
               [ 39.        , 382.        ,  41.        ],
               [ 39.06596884, 382.04580459,  41.99676981],
               [ 38.02896253, 381.77304869,  41.0746949 ],
               [ 39.22963959, 381.02717163,  41.02950626]])
        """

        # REQUIRED MARKERS:
        # RTOE
        # LTOE

        # REQUIRED JOINT CENTER & AXIS
        # ANKLE JOINT CENTER
        # ANKLE FLEXION AXIS

        ankle_jc_r = ankle_axis[0]
        ankle_jc_l = ankle_axis[4]
        ankle_flexion_r = ankle_axis[2]
        ankle_flexion_l = ankle_axis[6]

        # Toe axis's origin is marker position of TOE
        toe_jc_r = rtoe
        toe_jc_l = ltoe

        # HERE IS THE INCORRECT AXIS

        # Right
        # Z axis is from TOE marker to AJC, normalized.
        r_axis_z = ankle_jc_r - rtoe
        r_axis_z = r_axis_z / np.linalg.norm(r_axis_z)

        # Bring the flexion axis of ankle axes from ankle_axis, and normalize it.
        y_flex_r = ankle_flexion_r - ankle_jc_r
        y_flex_r = y_flex_r / np.linalg.norm(y_flex_r)

        # X axis is calculated as a np.cross product of Z axis and ankle flexion axis.
        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x = r_axis_x / np.linalg.norm(r_axis_x)

        # Y axis is then perpendicularly calculated from Z axis and X axis, and normalized.
        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y = r_axis_y / np.linalg.norm(r_axis_y)

        r_foot_axis = np.array([r_axis_x, r_axis_y, r_axis_z])

        # Left
        # Z axis is from TOE marker to AJC, normalized.
        l_axis_z = ankle_jc_l - ltoe
        l_axis_z = l_axis_z / np.linalg.norm(l_axis_z)

        # Bring the flexion axis of ankle axes from ankle_axis, and normalize it.
        y_flex_l = ankle_flexion_l - ankle_jc_l
        y_flex_l = y_flex_l / np.linalg.norm(y_flex_l)

        # X axis is calculated as a np.cross product of Z axis and ankle flexion axis.
        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x = l_axis_x / np.linalg.norm(l_axis_x)

        # Y axis is then perpendicularly calculated from Z axis and X axis, and normalized.
        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y = l_axis_y / np.linalg.norm(l_axis_y)

        l_foot_axis = np.array([l_axis_x, l_axis_y, l_axis_z])

        # Apply static offset angle to the incorrect foot axes

        # Static offset angle are taken from static_info variable in radians.
        r_alpha = measurements['RightStaticRotOff']
        r_beta = measurements['RightStaticPlantFlex']
        l_alpha = measurements['LeftStaticRotOff']
        l_beta = measurements['LeftStaticPlantFlex']

        r_alpha = np.around(degrees(r_alpha), decimals=5)
        r_beta = np.around(degrees(r_beta), decimals=5)
        l_alpha = np.around(degrees(l_alpha), decimals=5)
        l_beta = np.around(degrees(l_beta), decimals=5)

        r_alpha = -radians(r_alpha)
        r_beta = radians(r_beta)
        l_alpha = radians(l_alpha)
        l_beta = radians(l_beta)

        # Rotate incorrect foot axis around y axis first.
        # Right
        r_rotmat = [[(cos(r_beta) * r_foot_axis[0][0] + sin(r_beta) * r_foot_axis[2][0]),
                     (cos(r_beta) * r_foot_axis[0][1] + sin(r_beta) * r_foot_axis[2][1]),
                     (cos(r_beta) * r_foot_axis[0][2] + sin(r_beta) * r_foot_axis[2][2])],
                    [r_foot_axis[1][0], r_foot_axis[1][1], r_foot_axis[1][2]],
                    [(-1 * sin(r_beta) * r_foot_axis[0][0] + cos(r_beta) * r_foot_axis[2][0]),
                     (-1 * sin(r_beta) * r_foot_axis[0][1] + cos(r_beta) * r_foot_axis[2][1]),
                     (-1 * sin(r_beta) * r_foot_axis[0][2] + cos(r_beta) * r_foot_axis[2][2])]]
        # Left
        l_rotmat = [[(cos(l_beta) * l_foot_axis[0][0] + sin(l_beta) * l_foot_axis[2][0]),
                     (cos(l_beta) * l_foot_axis[0][1] + sin(l_beta) * l_foot_axis[2][1]),
                     (cos(l_beta) * l_foot_axis[0][2] + sin(l_beta) * l_foot_axis[2][2])],
                    [l_foot_axis[1][0], l_foot_axis[1][1], l_foot_axis[1][2]],
                    [(-1 * sin(l_beta) * l_foot_axis[0][0] + cos(l_beta) * l_foot_axis[2][0]),
                     (-1 * sin(l_beta) * l_foot_axis[0][1] + cos(l_beta) * l_foot_axis[2][1]),
                     (-1 * sin(l_beta) * l_foot_axis[0][2] + cos(l_beta) * l_foot_axis[2][2])]]

        # Rotate incorrect foot axis around x axis next.
        # Right
        r_rotmat = [[r_rotmat[0][0], r_rotmat[0][1], r_rotmat[0][2]],
                    [(cos(r_alpha) * r_rotmat[1][0] - sin(r_alpha) * r_rotmat[2][0]),
                     (cos(r_alpha) * r_rotmat[1][1] - sin(r_alpha) * r_rotmat[2][1]),
                     (cos(r_alpha) * r_rotmat[1][2] - sin(r_alpha) * r_rotmat[2][2])],
                    [(sin(r_alpha) * r_rotmat[1][0] + cos(r_alpha) * r_rotmat[2][0]),
                     (sin(r_alpha) * r_rotmat[1][1] + cos(r_alpha) * r_rotmat[2][1]),
                     (sin(r_alpha) * r_rotmat[1][2] + cos(r_alpha) * r_rotmat[2][2])]]

        # Left
        l_rotmat = [[l_rotmat[0][0], l_rotmat[0][1], l_rotmat[0][2]],
                    [(cos(l_alpha) * l_rotmat[1][0] - sin(l_alpha) * l_rotmat[2][0]),
                     (cos(l_alpha) * l_rotmat[1][1] - sin(l_alpha) * l_rotmat[2][1]),
                     (cos(l_alpha) * l_rotmat[1][2] - sin(l_alpha) * l_rotmat[2][2])],
                    [(sin(l_alpha) * l_rotmat[1][0] + cos(l_alpha) * l_rotmat[2][0]),
                     (sin(l_alpha) * l_rotmat[1][1] + cos(l_alpha) * l_rotmat[2][1]),
                     (sin(l_alpha) * l_rotmat[1][2] + cos(l_alpha) * l_rotmat[2][2])]]

        # Bring each x,y,z axis from rotation axes
        r_axis_x, r_axis_y, r_axis_z = r_rotmat
        l_axis_x, l_axis_y, l_axis_z = l_rotmat

        # Attach each axis to the origin
        rx_axis = r_axis_x + toe_jc_r
        ry_axis = r_axis_y + toe_jc_r
        rz_axis = r_axis_z + toe_jc_r

        lx_axis = l_axis_x + toe_jc_l
        ly_axis = l_axis_y + toe_jc_l
        lz_axis = l_axis_z + toe_jc_l

        return np.array([toe_jc_r, rx_axis, ry_axis, rz_axis, toe_jc_l, lx_axis, ly_axis, lz_axis])

    @staticmethod
    def head_axis_calc(rfhd, lfhd, rbhd, lbhd, measurements):
        """Head Axis Calculation function

        Calculates the head joint center and axis and returns them.

        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        rfhd, lfhd, rbhd, lbhd : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the head origin and the
            head x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> measurements = {'HeadOffset': 0.3}
        >>> rfhd, lfhd, rbhd, lbhd = np.array([[325, 402, 1722],
        ...                                    [184, 409, 1721],
        ...                                    [304, 242, 1694],
        ...                                    [197, 251, 1696]])
        >>> CGM.head_axis_calc(rfhd, lfhd, rbhd, lbhd, measurements)
        array([[ 254.5       ,  405.5       , 1721.5       ],
               [ 254.52565516,  406.49058239, 1721.36550709],
               [ 253.50032966,  405.52555761, 1721.49754796],
               [ 254.50100837,  405.63451148, 1722.49091152]])
        """

        # Get Global Values
        head_off = measurements['HeadOffset'] * -1

        # get the midpoints of the head to define the sides
        front = (rfhd + lfhd) / 2.0
        back = (rbhd + lbhd) / 2.0
        right = (rfhd + rbhd) / 2.0
        left = (lfhd + lbhd) / 2.0
        origin = front

        # Get the vectors from the sides with primary x axis facing front
        # First get the x direction
        x_vec = front - back
        x_vec = x_vec / np.linalg.norm(x_vec)

        # get the direction of the y axis
        y_vec = left - right
        y_vec = y_vec / np.linalg.norm(y_vec)

        # get z axis by np.cross-product of x axis and y axis.
        z_vec = np.cross(x_vec, y_vec)
        z_vec = z_vec / np.linalg.norm(z_vec)

        # make sure all x,y,z axis is orthogonal each other by np.cross-product
        y_vec = np.cross(z_vec, x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)

        # rotate the head axis around y axis about head offset angle.
        x_vec_rot = np.array([x_vec[0] * cos(head_off) + z_vec[0] * sin(head_off),
                              x_vec[1] * cos(head_off) + z_vec[1] * sin(head_off),
                              x_vec[2] * cos(head_off) + z_vec[2] * sin(head_off)])
        y_vec_rot = y_vec
        z_vec_rot = np.array([x_vec[0] * -1 * sin(head_off) + z_vec[0] * cos(head_off),
                              x_vec[1] * -1 * sin(head_off) + z_vec[1] * cos(head_off),
                              x_vec[2] * -1 * sin(head_off) + z_vec[2] * cos(head_off)])

        # Add the origin back to the vector to get it in the right position
        x_axis = x_vec_rot + origin
        y_axis = y_vec_rot + origin
        z_axis = z_vec_rot + origin

        return np.array([origin, x_axis, y_axis, z_axis])

    @staticmethod
    def thorax_axis_calc(clav, c7, strn, t10):
        """Thorax Axis Calculation function

        Calculates the thorax joint center and axis and returns them.

        Markers used: CLAV, C7, STRN, T10

        Parameters
        ----------
        clav, c7, strn, t10 : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> clav, c7, strn, t10 = np.array([[256, 371, 1459],
        ...                                 [256, 371, 1459],
        ...                                 [251, 414, 1292],
        ...                                 [228, 192, 1279]])
        >>> CGM.thorax_axis_calc(clav, c7, strn, t10)
        array([[ 255.4938561 ,  364.51623363, 1461.58932269],
               [ 255.56616237,  365.44248597, 1461.21941945],
               [ 256.48733126,  364.41655085, 1461.5339111 ],
               [ 255.405658  ,  364.15275055, 1460.66190631]])
        """

        # Set or get a marker size as mm
        marker_size = 7.0

        # Get the midpoints of the upper and lower sections, as well as the front and back sections
        upper = (clav + c7) / 2.0
        lower = (strn + t10) / 2.0
        front = (clav + strn) / 2.0
        back = (t10 + c7) / 2.0

        # Get the direction of the primary axis Z (facing down)
        z_direc = lower - upper
        z_vec = z_direc / np.array([np.linalg.norm(z_direc)])

        # The secondary axis X is from back to front
        x_direc = front - back
        x_vec = x_direc / np.array([np.linalg.norm(x_direc)])

        # make sure all the axes are orthogonal each othe by np.cross-product
        y_direc = np.cross(z_vec, x_vec)
        y_vec = y_direc / np.array([np.linalg.norm(y_direc)])
        x_direc = np.cross(y_vec, z_vec)
        x_vec = x_direc / np.array([np.linalg.norm(x_direc)])
        z_direc = np.cross(x_vec, y_vec)
        z_vec = z_direc / np.array([np.linalg.norm(z_direc)])

        # move the axes about offset along the x axis.
        offset = x_vec * marker_size

        # Add the CLAV back to the vector to get it in the right position before translating it
        origin = clav - offset

        # Attach all the axes to the origin.
        x_axis = x_vec + origin
        y_axis = y_vec + origin
        z_axis = z_vec + origin

        return np.array([origin, x_axis, y_axis, z_axis])

    @staticmethod
    def shoulder_axis_calc(rsho, lsho, thorax_origin, wand, measurements):
        """Shoulder Axis Calculation function

        Calculates the right and left shoulder joint center and axis and returns them.

        Markers used: RSHO, LSHO
        Subject Measurement values used: RightShoulderOffset, LeftShoulderOffset

        Parameters
        ----------
        rsho, lsho : dict
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_origin : ndarray
            A 1x3 ndarray of the thorax origin vector (joint center).
        wand : ndarray
            A 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right shoulder origin, right shoulder x, y, and z
            axis components, left shoulder origin, and left shoulder x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> rsho, lsho = np.array([[428, 270, 1500],
        ...                        [68, 269, 1510]])
        >>> thorax_origin = np.array([256, 364, 1459])
        >>> wand = np.array([[255, 364, 1460],
        ...                  [256, 364, 1460]])
        >>> measurements = {'RightShoulderOffset' : 40, 'LeftShoulderOffset' : 40}
        >>> CGM.shoulder_axis_calc(rsho, lsho, thorax_origin, wand, measurements)
        array([[ 434.48959862,  286.3674038 , 1456.42256085],
               [ 434.86398178,  287.21574011, 1456.79694401],
               [ 434.62767518,  286.71564644, 1455.4953813 ],
               [ 433.57266236,  286.76621776, 1456.43580167],
               [  67.20523086,  268.59838794, 1463.0084364 ],
               [  66.75422334,  269.49090808, 1463.0084364 ],
               [  67.22214085,  268.60693287, 1464.0082569 ],
               [  68.09759081,  269.0493145 , 1462.98949007]])
        """

        # First find the shoulder joint centers

        # Get Subject Measurement Values
        r_shoulderoffset = measurements['RightShoulderOffset']
        l_shoulderoffset = measurements['LeftShoulderOffset']
        mm = 7.0
        r_delta = r_shoulderoffset + mm
        l_delta = l_shoulderoffset + mm

        # REQUIRED MARKERS:
        # RSHO
        # LSHO

        r_wand, l_wand = wand

        r_sho_jc = CGM.find_joint_center(r_wand, thorax_origin, rsho, r_delta)
        l_sho_jc = CGM.find_joint_center(l_wand, thorax_origin, lsho, l_delta)

        # Next calculate the axes
        r_wand_direc = r_wand - thorax_origin
        l_wand_direc = l_wand - thorax_origin
        r_wand_direc = r_wand_direc / np.array([np.linalg.norm(r_wand_direc)])
        l_wand_direc = l_wand_direc / np.array([np.linalg.norm(l_wand_direc)])

        # Right
        # Get the direction of the primary axis Z,X,Y
        z_direc = thorax_origin - r_sho_jc
        z_direc = z_direc / np.array([np.linalg.norm(z_direc)])
        y_direc = r_wand_direc * -1
        x_direc = np.cross(y_direc, z_direc)
        x_direc = x_direc / np.array([np.linalg.norm(x_direc)])
        y_direc = np.cross(z_direc, x_direc)
        y_direc = y_direc / np.array([np.linalg.norm(y_direc)])

        # backwards to account for marker size
        r_x_axis = x_direc + r_sho_jc
        r_y_axis = y_direc + r_sho_jc
        r_z_axis = z_direc + r_sho_jc

        # Left
        # Get the direction of the primary axis Z,X,Y
        z_direc = thorax_origin - l_sho_jc
        z_direc = z_direc / np.array([np.linalg.norm(z_direc)])
        y_direc = l_wand_direc
        x_direc = np.cross(y_direc, z_direc)
        x_direc = x_direc / np.array([np.linalg.norm(x_direc)])
        y_direc = np.cross(z_direc, x_direc)
        y_direc = y_direc / np.array([np.linalg.norm(y_direc)])

        # backwards to account for marker size
        l_x_axis = x_direc + l_sho_jc
        l_y_axis = y_direc + l_sho_jc
        l_z_axis = z_direc + l_sho_jc

        return np.array([r_sho_jc, r_x_axis, r_y_axis, r_z_axis, l_sho_jc, l_x_axis, l_y_axis, l_z_axis])

    @staticmethod
    def elbow_axis_calc(relb, lelb, rwra, rwrb, lwra, lwrb,
                        shoulder_origin, measurements):
        """Elbow Axis Calculation function

        Calculates the right and left elbow joint center and axis and returns them.

        Markers used: RELB, LELB, RWRA, RWRB, LWRA, LWRB
        Subject Measurement values used: RightElbowWidth, LeftElbowWidth

        Parameters
        ----------
        relb, lelb, rwra, rwrb, lwra, lwrb : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        shoulder_origin : ndarray
            A 2x3 ndarray of the right and left shoulder origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right elbow origin, right elbow x, y, and z
            axis components, left elbow origin, and left elbow x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> markers = np.array([[658, 326, 1285],
        ...                     [-156, 335, 1287],
        ...                     [776,495, 1108],
        ...                     [830, 436, 1119],
        ...                     [-249, 525, 1117],
        ...                     [-311, 477, 1125]])
        >>> relb, lelb, rwra, rwrb, lwra, lwrb = markers
        >>> shoulder_origin = np.array([[429, 275, 1453],
        ...                             [64, 274, 1463]])
        >>> measurements = {'RightElbowWidth': 74, 'LeftElbowWidth': 74,
        ...                 'RightWristWidth': 55, 'LeftWristWidth': 55}
        >>> CGM.elbow_axis_calc(relb, lelb, rwra, rwrb, lwra, lwrb,
        ...                     shoulder_origin, measurements)
        array([[ 632.87661106,  304.71333701, 1255.816215  ],
               [ 633.02165895,  303.72391276, 1255.81709122],
               [ 633.56063358,  304.81425332, 1256.5386616 ],
               [ 632.16171645,  304.60914701, 1256.50764117],
               [-128.83646766,  316.61340085, 1257.67295646],
               [-128.99095195,  315.62720417, 1257.61336865],
               [-128.11677323,  316.54239358, 1256.98230612],
               [-128.15958175,  316.4638212 , 1258.39368623]])
        """

        r_elbow_width = measurements['RightElbowWidth']
        l_elbow_width = measurements['LeftElbowWidth']
        r_elbow_width = r_elbow_width * -1
        l_elbow_width = l_elbow_width
        mm = 7.0
        r_delta = (r_elbow_width / 2.0) - mm
        l_delta = (l_elbow_width / 2.0) + mm

        rwri = (rwra + rwrb) / 2.0
        lwri = (lwra + lwrb) / 2.0

        # right axis

        rsjc, lsjc = shoulder_origin

        # make the construction vector for finding Elbow joint center
        r_con_1 = np.subtract(rsjc, relb)
        r_con_1 = r_con_1 / np.linalg.norm(r_con_1)

        r_con_2 = np.subtract(rwri, relb)
        r_con_2 = r_con_2 / np.linalg.norm(r_con_2)

        r_cons_vec = np.cross(r_con_1, r_con_2)
        r_cons_vec = r_cons_vec / np.linalg.norm(r_cons_vec)

        r_cons_vec = r_cons_vec * 500 + relb

        l_con_1 = np.subtract(lsjc, lelb)
        l_con_1 = l_con_1 / np.linalg.norm(l_con_1)

        l_con_2 = np.subtract(lwri, lelb)
        l_con_2 = l_con_2 / np.linalg.norm(l_con_2)

        l_cons_vec = np.cross(l_con_1, l_con_2)
        l_cons_vec = l_cons_vec / np.linalg.norm(l_cons_vec)

        l_cons_vec = l_cons_vec * 500 + lelb

        rejc = CGM.find_joint_center(r_cons_vec, rsjc, relb, r_delta)
        lejc = CGM.find_joint_center(l_cons_vec, lsjc, lelb, l_delta)

        # this is radius axis for humerus
        # The wrist values are calculated here because they are needed during the calculation of the elbow axes.
        # They are calculated again in the wrist's respective function for clarity.

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

        # calculate wrist joint center for humerus
        r_wrist_thickness = measurements['RightWristWidth']
        l_wrist_thickness = measurements['LeftWristWidth']
        r_wrist_thickness = r_wrist_thickness / 2.0 + mm
        l_wrist_thickness = l_wrist_thickness / 2.0 + mm

        rwjc = rwri + r_wrist_thickness * r_wri_y_axis
        lwjc = lwri - l_wrist_thickness * l_wri_y_axis

        # recombine the humerus axis

        # right
        z_axis = np.subtract(rsjc, rejc)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.subtract(rwjc, rejc)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # attach each calculated elbow axis to elbow joint center.
        x_axis += rejc
        y_axis += rejc
        z_axis += rejc

        r_elb_x_axis = x_axis
        r_elb_y_axis = y_axis
        r_elb_z_axis = z_axis

        # left
        z_axis = np.subtract(lsjc, lejc)
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.subtract(lwjc, lejc)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # attach each calculated elbow axis to elbow joint center.
        x_axis += lejc
        y_axis += lejc
        z_axis += lejc

        l_elb_x_axis = x_axis
        l_elb_y_axis = y_axis
        l_elb_z_axis = z_axis

        return np.array([rejc, r_elb_x_axis, r_elb_y_axis, r_elb_z_axis,
                         lejc, l_elb_x_axis, l_elb_y_axis, l_elb_z_axis])

    @staticmethod
    def wrist_axis_calc(rwra, rwrb, lwra, lwrb, elbow_axis, measurements):
        """Wrist Axis Calculation function

        Calculates the right and left wrist joint center and axis and returns them.

        Markers used: RWRA, RWRB, LWRA, LWRB

        Parameters
        ----------
        rwra, rwrb, lwra, lwrb : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        elbow_axis : ndarray
            An 8x3 ndarray that contains the right elbow origin, right elbow x, y, and z
            axis components, left elbow origin, and left elbow x, y, and z axis components.
        measurements : dict

        Returns
        --------
        array
            Returns an 8x3 ndarray that contains the right wrist origin, right wrist x, y, and z
            axis components, left wrist origin, and left wrist x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> markers = np.array([[776,495, 1108],
        ...                     [830, 436, 1119],
        ...                     [-249, 525, 1117],
        ...                     [-311, 477, 1125]])
        >>> rwra, rwrb, lwra, lwrb = markers
        >>> elbow_axis = np.array([[633, 304, 1256],
        ...                        [633, 303, 1256],
        ...                        [634, 305, 1256],
        ...                        [632, 304, 1256],
        ...                        [-129,  316, 1258],
        ...                        [-129, 315, 1258],
        ...                        [-128, 316, 1257],
        ...                        [-128, 316, 1258]])
        >>> measurements = {'RightWristWidth': 55.0, 'LeftWristWidth': 55.0}
        >>> CGM.wrist_axis_calc(rwra, rwrb, lwra, lwrb, elbow_axis, measurements)
        array([[ 792.63409644,  450.54752412, 1084.18751958],
               [ 793.34017987,  449.84144069, 1084.24130037],
               [ 793.34120322,  451.2546309 , 1084.18751958],
               [ 792.59606767,  450.58555289, 1085.18607234],
               [-271.89437608,  485.56813908, 1091.22741984],
               [-272.32731946,  484.77749374, 1090.79447646],
               [-271.1872693 ,  485.56813908, 1090.52031306],
               [-271.3353054 ,  484.95586468, 1091.78649052]])
        """

        mm = 7.0

        rwri = (rwra + rwrb) / 2.0
        lwri = (lwra + lwrb) / 2.0

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

    @staticmethod
    def hand_axis_calc(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_origin, measurements):
        """Hand Axis Calculation function

        Calculates the right and left hand joint center and axis and returns them.

        Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN
        Subject Measurement values used: RightHandThickness, LeftHandThickness

        Parameters
        ----------
        rwra, rwrb, lwra, lwrb, rfin, lfin : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        wrist_origin : ndarray
            A 2x3 array containing the x,y,z position of the right and left wrist joint center.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right hand origin, right hand x, y, and z
            axis components, left hand origin, and left hand x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> markers = np.array([[776,495, 1108],
        ...                     [830, 436, 1119],
        ...                     [-249, 525, 1117],
        ...                     [-311, 477, 1125],
        ...                     [863, 524, 1074],
        ...                     [-326, 558, 1091]])
        >>> rwra, rwrb, lwra, lwrb, rfin, lfin = markers
        >>> wrist_origin = np.array([[793, 451, 1084],
        ...                          [-272, 485, 1091]])
        >>> measurements = {'RightHandThickness': 34.0, 'LeftHandThickness': 34.0}
        >>> CGM.hand_axis_calc(rwra, rwrb, lwra, lwrb, rfin, lfin, wrist_origin, measurements)
        array([[ 859.03526827,  516.82158979, 1051.4444834 ],
               [ 859.18590744,  517.13431139, 1052.38230696],
               [ 858.30712092,  517.49833553, 1051.33577905],
               [ 858.36660598,  516.15509117, 1051.77413522],
               [-324.13023947,  551.49269957, 1067.97481734],
               [-324.21331368,  551.76000767, 1068.9348408 ],
               [-324.92945279,  550.89937566, 1068.07086319],
               [-323.53496076,  550.73341496, 1068.23774345]])
        """

        rwri = (rwra + rwrb) / 2.0
        lwri = (lwra + lwrb) / 2.0

        rwjc, lwjc = wrist_origin

        mm = 7.0
        r_hand_thickness = measurements['RightHandThickness']
        l_hand_thickness = measurements['LeftHandThickness']

        r_delta = (r_hand_thickness / 2.0 + mm)
        l_delta = (l_hand_thickness / 2.0 + mm)

        rhnd = CGM.find_joint_center(rwri, rwjc, rfin, r_delta)
        lhnd = CGM.find_joint_center(lwri, lwjc, lfin, l_delta)

        # Right
        z_axis = rwjc - rhnd
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = rwra - rwri
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        r_x_axis = x_axis
        r_y_axis = y_axis
        r_z_axis = z_axis

        # Left
        z_axis = lwjc - lhnd
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = lwri - lwra
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        l_x_axis = x_axis
        l_y_axis = y_axis
        l_z_axis = z_axis

        # Attach it to the origin.
        r_x_axis += rhnd
        r_y_axis += rhnd
        r_z_axis += rhnd

        l_x_axis += lhnd
        l_y_axis += lhnd
        l_z_axis += lhnd

        return np.array([rhnd, r_x_axis, r_y_axis, r_z_axis, lhnd, l_x_axis, l_y_axis, l_z_axis])

    # Angle calculation functions
    @staticmethod
    def pelvis_angle_calc(global_axis, pelvis_axis):
        """Pelvis Angle Calculation function

        Calculates the global pelvis angle.

        Parameters
        ----------
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        pelvis_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the pelvis axis.

        Returns
        -------
        ndarray
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the pelvis.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> global_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> pelvis_axis = np.array([[251.1, 391, 1032.3],
        ...                         [251, 392.2, 1032.4],
        ...                         [250, 391, 1032],
        ...                         [251.5, 391, 1033]])
        >>> CGM.pelvis_angle_calc(global_axis, pelvis_axis)
        array([29.7448813, -0.       ,  0.       ])
        """

        pelvis_axis_mod = CGM.subtract_origin(pelvis_axis)
        return CGM.get_angle(global_axis, pelvis_axis_mod)

    @staticmethod
    def hip_angle_calc(hip_axis, knee_axis):
        """Hip Angle Calculation function

        Calculates the hip angle.

        Parameters
        ----------
        hip_axis : ndarray
            A 6x3 ndarray containing the right and left hip joint centers, the hip origin,
            and the hip unit vectors.
        knee_axis : ndarray
            An 8x3 ndarray containing the right knee origin, right knee unit vectors,
            left knee origin, and left knee unit vectors.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left hip.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> hip_axis = np.array([[np.nan, np.nan, np.nan],
        ...                      [np.nan, np.nan, np.nan],
        ...                      [245, 331, 936],
        ...                      [245, 332, 936],
        ...                      [244, 331, 936],
        ...                      [245, 331, 937]])
        >>> knee_axis = np.array([[364, 292, 515],
        ...                       [364, 293, 515],
        ...                       [363, 292, 515],
        ...                       [364, 292, 516],
        ...                       [143, 279, 524],
        ...                       [143, 280, 524],
        ...                       [142, 280, 524],
        ...                       [143, 280, 525]])
        >>> CGM.hip_angle_calc(hip_axis, knee_axis)
        array([[ -0.,   0.,   0.],
               [-45.,   0.,   0.]])
        """

        hip_axis_mod = CGM.subtract_origin(hip_axis[2:])
        r_knee_axis_mod = CGM.subtract_origin(knee_axis[:4])
        l_knee_axis_mod = CGM.subtract_origin(knee_axis[4:])

        r_hip_angle = CGM.get_angle(hip_axis_mod, r_knee_axis_mod)
        l_hip_angle = CGM.get_angle(hip_axis_mod, l_knee_axis_mod)

        # GCS fix  # TODO: don't assume it's GCS related
        r_hip_angle = np.array([r_hip_angle[0] * -1, r_hip_angle[1], r_hip_angle[2] * -1 + 90.0])
        l_hip_angle = np.array([l_hip_angle[0] * -1, l_hip_angle[1] * -1, l_hip_angle[2] - 90.0])

        return np.array([r_hip_angle, l_hip_angle])

    @staticmethod
    def knee_angle_calc(knee_axis, ankle_axis):
        """Knee Angle Calculation function

        Calculates the knee angle.

        Parameters
        ----------
        knee_axis : ndarray
            An 8x3 ndarray containing the right knee origin, right knee unit vectors,
            left knee origin, and left knee unit vectors.
        ankle_axis : ndarray
            An 8x3 ndarray containing the right ankle origin, right ankle unit vectors,
            left ankle origin, and left ankle unit vectors.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left ankle.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> knee_axis = np.array([[364.17, 292.17, 515.19],
        ...                       [364.61, 293.06, 515.18],
        ...                       [363.29, 292.60, 515.04],
        ...                       [364.04, 292.24, 516.18],
        ...                       [143.55, 279.90, 524.78],
        ...                       [143.65, 280.88, 524.63],
        ...                       [142.56, 280.01, 524.86],
        ...                       [143.64, 280.04, 525.76]])
        >>> ankle_axis = np.array([[393.76, 247.67, 87.73],
        ...                        [394.48, 248.37, 87.71],
        ...                        [393.07, 248.39, 87.61],
        ...                        [393.69, 247.78, 88.73],
        ...                        [ 98.74, 219.46, 80.63],
        ...                        [ 98.47, 220.42, 80.52],
        ...                        [ 97.79, 219.21, 80.76],
        ...                        [ 98.84, 219.61, 81.61]])
        >>> CGM.knee_angle_calc(knee_axis, ankle_axis)
        array([[  3.24601515,   2.35552002, -19.422079  ],
               [  0.57849185,  -0.23491335, -21.5194992 ]])
        """
        r_knee_axis_mod = CGM.subtract_origin(knee_axis[:4])
        l_knee_axis_mod = CGM.subtract_origin(knee_axis[4:])
        r_ankle_axis_mod = CGM.subtract_origin(ankle_axis[:4])
        l_ankle_axis_mod = CGM.subtract_origin(ankle_axis[4:])

        r_knee_angle = CGM.get_angle(r_knee_axis_mod, r_ankle_axis_mod)
        l_knee_angle = CGM.get_angle(l_knee_axis_mod, l_ankle_axis_mod)

        # GCS fix
        r_knee_angle = np.array([r_knee_angle[0], r_knee_angle[1], r_knee_angle[2] * -1 + 90.0])
        l_knee_angle = np.array([l_knee_angle[0], l_knee_angle[1] * -1, l_knee_angle[2] - 90.0])

        return np.array([r_knee_angle, l_knee_angle])

    @staticmethod
    def ankle_angle_calc(ankle_axis, foot_axis):
        """Ankle Angle Calculation function

        Calculates the ankle angle.

        Parameters
        ----------
        ankle_axis : ndarray
            An 8x3 ndarray containing the right ankle origin, right ankle unit vectors,
            left ankle origin, and left ankle unit vectors.
        foot_axis : ndarray
            An 8x3 ndarray containing the right foot origin, right foot unit vectors,
            left foot origin, and left foot unit vectors.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left ankle.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394, 248, 87],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [ 98, 219, 80],
        ...                        [ 98, 220, 80],
        ...                        [ 97, 219, 80],
        ...                        [ 98, 219, 81]])
        >>> foot_axis = np.array([[442, 381, 42],
        ...                       [442, 381, 43],
        ...                       [441, 381, 42],
        ...                       [442, 380, 42],
        ...                       [39 , 382, 41],
        ...                       [39 , 382, 42],
        ...                       [38 , 382, 41],
        ...                       [39 , 381, 41]])
        >>> CGM.ankle_angle_calc(ankle_axis, foot_axis) #doctest: +NORMALIZE_WHITESPACE
         array([[ 0., 90., 90.],
                [ 0.,  0., -0.]])
        """
        r_ankle_axis_mod = CGM.subtract_origin(ankle_axis[:4])
        l_ankle_axis_mod = CGM.subtract_origin(ankle_axis[4:])
        r_foot_axis_mod = CGM.subtract_origin(foot_axis[:4])
        l_foot_axis_mod = CGM.subtract_origin(foot_axis[4:])

        r_ankle_angle = CGM.get_angle(r_ankle_axis_mod, r_foot_axis_mod)
        l_ankle_angle = CGM.get_angle(l_ankle_axis_mod, l_foot_axis_mod)

        # GCS fix
        r_ankle_angle = np.array([r_ankle_angle[0] * -1 - 90.0, r_ankle_angle[2] * -1 + 90.0, r_ankle_angle[1]])
        l_ankle_angle = np.array([l_ankle_angle[0] * -1 - 90.0, l_ankle_angle[2] - 90.0, l_ankle_angle[1] * -1])

        return np.array([r_ankle_angle, l_ankle_angle])

    @staticmethod
    def foot_angle_calc(global_axis, foot_axis):
        """Foot Angle Calculation function

        Calculates the global foot angle.

        Parameters
        ----------
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        foot_axis : ndarray
            An 8x3 ndarray containing the right foot origin, right foot unit vectors,
            left foot origin, and left foot unit vectors.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left foot.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> global_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> foot_axis = np.array([[442.81, 381.62, 42.66],
        ...                       [442.84, 381.65, 43.65],
        ...                       [441.87, 381.95, 42.67],
        ...                       [442.48, 380.68, 42.69],
        ...                       [39.43 , 382.44, 41.78],
        ...                       [39.56 , 382.50, 42.77],
        ...                       [38.49 , 382.14, 41.93],
        ...                       [39.74 , 381.49, 41.81]])
        >>> CGM.foot_angle_calc(global_axis, foot_axis)
        array([[-84.80557109,  -5.19442891,  70.05155641],
               [ 84.47245985, 168.69006753, -71.80512766]])
        """

        r_foot_axis_mod = CGM.subtract_origin(foot_axis[:4])
        l_foot_axis_mod = CGM.subtract_origin(foot_axis[4:])

        r_global_foot_angle = CGM.get_angle(global_axis, r_foot_axis_mod)
        l_global_foot_angle = CGM.get_angle(global_axis, l_foot_axis_mod)

        # GCS fix
        r_foot_angle = np.array([r_global_foot_angle[0], r_global_foot_angle[2] - 90.0, r_global_foot_angle[1]])
        l_foot_angle = np.array(
            [l_global_foot_angle[0], l_global_foot_angle[2] * -1 + 90.0, l_global_foot_angle[1] * -1])

        return np.array([r_foot_angle, l_foot_angle])

    @staticmethod
    def head_angle_calc(global_axis, head_axis):
        """Head Angle Calculation function

        Calculates the global head angle.

        Parameters
        ----------
        global_axis : ndarray
            A 3x3 ndarray representing the global coordinate system.
        head_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the head axis.

        Returns
        -------
        ndarray
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the head.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> global_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> head_axis = np.array([[99.5, 82.2, 1483.8],
        ...                    [100.2, 83.4, 1484.7],
        ...                    [98.1, 83.6, 1483.3],
        ...                    [99.8, 82.4, 1484.2]])
        >>> CGM.head_angle_calc(global_axis, head_axis)
        array([ -36.86989765,    6.19039946, -139.39870535])
        """
        head_axis_mod = CGM.subtract_origin(head_axis)
        # Old repo subtracts [0, 0, 0] from global_axis here
        global_head_angle = CGM.get_head_angle(global_axis, head_axis_mod)

        # GCS fix
        head_x = global_head_angle[0] * -1
        if head_x < -180.0:
            head_x += 360.0
        head_y = global_head_angle[1] * -1
        head_z = global_head_angle[2]
        if head_z < -180.0:
            head_z -= 360.0

        return np.array([head_x, head_y, head_z])

    @staticmethod
    def thorax_angle_calc(thorax_axis):
        """Thorax Angle Calculation function

        Calculates the global thorax angle.

        Parameters
        ----------
        thorax_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the thorax axis.

        Returns
        -------
        ndarray
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the thorax.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> thorax_axis = np.array([[-252.2, -364.5, -1459.1],
        ...                         [256.4, -365.5, -1459.4],
        ...                         [-257.1, 364.2, -1459.3],
        ...                         [256.2, 354.7, 1458.2]])
        >>> CGM.thorax_angle_calc(thorax_axis)
        array([-170.114302  ,           nan,  179.92137266])
        """  # TODO - Resolve this
        thorax_axis_mod = CGM.subtract_origin(thorax_axis)
        global_axis = CGM.rotation_matrix(x=0, y=0, z=180.0)
        global_thorax_angle = CGM.get_angle(global_axis, thorax_axis_mod)
        if global_thorax_angle[0] > 0:
            global_thorax_angle[0] -= 180.0
        elif global_thorax_angle[0] < 0:
            global_thorax_angle[0] += 180.0

        # GCS fix
        return np.array([global_thorax_angle[0], global_thorax_angle[1], global_thorax_angle[2] + 90.0])

    @staticmethod
    def neck_angle_calc(head_axis, thorax_axis):
        """Neck Angle Calculation function

        Calculates the neck angle.

        Parameters
        ----------
        head_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the head axis.
        thorax_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the thorax axis.

        Returns
        -------
        ndarray
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the neck.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> head_axis = np.array([[99, 82, 1483],
        ...                       [100, 83, 1484],
        ...                       [98, 83, 1483],
        ...                       [99, 82, 1484]])
        >>> thorax_axis = np.array([[256, 364, 1459],
        ...                         [256, 365, 1459],
        ...                         [257, 364, 1459],
        ...                         [256, 354, 1458]])
        >>> CGM.neck_angle_calc(head_axis, thorax_axis)
        array([-84.80557109,  81.95053302,  45.        ])
        """
        head_axis_mod = CGM.subtract_origin(head_axis)
        thorax_axis_mod = CGM.subtract_origin(thorax_axis)
        neck_angle = CGM.get_head_angle(head_axis_mod, thorax_axis_mod)

        # GCS fix
        return np.array([180.0 - neck_angle[0], neck_angle[1], neck_angle[2] * -1])

    @staticmethod
    def spine_angle_calc(pelvis_axis, thorax_axis):
        """Spine Angle Calculation function

        Calculates the spine angle.

        Parameters
        ----------
        pelvis_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the pelvis axis.
        thorax_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the thorax axis.

        Returns
        -------
        ndarray
            A 1x3 ndarray containing the flexion, abduction, and rotation angles of the spine.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> pelvis_axis = np.array([[ 251.2, 391, -1032],
        ...                         [ 251.3, -392, 1032],
        ...                         [ 250.1, -391.5, 1032],
        ...                         [ 251.2, 391, -1033]])
        >>> thorax_axis = np.array([[256, 364.5, 1459],
        ...                         [256, 365, -1459],
        ...                         [-257, 364.8, 1459],
        ...                         [-256, 354.6, 1458]])
        >>> CGM.spine_angle_calc(pelvis_axis, thorax_axis)
        array([nan, -0., nan])
        """  # TODO - resolve this 
        pelvis_axis_mod = CGM.subtract_origin(pelvis_axis)
        thorax_axis_mod = CGM.subtract_origin(thorax_axis)
        spine_angle = CGM.get_spine_angle(pelvis_axis_mod, thorax_axis_mod)

        # GCS fix
        return np.array([spine_angle[0], spine_angle[2] * -1, spine_angle[1]])

    @staticmethod
    def shoulder_angle_calc(thorax_axis, elbow_axis):
        """Shoulder Angle Calculation function

        Calculates the shoulder angle.

        Parameters
        ----------
        thorax_axis : ndarray
            A 4x3 ndarray containing the origin and three unit vectors of the thorax axis.
        elbow_axis : ndarray
            An 8x3 ndarray containing the origin and three unit vectors of the right elbow axis,
            followed by the origin and three unit vectors of the left elbow axis.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left shoulders.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> thorax_axis = np.array([[256, 364.5, 1459],
        ...                         [256, 365, -1459],
        ...                         [-257, 364.8, 1459],
        ...                         [-256, 354.6, 1458]])
        >>> elbow_axis = np.array([[433.5, 304.9, 1256],
        ...                        [433.3, 303.8, 1256],
        ...                        [434.8, 305.4, 1256],
        ...                        [432, 304, 1256],
        ...                        [-529.4,  316, 1258],
        ...                        [-549, 315, 1258],
        ...                        [-548, 316, 1257],
        ...                        [-518, 316, 1258]])
        >>> CGM.shoulder_angle_calc(thorax_axis, elbow_axis)
        array([[ 26.74368395, 135.28459775,  24.44395478],
               [ -0.        , -45.0558983 , -89.99018235]])
        """
        thorax_axis_mod = CGM.subtract_origin(thorax_axis)
        r_elbow_axis_mod = CGM.subtract_origin(elbow_axis[:4])
        l_elbow_axis_mod = CGM.subtract_origin(elbow_axis[4:])
        r_shoulder_angle = CGM.get_shoulder_angle(thorax_axis_mod, r_elbow_axis_mod)
        l_shoulder_angle = CGM.get_shoulder_angle(thorax_axis_mod, l_elbow_axis_mod)

        if r_shoulder_angle[2] < 0:
            r_shoulder_angle[2] += 180.0
        elif r_shoulder_angle[2] > 0:
            r_shoulder_angle[2] -= 180.0

        if r_shoulder_angle[1] > 0:
            r_shoulder_angle[1] -= 180.0
        elif r_shoulder_angle[1] < 0:
            r_shoulder_angle[1] = -180.0 - r_shoulder_angle[1]

        if l_shoulder_angle[1] < 0:
            l_shoulder_angle[1] += 180.0
        elif l_shoulder_angle[1] > 0:
            l_shoulder_angle[1] -= 180.0

        # GCS fix
        r_shoulder_angle = np.array([r_shoulder_angle[0] * -1, r_shoulder_angle[1] * -1, r_shoulder_angle[2]])
        l_shoulder_angle = np.array([l_shoulder_angle[0] * -1, l_shoulder_angle[1], 180.0 - l_shoulder_angle[2]])

        if l_shoulder_angle[2] > 180:
            l_shoulder_angle[2] -= 360

        return np.array([r_shoulder_angle, l_shoulder_angle])

    @staticmethod
    def elbow_angle_calc(elbow_axis, wrist_axis):
        """Elbow Angle Calculation function

        Calculates the elbow angle.

        Parameters
        ----------
        elbow_axis : ndarray
            An 8x3 ndarray containing the origin and three unit vectors of the right elbow axis,
            followed by the origin and three unit vectors of the left elbow axis.
        wrist_axis : ndarray
            An 8x3 ndarray containing the origin and three unit vectors of the right wrist axis,
            followed by the origin and three unit vectors of the left wrist axis.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left elbows.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> elbow_axis = np.array([[433.5, 304.9, 1256],
        ...                        [433.3, 303.8, 1256],
        ...                        [434.8, 305.4, 1256],
        ...                        [432, 304, 1256],
        ...                        [-529.4,  316, 1258],
        ...                        [-549, 315, 1258],
        ...                        [-548, 316, 1257],
        ...                        [-518, 316, 1258]])
        >>> wrist_axis = np.array([[212.5, 251.9, 103],
        ...                        [214.3, 242.8, 126],
        ...                        [210.8, 244.4, 156],
        ...                        [211.5, 244, 123],
        ...                        [329.4,  316, 1258],
        ...                        [249, 445, 214],
        ...                        [242, 446, 217],
        ...                        [221, 432, 258]])
        >>> CGM.elbow_angle_calc(elbow_axis, wrist_axis)
        array([[ -45.91665426,           nan,   20.34505085],
               [-121.60075378,           nan, -136.39962763]])
        """  # TODO - Resolve this
        r_elbow_axis_mod = CGM.subtract_origin(elbow_axis[:4])
        l_elbow_axis_mod = CGM.subtract_origin(elbow_axis[4:])
        r_wrist_axis_mod = CGM.subtract_origin(wrist_axis[:4])
        l_wrist_axis_mod = CGM.subtract_origin(wrist_axis[4:])

        r_elbow_angle = CGM.get_angle(r_elbow_axis_mod, r_wrist_axis_mod)
        l_elbow_angle = CGM.get_angle(l_elbow_axis_mod, l_wrist_axis_mod)

        r_elbow_angle = np.array([r_elbow_angle[0], r_elbow_angle[1], r_elbow_angle[2] - 90.0])
        l_elbow_angle = np.array([l_elbow_angle[0], l_elbow_angle[1], l_elbow_angle[2] - 90.0])

        return np.array([r_elbow_angle, l_elbow_angle])

    @staticmethod
    def wrist_angle_calc(wrist_axis, hand_axis):
        """Wrist Angle Calculation function

        Calculates the wrist angle.

        Parameters
        ----------
        wrist_axis : ndarray
            An 8x3 ndarray containing the origin and three unit vectors of the right wrist axis,
            followed by the origin and three unit vectors of the left wrist axis.
        hand_axis : ndarray
            An 8x3 ndarray containing the origin and three unit vectors of the right hand axis,
            followed by the origin and three unit vectors of the left hand axis.

        Returns
        -------
        ndarray
            A 2x3 ndarray containing the flexion, abduction, and rotation angles
            of the right and left wrists.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import CGM
        >>> wrist_axis = np.array([[433.5, 304.9, 1256],
        ...                        [433.3, 303.8, 1256],
        ...                        [434.8, 305.4, 1256],
        ...                        [432, 304, 1256],
        ...                        [-529.4,  316, 1258],
        ...                        [-549, 315, 1258],
        ...                        [-548, 316, 1257],
        ...                        [-518, 316, 1258]])
        >>> hand_axis = np.array([[212.5, 251.9, 103],
        ...                        [214.3, 242.8, 126],
        ...                        [210.8, 244.4, 156],
        ...                        [211.5, 244, 123],
        ...                        [329.4,  316, 1258],
        ...                        [249, 445, 214],
        ...                        [242, 446, 217],
        ...                        [221, 432, 258]])
        >>> CGM.wrist_angle_calc(wrist_axis, hand_axis)
        array([[ -45.91665426,           nan,  -20.34505085],
               [-121.60075378,           nan, -136.39962763]])
        """  # TODO - Resolve this
        r_wrist_axis_mod = CGM.subtract_origin(wrist_axis[:4])
        l_wrist_axis_mod = CGM.subtract_origin(wrist_axis[4:])
        r_hand_axis_mod = CGM.subtract_origin(hand_axis[:4])
        l_hand_axis_mod = CGM.subtract_origin(hand_axis[4:])

        r_wrist_angle = CGM.get_angle(r_wrist_axis_mod, r_hand_axis_mod)
        l_wrist_angle = CGM.get_angle(l_wrist_axis_mod, l_hand_axis_mod)

        r_wrist_x = r_wrist_angle[0]
        r_wrist_y = r_wrist_angle[1]
        r_wrist_z = 90.0 - r_wrist_angle[2]
        l_wrist_x = l_wrist_angle[0]
        l_wrist_y = l_wrist_angle[1] * -1
        l_wrist_z = l_wrist_angle[2] - 90.0

        if l_wrist_z < -180.0:
            l_wrist_z += 360.0

        r_wrist_angle = np.array([r_wrist_x, r_wrist_y, r_wrist_z])
        l_wrist_angle = np.array([l_wrist_x, l_wrist_y, l_wrist_z])

        return np.array([r_wrist_angle, l_wrist_angle])

    # Center of Mass / Kinetics calculation Methods:
    @staticmethod
    def get_kinetics(joint_centers, jc_mapping, seg_scale, body_mass):
        """Estimate center of mass values in the global coordinate system.

        Estimates whole body CoM in global coordinate system using PiG scaling
        factors for determining individual segment CoM.

        Parameters
        ----------
        joint_centers : 2darray
            2D numpy array where each index corresponds to a joint center or marker value
            that is used to estimate the center of mass for that frame. Each value
            is a 1x3 array indicating the XYZ coordinate of that marker or joint
            center.
        jc_mapping : dict
            Dictionary where keys are joint center or marker names, and values are
            indices that indicate which index in `joint_centers` correspond to that
            joint center or marker name.
        seg_scale : dict
            Segment scaling factors loaded in from the segments.csv file.
        body_mass : int, float
            Total bodymass (kg) of the subject.

        Returns
        -------
        com_coords : ndarray
            Numpy array containing center of mass coordinates for the frame
            of trial. The coordinate is a 1x3 array of the XYZ position of the center
            of mass.

        Examples
        --------

        This example uses the CGM interface to load inputs used in the center of
        mass calculations for one frame in `get_kinetics`.

        >>> from pycgm.pycgm import CGM
        >>> >>> from __future__ import absolute_import
        >>> from pycgm.io import IO
        >>> import pycgm
        >>> static_trial, dynamic_trial, measurements = pycgm.get_robo_data()
        >>> subject = CGM(static_trial, dynamic_trial, measurements, start=0, end=1)
        >>> subject.run()
        >>> seg_scale = IO.load_scaling_table()
        >>> jc_mapping = subject.jc_idx
        >>> joint_centers = subject.joint_centers[0]
        >>> body_mass = 72.0
        >>> CGM.get_kinetics(joint_centers, jc_mapping, seg_scale, body_mass)
        array([-942.7636386 ,   -3.58139618,  865.32990601])
        """
        # Define names of segments
        sides = ['L', 'R']
        segments = ['Foot', 'Tibia', 'Femur', 'Pelvis', 'Radius', 'Hand', 'Humerus', 'Head', 'Thorax']

        frame = joint_centers
        # Find distal and proximal joint centers
        seg_temp = {}
        for s in sides:
            for seg in segments:
                if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
                    seg_temp[s + seg] = {}
                else:
                    seg_temp[seg] = {}

                if seg == 'Foot':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Foot']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'HEE']]

                if seg == 'Tibia':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Knee']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'Ankle']]

                if seg == 'Femur':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Hip']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'Knee']]

                if seg == 'Pelvis':
                    lhjc = frame[jc_mapping['LHip']]
                    rhjc = frame[jc_mapping['RHip']]
                    pelvis_origin = frame[jc_mapping['pelvis_origin']]
                    pelvis_x = frame[jc_mapping['pelvis_x']]
                    pelvis_y = frame[jc_mapping['pelvis_y']]
                    pelvis_z = frame[jc_mapping['pelvis_z']]
                    pelvis_axis = [pelvis_origin, pelvis_x, pelvis_y, pelvis_z]
                    mid_hip, l5 = CGM.find_l5(lhjc, rhjc, pelvis_axis)
                    seg_temp[seg]['Prox'] = mid_hip
                    seg_temp[seg]['Dist'] = l5

                if seg == 'Thorax':
                    # The thorax length is taken as the distance between an
                    # approximation to the C7 vertebra and the L5 vertebra in the
                    # Thorax reference frame. C7 is estimated from the C7 marker,
                    # and offset by half a marker diameter in the direction of
                    # the X axis. L5 is estimated from the L5 provided from the
                    # pelvis segment, but localised to the thorax.

                    lhjc = frame[jc_mapping['LHip']]
                    rhjc = frame[jc_mapping['RHip']]
                    thorax_origin = frame[jc_mapping['thorax_origin']]
                    thorax_x = frame[jc_mapping['thorax_x']]
                    thorax_y = frame[jc_mapping['thorax_y']]
                    thorax_z = frame[jc_mapping['thorax_z']]
                    thorax_axis = [thorax_origin, thorax_x, thorax_y, thorax_z]
                    _, l5 = CGM.find_l5(lhjc, rhjc, thorax_axis)
                    c7 = frame[jc_mapping['C7']]
                    clav = frame[jc_mapping['CLAV']]
                    strn = frame[jc_mapping['STRN']]
                    t10 = frame[jc_mapping['T10']]

                    upper = np.array([(clav[0] + c7[0]) / 2.0, (clav[1] + c7[1]) / 2.0, (clav[2] + c7[2]) / 2.0])
                    lower = np.array([(strn[0] + t10[0]) / 2.0, (strn[1] + t10[1]) / 2.0, (strn[2] + t10[2]) / 2.0])

                    # Get the direction of the primary axis Z (facing down)
                    z_vec = upper - lower
                    z_dir = z_vec / np.linalg.norm(z_vec)
                    new_start = upper + (z_dir * 300)
                    new_end = lower - (z_dir * 300)

                    _, new_l5, _ = CGM.point_to_line(l5, new_start, new_end)
                    _, new_c7, _ = CGM.point_to_line(c7, new_start, new_end)

                    seg_temp[seg]['Prox'] = np.array(new_c7)
                    seg_temp[seg]['Dist'] = np.array(new_l5)

                if seg == 'Humerus':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Shoulder']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'Humerus']]

                if seg == 'Radius':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Humerus']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'Radius']]

                if seg == 'Hand':
                    seg_temp[s + seg]['Prox'] = frame[jc_mapping[s + 'Radius']]
                    seg_temp[s + seg]['Dist'] = frame[jc_mapping[s + 'Hand']]

                if seg == 'Head':
                    seg_temp[seg]['Prox'] = frame[jc_mapping['Back_Head']]
                    seg_temp[seg]['Dist'] = frame[jc_mapping['Front_Head']]

                # Iterate through scaling values
                for row in list(seg_scale.keys()):
                    scale = seg_scale[row]['com']
                    mass = seg_scale[row]['mass']
                    if seg == row:
                        if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
                            prox = seg_temp[s + seg]['Prox']
                            dist = seg_temp[s + seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment center of mass
                            com = dist + length * scale

                            seg_temp[s + seg]['CoM'] = com

                            # segment mass (kg)
                            mass = body_mass * mass  # row[2] contains mass corrections
                            seg_temp[s + seg]['Mass'] = mass

                            # segment torque
                            torque = com * mass
                            seg_temp[s + seg]['Torque'] = torque

                            # vector
                            vector = np.array(com) - np.array([0, 0, 0])
                            val = vector * mass
                            seg_temp[s + seg]['val'] = val

                        # no side allocation
                        else:
                            prox = seg_temp[seg]['Prox']
                            dist = seg_temp[seg]['Dist']

                            # segment length
                            length = prox - dist

                            # segment CoM
                            com = dist + length * scale

                            seg_temp[seg]['CoM'] = com

                            # segment mass (kg)
                            mass = body_mass * mass  # row[2] is mass correction factor
                            seg_temp[seg]['Mass'] = mass

                            # segment torque
                            torque = com * mass
                            seg_temp[seg]['Torque'] = torque

                            # vector
                            vector = np.array(com) - np.array([0, 0, 0])
                            val = vector * mass
                            seg_temp[seg]['val'] = val

        vals = []

        if pyver == 2:
            for_iter = seg_temp.iteritems()
        elif pyver == 3:
            for_iter = seg_temp.items()

        for attr, value in for_iter:
            vals.append(value['val'])

        com_coords = sum(vals) / body_mass
        return com_coords

    # Properties for accessing subsets of results
    @staticmethod
    def repack(array):
        # Used to combine multiple columns of data by frame
        # i.e., if X, Y, and Z are three different columns, repack makes them one column of XYZ
        return np.flip(np.rot90(array), 0)

    @property
    def pelvis_axes(self):
        pel = np.array([self.axis_results[0:, self.axis_idx["PELO"]],
                        self.axis_results[0:, self.axis_idx["PELX"]],
                        self.axis_results[0:, self.axis_idx["PELY"]],
                        self.axis_results[0:, self.axis_idx["PELZ"]]])
        return pel

    @property
    def pelvis_joint_centers(self):
        return self.axis_results[0:, self.axis_idx["PELO"]]

    @property
    def pelvis_angles(self):
        return self.repack(self.angle_results[0:, self.angle_idx["Pelvis"]])

    @property
    def pelvis_flexions(self):
        return self.angle_results[0:, self.angle_idx["Pelvis"]][:, 0]

    @property
    def pelvis_abductions(self):
        return self.angle_results[0:, self.angle_idx["Pelvis"]][:, 1]

    @property
    def pelvis_rotations(self):
        return self.angle_results[0:, self.angle_idx["Pelvis"]][:, 2]

    @property
    def hip_axes(self):
        hip = np.array([self.axis_results[0:, self.axis_idx["HIPO"]],
                        self.axis_results[0:, self.axis_idx["HIPX"]],
                        self.axis_results[0:, self.axis_idx["HIPY"]],
                        self.axis_results[0:, self.axis_idx["HIPZ"]]])
        return hip

    @property
    def hip_joint_centers(self):
        return self.axis_results[0:, self.axis_idx["HIPO"]]

    @property
    def hip_angles(self):
        hip = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Hip"]]),
                        self.repack(self.angle_results[0:, self.angle_idx["L Hip"]])])
        return hip

    @property
    def hip_flexions(self):
        hip = np.array([self.angle_results[0:, self.angle_idx["R Hip"]][:, 0],
                        self.angle_results[0:, self.angle_idx["L Hip"]][:, 0]])
        return hip

    @property
    def hip_abductions(self):
        hip = np.array([self.angle_results[0:, self.angle_idx["R Hip"]][:, 1],
                        self.angle_results[0:, self.angle_idx["L Hip"]][:, 1]])
        return hip

    @property
    def hip_rotations(self):
        hip = np.array([self.angle_results[0:, self.angle_idx["R Hip"]][:, 2],
                        self.angle_results[0:, self.angle_idx["L Hip"]][:, 2]])
        return hip

    @property
    def knee_axes(self):
        knee = np.array([[self.axis_results[0:, self.axis_idx["R KNEO"]],
                          self.axis_results[0:, self.axis_idx["R KNEX"]],
                          self.axis_results[0:, self.axis_idx["R KNEY"]],
                          self.axis_results[0:, self.axis_idx["R KNEZ"]]],
                         [self.axis_results[0:, self.axis_idx["L KNEO"]],
                          self.axis_results[0:, self.axis_idx["L KNEX"]],
                          self.axis_results[0:, self.axis_idx["L KNEY"]],
                          self.axis_results[0:, self.axis_idx["L KNEZ"]]]])
        return knee

    @property
    def knee_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R KNEO"]],
                         self.axis_results[0:, self.axis_idx["L KNEO"]]])

    @property
    def knee_angles(self):
        knee = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Knee"]]),
                         self.repack(self.angle_results[0:, self.angle_idx["L Knee"]])])
        return knee

    @property
    def knee_flexions(self):
        knee = np.array([self.angle_results[0:, self.angle_idx["R Knee"]][:, 0],
                         self.angle_results[0:, self.angle_idx["L Knee"]][:, 0]])
        return knee

    @property
    def knee_abductions(self):
        knee = np.array([self.angle_results[0:, self.angle_idx["R Knee"]][:, 1],
                         self.angle_results[0:, self.angle_idx["L Knee"]][:, 1]])
        return knee

    @property
    def knee_rotations(self):
        knee = np.array([self.angle_results[0:, self.angle_idx["R Knee"]][:, 2],
                         self.angle_results[0:, self.angle_idx["L Knee"]][:, 2]])
        return knee

    @property
    def ankle_axes(self):
        ankle = np.array([[self.axis_results[0:, self.axis_idx["R ANKO"]],
                           self.axis_results[0:, self.axis_idx["R ANKX"]],
                           self.axis_results[0:, self.axis_idx["R ANKY"]],
                           self.axis_results[0:, self.axis_idx["R ANKZ"]]],
                          [self.axis_results[0:, self.axis_idx["L ANKO"]],
                           self.axis_results[0:, self.axis_idx["L ANKX"]],
                           self.axis_results[0:, self.axis_idx["L ANKY"]],
                           self.axis_results[0:, self.axis_idx["L ANKZ"]]]])
        return ankle

    @property
    def ankle_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R ANKO"]],
                         self.axis_results[0:, self.axis_idx["L ANKO"]]])

    @property
    def ankle_angles(self):
        ankle = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Ankle"]]),
                          self.repack(self.angle_results[0:, self.angle_idx["L Ankle"]])])
        return ankle

    @property
    def ankle_flexions(self):
        ankle = np.array([self.angle_results[0:, self.angle_idx["R Ankle"]][:, 0],
                          self.angle_results[0:, self.angle_idx["L Ankle"]][:, 0]])
        return ankle

    @property
    def ankle_abductions(self):
        ankle = np.array([self.angle_results[0:, self.angle_idx["R Ankle"]][:, 1],
                          self.angle_results[0:, self.angle_idx["L Ankle"]][:, 1]])
        return ankle

    @property
    def ankle_rotations(self):
        ankle = np.array([self.angle_results[0:, self.angle_idx["R Ankle"]][:, 2],
                          self.angle_results[0:, self.angle_idx["L Ankle"]][:, 2]])
        return ankle

    @property
    def foot_axes(self):
        foot = np.array([[self.axis_results[0:, self.axis_idx["R FOOO"]],
                          self.axis_results[0:, self.axis_idx["R FOOX"]],
                          self.axis_results[0:, self.axis_idx["R FOOY"]],
                          self.axis_results[0:, self.axis_idx["R FOOZ"]]],
                         [self.axis_results[0:, self.axis_idx["L FOOO"]],
                          self.axis_results[0:, self.axis_idx["L FOOX"]],
                          self.axis_results[0:, self.axis_idx["L FOOY"]],
                          self.axis_results[0:, self.axis_idx["L FOOZ"]]]])
        return foot

    @property
    def foot_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R FOOO"]],
                         self.axis_results[0:, self.axis_idx["L FOOO"]]])

    @property
    def foot_angles(self):
        foot = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Foot"]]),
                         self.repack(self.angle_results[0:, self.angle_idx["L Foot"]])])
        return foot

    @property
    def foot_flexions(self):
        foot = np.array([self.angle_results[0:, self.angle_idx["R Foot"]][:, 0],
                         self.angle_results[0:, self.angle_idx["L Foot"]][:, 0]])
        return foot

    @property
    def foot_abductions(self):
        foot = np.array([self.angle_results[0:, self.angle_idx["R Foot"]][:, 1],
                         self.angle_results[0:, self.angle_idx["L Foot"]][:, 1]])
        return foot

    @property
    def foot_rotations(self):
        foot = np.array([self.angle_results[0:, self.angle_idx["R Foot"]][:, 2],
                         self.angle_results[0:, self.angle_idx["L Foot"]][:, 2]])
        return foot

    @property
    def head_axes(self):
        head = np.array([self.axis_results[0:, self.axis_idx["HEAO"]],
                         self.axis_results[0:, self.axis_idx["HEAX"]],
                         self.axis_results[0:, self.axis_idx["HEAY"]],
                         self.axis_results[0:, self.axis_idx["HEAZ"]]])
        return head

    @property
    def head_joint_center(self):
        return self.axis_results[0:, self.axis_idx["HEAO"]]

    @property
    def head_angles(self):
        return self.repack(self.angle_results[0:, self.angle_idx["Head"]])

    @property
    def head_flexions(self):
        return self.angle_results[0:, self.angle_idx["Head"]][:, 0]

    @property
    def head_abductions(self):
        return self.angle_results[0:, self.angle_idx["Head"]][:, 1]

    @property
    def head_rotations(self):
        return self.angle_results[0:, self.angle_idx["Head"]][:, 2]

    @property
    def thorax_axes(self):
        thorax = np.array([self.axis_results[0:, self.axis_idx["THOO"]],
                           self.axis_results[0:, self.axis_idx["THOX"]],
                           self.axis_results[0:, self.axis_idx["THOY"]],
                           self.axis_results[0:, self.axis_idx["THOZ"]]])
        return thorax

    @property
    def thorax_joint_center(self):
        return self.axis_results[0:, self.axis_idx["THOO"]]

    @property
    def thorax_angles(self):
        return self.repack(self.angle_results[0:, self.angle_idx["Thorax"]])

    @property
    def thorax_flexions(self):
        return self.angle_results[0:, self.angle_idx["Thorax"]][:, 0]

    @property
    def thorax_abductions(self):
        return self.angle_results[0:, self.angle_idx["Thorax"]][:, 1]

    @property
    def thorax_rotations(self):
        return self.angle_results[0:, self.angle_idx["Thorax"]][:, 2]

    @property
    def neck_angles(self):
        return self.repack(self.angle_results[0:, self.angle_idx["Neck"]])

    @property
    def neck_flexions(self):
        return self.angle_results[0:, self.angle_idx["Neck"]][:, 0]

    @property
    def neck_abductions(self):
        return self.angle_results[0:, self.angle_idx["Neck"]][:, 1]

    @property
    def neck_rotations(self):
        return self.angle_results[0:, self.angle_idx["Neck"]][:, 2]

    @property
    def spine_angles(self):
        return self.repack(self.angle_results[0:, self.angle_idx["Spine"]])

    @property
    def spine_flexions(self):
        return self.angle_results[0:, self.angle_idx["Spine"]][:, 0]

    @property
    def spine_abductions(self):
        return self.angle_results[0:, self.angle_idx["Spine"]][:, 1]

    @property
    def spine_rotations(self):
        return self.angle_results[0:, self.angle_idx["Spine"]][:, 2]

    @property
    def shoulder_axes(self):
        shoulder = np.array([[self.axis_results[0:, self.axis_idx["R CLAO"]],
                              self.axis_results[0:, self.axis_idx["R CLAX"]],
                              self.axis_results[0:, self.axis_idx["R CLAY"]],
                              self.axis_results[0:, self.axis_idx["R CLAZ"]]],
                             [self.axis_results[0:, self.axis_idx["L CLAO"]],
                              self.axis_results[0:, self.axis_idx["L CLAX"]],
                              self.axis_results[0:, self.axis_idx["L CLAY"]],
                              self.axis_results[0:, self.axis_idx["L CLAZ"]]]])
        return shoulder

    @property
    def shoulder_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R CLAO"]],
                         self.axis_results[0:, self.axis_idx["L CLAO"]]])

    @property
    def shoulder_angles(self):
        shoulder = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Shoulder"]]),
                             self.repack(self.angle_results[0:, self.angle_idx["L Shoulder"]])])
        return shoulder

    @property
    def shoulder_flexions(self):
        shoulder = np.array([self.angle_results[0:, self.angle_idx["R Shoulder"]][:, 0],
                             self.angle_results[0:, self.angle_idx["L Shoulder"]][:, 0]])
        return shoulder

    @property
    def shoulder_abductions(self):
        shoulder = np.array([self.angle_results[0:, self.angle_idx["R Shoulder"]][:, 1],
                             self.angle_results[0:, self.angle_idx["L Shoulder"]][:, 1]])
        return shoulder

    @property
    def shoulder_rotations(self):
        shoulder = np.array([self.angle_results[0:, self.angle_idx["R Shoulder"]][:, 2],
                             self.angle_results[0:, self.angle_idx["L Shoulder"]][:, 2]])
        return shoulder

    @property
    def elbow_axes(self):
        elbow = np.array([[self.axis_results[0:, self.axis_idx["R HUMO"]],
                           self.axis_results[0:, self.axis_idx["R HUMX"]],
                           self.axis_results[0:, self.axis_idx["R HUMY"]],
                           self.axis_results[0:, self.axis_idx["R HUMZ"]]],
                          [self.axis_results[0:, self.axis_idx["L HUMO"]],
                           self.axis_results[0:, self.axis_idx["L HUMX"]],
                           self.axis_results[0:, self.axis_idx["L HUMY"]],
                           self.axis_results[0:, self.axis_idx["L HUMZ"]]]])
        return elbow

    @property
    def elbow_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R HUMO"]],
                         self.axis_results[0:, self.axis_idx["L HUMO"]]])

    @property
    def elbow_angles(self):
        elbow = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Elbow"]]),
                          self.repack(self.angle_results[0:, self.angle_idx["L Elbow"]])])
        return elbow

    @property
    def elbow_flexions(self):
        elbow = np.array([self.angle_results[0:, self.angle_idx["R Elbow"]][:, 0],
                          self.angle_results[0:, self.angle_idx["L Elbow"]][:, 0]])
        return elbow

    @property
    def elbow_abductions(self):
        elbow = np.array([self.angle_results[0:, self.angle_idx["R Elbow"]][:, 1],
                          self.angle_results[0:, self.angle_idx["L Elbow"]][:, 1]])
        return elbow

    @property
    def elbow_rotations(self):
        elbow = np.array([self.angle_results[0:, self.angle_idx["R Elbow"]][:, 2],
                          self.angle_results[0:, self.angle_idx["L Elbow"]][:, 2]])
        return elbow

    @property
    def wrist_axes(self):
        wrist = np.array([[self.axis_results[0:, self.axis_idx["R RADO"]],
                           self.axis_results[0:, self.axis_idx["R RADX"]],
                           self.axis_results[0:, self.axis_idx["R RADY"]],
                           self.axis_results[0:, self.axis_idx["R RADZ"]]],
                          [self.axis_results[0:, self.axis_idx["L RADO"]],
                           self.axis_results[0:, self.axis_idx["L RADX"]],
                           self.axis_results[0:, self.axis_idx["L RADY"]],
                           self.axis_results[0:, self.axis_idx["L RADZ"]]]])
        return wrist

    @property
    def wrist_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R RADO"]],
                         self.axis_results[0:, self.axis_idx["L RADO"]]])

    @property
    def wrist_angles(self):
        wrist = np.array([self.repack(self.angle_results[0:, self.angle_idx["R Wrist"]]),
                          self.repack(self.angle_results[0:, self.angle_idx["L Wrist"]])])
        return wrist

    @property
    def wrist_flexions(self):
        wrist = np.array([self.angle_results[0:, self.angle_idx["R Wrist"]][:, 0],
                          self.angle_results[0:, self.angle_idx["L Wrist"]][:, 0]])
        return wrist

    @property
    def wrist_abductions(self):
        wrist = np.array([self.angle_results[0:, self.angle_idx["R Wrist"]][:, 1],
                          self.angle_results[0:, self.angle_idx["L Wrist"]][:, 1]])
        return wrist

    @property
    def wrist_rotations(self):
        wrist = np.array([self.angle_results[0:, self.angle_idx["R Wrist"]][:, 2],
                          self.angle_results[0:, self.angle_idx["L Wrist"]][:, 2]])
        return wrist

    @property
    def hand_axes(self):
        hand = np.array([[self.axis_results[0:, self.axis_idx["R HANO"]],
                          self.axis_results[0:, self.axis_idx["R HANX"]],
                          self.axis_results[0:, self.axis_idx["R HANY"]],
                          self.axis_results[0:, self.axis_idx["R HANZ"]]],
                         [self.axis_results[0:, self.axis_idx["L HANO"]],
                          self.axis_results[0:, self.axis_idx["L HANX"]],
                          self.axis_results[0:, self.axis_idx["L HANY"]],
                          self.axis_results[0:, self.axis_idx["L HANZ"]]]])
        return hand

    @property
    def hand_joint_centers(self):
        return np.array([self.axis_results[0:, self.axis_idx["R HANO"]],
                         self.axis_results[0:, self.axis_idx["L HANO"]]])


class SubjectManager:
    def __init__(self, *subjects):
        pass


class StaticCGM:
    """
    A class to used to calculate the offsets in a static trial and calibrate subject measurements.

    Attributes
    ----------
    motion_data : ndarray
        `motion_data` is a 3d numpy array. Each index `i` corresponds to frame `i`
        of trial. `motion_data[i]` contains a list of coordinate values for each marker.
        Each coordinate value is a 1x3 list: [X, Y, Z].
    mapping : dict
        `mapping` is a dictionary that indicates which marker corresponds to which index
        in `motion_data[i]`.
    measurements : dict
        A dictionary of the measurements of the subject, obtained from file input.
    """

    def __init__(self, path_static, path_measurements):
        """Initialization of StaticCGM object function

        Instantiates various class attributes based on parameters and default values.

        Parameters
        ----------
        path_static : str
            File path of the static trial in csv or c3d form
        path_measurements : str
            File path of the subject measurements in csv or vsk form
        """
        self.motion_data, self.mapping = IO.load_marker_data(path_static)
        self.measurements = IO.load_sm(path_measurements)

    @staticmethod
    def iad_calculation(rasi, lasi):
        """Calculates the Inter ASIS Distance.
        Given the markers RASI and LASI, the Inter ASIS Distance is defined as:
        .. math::
            InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 + (RASI_z-LASI_z)^2}
        where :math:`RASI_x` is the x-coordinate of the RASI marker in frame.
        Markers used: RASI, LASI
        Parameters
        ----------
        rasi, lasi : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        Returns
        -------
        iad : float
            The Inter ASIS distance as a float.

        Examples
        --------
        >>> from numpy import around, array
        >>> from pycgm.pycgm import StaticCGM
        >>> lasi = array([ 183,  422, 1033])
        >>> rasi = array([ 395,  428, 1036])
        >>> around(StaticCGM.iad_calculation(rasi, lasi), 2)
        212.11
        """
        x_diff = rasi[0] - lasi[0]
        y_diff = rasi[1] - lasi[1]
        z_diff = rasi[2] - lasi[2]
        iad = np.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)
        return iad

    @staticmethod
    def static_calculation_head(head_axis):
        """Calculates the offset angle of the head.
        Uses the x,y,z axes of the head and the head origin to calculate
        the head offset angle. Uses the global axis.
        Parameters
        ----------
        head_axis : ndarray
            Array containing 4 1x3 arrays. The first gives the XYZ coordinates of the head
            origin. The remaining 3 are 1x3 arrays that give the XYZ coordinates of the
            X, Y, and Z head axis respectively.

        Returns
        -------
        offset : float
            The head offset angle.

        Examples
        --------
        >>> from numpy import around, array
        >>> from pycgm.pycgm import StaticCGM
        >>> head_axis = array([[99, 82, 1483],
        ...                    [100, 83, 1484],
        ...                    [98, 83, 1483],
        ...                    [99, 82, 1484]])
        >>> around(StaticCGM.static_calculation_head(head_axis), 8)
        0.78539816
        """
        head_axis = CGM.subtract_origin(head_axis)
        global_axis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        # Global axis is the proximal axis
        # Head axis is the distal axis
        axis_p = global_axis
        axis_d = head_axis

        axis_p_inverse = np.linalg.inv(axis_p)
        rotation_matrix = np.matmul(axis_d, axis_p_inverse)
        offset = np.arctan(rotation_matrix[0][2] / rotation_matrix[2][2])

        return offset

    @staticmethod
    def static_calculation(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements):
        """The Static Angle Calculation function

        Takes in anatomical uncorrect axis and anatomical correct axis.
        Correct axis depends on foot flat options.

        Calculates the offset angle between that two axis.

        It is rotated from uncorrect axis in YXZ order.

        Parameters
        ----------
        rtoe, ltoe, rhee, lhee : dict
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        flat_foot : boolean
            A boolean indicating if the feet are flat or not.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        angle : ndarray
            Returns the offset angle represented by a 2x3 ndarray.
            The array contains the right flexion, abduction, and rotation angles,
            followed by the left flexion, abduction, and rotation angles.

        Modifies
        --------
        The correct axis changes following to the foot flat option.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rtoe, ltoe, rhee, lhee = np.array([[427, 437,  41],
        ...                                    [175, 379,  42],
        ...                                    [406, 227,  48],
        ...                                    [223, 173,  47]])
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394, 248, 87],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [98, 219, 80],
        ...                        [98, 220, 80],
        ...                        [97, 219, 80],
        ...                        [98, 219, 81]])
        >>> flat_foot = True
        >>> measurements = { 'RightSoleDelta': 0.45,'LeftSoleDelta': 0.45 }
        >>> parameters = [rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements]
        >>> np.around(StaticCGM.static_calculation(*parameters),8)
        array([[-0.23229863,  0.0823201 ,  0.87549777],
               [-0.65268629,  0.28720026, -0.30918223]])
        >>> flat_foot = False
        >>> parameters = [rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, measurements]
        >>> np.around(StaticCGM.static_calculation(*parameters),8)
        array([[-0.20601627,  0.06161168, -0.59732228],
               [-0.65539028,  0.25761562,  0.11115736]])
        """
        # Get the each axis from the function.
        uncorrect_foot = StaticCGM.foot_axis_calc(rtoe, ltoe, ankle_axis)

        rf1_r_axis = CGM.subtract_origin(uncorrect_foot[:4])
        rf1_l_axis = CGM.subtract_origin(uncorrect_foot[4:])

        # Check if it is flat foot or not.
        if not flat_foot:
            rf_axis2 = StaticCGM.non_flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis)
            # make the array to same format for calculating angle.
            rf2_r_axis = CGM.subtract_origin(rf_axis2[:4])
            rf2_l_axis = CGM.subtract_origin(rf_axis2[4:])

            r_ankle_flex_angle = StaticCGM.ankle_angle_calc(rf1_r_axis, rf2_r_axis)
            l_ankle_flex_angle = StaticCGM.ankle_angle_calc(rf1_l_axis, rf2_l_axis)

        else:
            rf_axis3 = StaticCGM.flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis, measurements)
            # make the array to same format for calculating angle.
            rf3_r_axis = CGM.subtract_origin(rf_axis3[:4])
            rf3_l_axis = CGM.subtract_origin(rf_axis3[4:])

            r_ankle_flex_angle = StaticCGM.ankle_angle_calc(rf1_r_axis, rf3_r_axis)
            l_ankle_flex_angle = StaticCGM.ankle_angle_calc(rf1_l_axis, rf3_l_axis)

        angle = np.array([r_ankle_flex_angle, l_ankle_flex_angle])

        return angle

    @staticmethod
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        """Pelvis Axis Calculation function

        Calculates the pelvis joint center and axis and returns them.

        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: CGM.cross product of x_axis and y_axis.

        Parameters
        ----------
        rasi, lasi : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        sacr, rpsi, lpsi : ndarray, optional
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.

        References
        ----------
        .. [12] Kadaba MP, Ramakrishnan HK, Wootten ME.
           Measurement of lower extremity kinematics during level walking.
           Journal of orthopaedic research: official publication of the Orthopaedic Research Society.
           1990;8(3):383–92.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rasi, lasi, rpsi, lpsi = np.array([[ 395,  428, 1036],
        ...                                    [ 183,  422, 1033],
        ...                                    [ 341,  246, 1055],
        ...                                    [ 255,  241, 1057]])
        >>> CGM.pelvis_axis_calc(rasi, lasi, rpsi=rpsi, lpsi=lpsi)
        array([[ 289.        ,  425.        , 1034.5       ],
               [ 288.97356163,  425.99275625, 1034.38279911],
               [ 288.00050025,  424.97171227, 1034.48585614],
               [ 288.98264324,  425.11676832, 1035.4930075 ]])
        >>> rasi, lasi, sacr = np.array([[ 395,  428, 1036],
        ...                              [ 183,  422, 1033],
        ...                              [ 294,  242, 1049]])
        >>> StaticCGM.pelvis_axis_calc(rasi, lasi, sacr=sacr)
        array([[ 289.        ,  425.        , 1034.5       ],
               [ 288.97291419,  425.99651005, 1034.42104387],
               [ 288.00050025,  424.97171227, 1034.48585614],
               [ 288.98367201,  425.07853354, 1035.49677775]])
        """

        # REQUIRED MARKERS:
        # RASI
        # LASI
        # RPSI
        # LPSI

        # If sacrum marker is present, use it
        if sacr is not None:
            sacrum = sacr
        # Otherwise mean of posterior markers is used as the sacrum
        else:
            sacrum = (rpsi + lpsi) / 2.0

        # REQUIRED LANDMARKS:
        # origin
        # sacrum

        # Origin is the midpoint between RASI and LASI
        origin = (rasi + lasi) / 2.0

        # Calculate each axis; beta{n} are arbitrary names
        beta1 = origin - sacrum
        beta2 = lasi - rasi

        # Y_axis is normalized beta2
        y_axis = beta2 / np.linalg.norm(beta2)

        # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
        # and then normalized.
        beta3_cal = np.dot(beta1, y_axis) * y_axis
        beta3 = beta1 - beta3_cal
        x_axis = beta3 / np.array(np.linalg.norm(beta3))

        # Z-axis is np.cross product of x_axis and y_axis
        z_axis = np.cross(x_axis, y_axis)

        # Add the origin back to the vector
        y_axis += origin
        z_axis += origin
        x_axis += origin

        return np.array([origin, x_axis, y_axis, z_axis])

    @staticmethod
    def hip_axis_calc(pelvis_axis, measurements):
        """Hip Axis Calculation function

        Calculates the right and left hip joint center and axis and returns them.

        Other landmarks used: origin, sacrum
        Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure,
        InterAsisDistance, L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

        Parameters
        ----------
        pelvis_axis : ndarray
            A 4x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the hip origin and the
            hip x, y, and z axis components.

        References
        ----------
        .. [20]  Davis RB, Ounpuu S, Tyburski D, Gage JR.
           A gait analysis data collection and reduction technique. Human Movement Science.
           1991;10(5):575–587.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> pelvis_axis = np.array([[ 251, 391, 1032],
        ...                         [ 251, 392, 1032],
        ...                         [ 250, 391, 1032],
        ...                         [ 251, 391, 1033]])
        >>> measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.51,
        ...                 'L_AsisToTrocanterMeasure': 72.51, 'InterAsisDistance': 215}
        >>> StaticCGM.hip_axis_calc(pelvis_axis, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[314.00929545, 340.53152887, 929.98436036],
               [187.99070455, 340.53152887, 929.98436036],
               [251.        , 340.53152887, 929.98436036],
               [251.        , 341.53152887, 929.98436036],
               [250.        , 340.53152887, 929.98436036],
               [251.        , 340.53152887, 930.98436036]])
        """

        # Requires
        # pelvis axis

        pel_o, pel_x, pel_y, pel_z = pelvis_axis

        # Model's eigen value

        # LegLength
        # MeanLegLength
        # mm (marker radius)
        # interAsisMeasure

        # Set the variables needed to calculate the joint angle
        # Half of marker size
        mm = 7.0

        mean_leg_length = measurements['MeanLegLength']
        r_asis_to_trocanter_measure = measurements['R_AsisToTrocanterMeasure']
        l_asis_to_trocanter_measure = measurements['L_AsisToTrocanterMeasure']
        inter_asis_measure = measurements['InterAsisDistance']
        c = (mean_leg_length * 0.115) - 15.3
        theta = 0.500000178813934
        beta = 0.314000427722931
        aa = inter_asis_measure / 2.0
        s = -1

        # Hip Joint Center Calculation (ref. Davis_1991)

        # Calculate the distance to translate along the pelvis axis
        # Left
        l_xh = (-l_asis_to_trocanter_measure - mm) * cos(beta) + c * cos(theta) * sin(beta)
        l_yh = s * (c * sin(theta) - aa)
        l_zh = (-l_asis_to_trocanter_measure - mm) * sin(beta) - c * cos(theta) * cos(beta)

        # Right
        r_xh = (-r_asis_to_trocanter_measure - mm) * cos(beta) + c * cos(theta) * sin(beta)
        r_yh = (c * sin(theta) - aa)
        r_zh = (-r_asis_to_trocanter_measure - mm) * sin(beta) - c * cos(theta) * cos(beta)

        # Get the unit pelvis axis
        pelvis_xaxis = pel_x - pel_o
        pelvis_yaxis = pel_y - pel_o
        pelvis_zaxis = pel_z - pel_o

        # Multiply the distance to the unit pelvis axis
        l_hip_jc_x = pelvis_xaxis * l_xh
        l_hip_jc_y = pelvis_yaxis * l_yh
        l_hip_jc_z = pelvis_zaxis * l_zh
        l_hip_jc = np.array([l_hip_jc_x[0] + l_hip_jc_y[0] + l_hip_jc_z[0],
                             l_hip_jc_x[1] + l_hip_jc_y[1] + l_hip_jc_z[1],
                             l_hip_jc_x[2] + l_hip_jc_y[2] + l_hip_jc_z[2]])

        r_hip_jc_x = pelvis_xaxis * r_xh
        r_hip_jc_y = pelvis_yaxis * r_yh
        r_hip_jc_z = pelvis_zaxis * r_zh
        r_hip_jc = np.array([r_hip_jc_x[0] + r_hip_jc_y[0] + r_hip_jc_z[0],
                             r_hip_jc_x[1] + r_hip_jc_y[1] + r_hip_jc_z[1],
                             r_hip_jc_x[2] + r_hip_jc_y[2] + r_hip_jc_z[2]])

        l_hip_jc += pel_o
        r_hip_jc += pel_o

        # Get shared hip axis, it is inbetween the two hip joint centers
        hip_axis_center = [(r_hip_jc[0] + l_hip_jc[0]) / 2.0, (r_hip_jc[1] + l_hip_jc[1]) / 2.0,
                           (r_hip_jc[2] + l_hip_jc[2]) / 2.0]

        # Convert pelvis_axis to x, y, z axis to use more easily
        pelvis_x_axis = np.subtract(pelvis_axis[1], pelvis_axis[0])
        pelvis_y_axis = np.subtract(pelvis_axis[2], pelvis_axis[0])
        pelvis_z_axis = np.subtract(pelvis_axis[3], pelvis_axis[0])

        # Translate pelvis axis to shared hip center
        # Add the origin back to the vector
        y_axis = [pelvis_y_axis[0] + hip_axis_center[0], pelvis_y_axis[1] + hip_axis_center[1],
                  pelvis_y_axis[2] + hip_axis_center[2]]
        z_axis = [pelvis_z_axis[0] + hip_axis_center[0], pelvis_z_axis[1] + hip_axis_center[1],
                  pelvis_z_axis[2] + hip_axis_center[2]]
        x_axis = [pelvis_x_axis[0] + hip_axis_center[0], pelvis_x_axis[1] + hip_axis_center[1],
                  pelvis_x_axis[2] + hip_axis_center[2]]

        return np.array([r_hip_jc, l_hip_jc, hip_axis_center, x_axis, y_axis, z_axis])

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements):
        """Knee Axis Calculation function

        Calculates the right and left knee joint center and axis and returns them.

        Markers used: RTHI, LTHI, RKNE, LKNE
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        hip_origin : ndarray
            A 2x3 ndarray of the right and left hip origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right knee origin, right knee x, y, and z
            axis components, left knee origin, and left knee x, y, and z axis components.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rthi, lthi, rkne, lkne = np.array([[426, 262, 673],
        ...                                    [51 , 320, 723],
        ...                                    [416, 266, 524],
        ...                                    [84 , 286, 529]])
        >>> hip_origin = np.array([[309, 322, 937],
        ...                        [182, 339, 935]])
        >>> measurements = {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0 }
        >>> StaticCGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[363.28036639, 292.19664005, 515.36134954],
               [363.72612681, 293.091773  , 515.35546315],
               [362.39432212, 292.63691972, 515.21616214],
               [363.15299602, 292.26657445, 516.3507362 ],
               [142.89606353, 278.88369813, 524.43251176],
               [143.00280898, 279.86598662, 524.27851584],
               [141.90621372, 279.00329985, 524.50927627],
               [142.98988659, 279.0279367 , 525.41759676]])
        """
        # Get Global Values
        mm = 7.0
        r_knee_width = measurements['RightKneeWidth']
        l_knee_width = measurements['LeftKneeWidth']
        r_delta = (r_knee_width / 2.0) + mm
        l_delta = (l_knee_width / 2.0) + mm

        # REQUIRED MARKERS:
        # RTHI
        # LTHI
        # RKNE
        # LKNE
        # hip_JC

        r_hip_jc, l_hip_jc = hip_origin

        # Determine the position of kneeJointCenter using CGM.find_joint_center function
        r_knee_jc = CGM.find_joint_center(rthi, r_hip_jc, rkne, r_delta)
        l_knee_jc = CGM.find_joint_center(lthi, l_hip_jc, lkne, l_delta)

        # Right axis calculation
        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = r_hip_jc - r_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector np.cross vector.
        # the axis is then normalized.
        axis_x = np.cross(axis_z, rkne - r_hip_jc)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation
        # Z axis is Thigh bone calculated by the hipJC and kneeJC
        # the axis is then normalized
        axis_z = l_hip_jc - l_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector np.cross vector.
        # the axis is then normalized.
        axis_x = np.cross(lkne - l_hip_jc, axis_z)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then normalize it.
        r_knee_x_axis, r_knee_y_axis, r_knee_z_axis = r_axis
        r_knee_x_axis = r_knee_x_axis / np.array([np.linalg.norm(r_knee_x_axis)])
        r_knee_y_axis = r_knee_y_axis / np.array([np.linalg.norm(r_knee_y_axis)])
        r_knee_z_axis = r_knee_z_axis / np.array([np.linalg.norm(r_knee_z_axis)])
        l_knee_x_axis, l_knee_y_axis, l_knee_z_axis = l_axis
        l_knee_x_axis = l_knee_x_axis / np.array([np.linalg.norm(l_knee_x_axis)])
        l_knee_y_axis = l_knee_y_axis / np.array([np.linalg.norm(l_knee_y_axis)])
        l_knee_z_axis = l_knee_z_axis / np.array([np.linalg.norm(l_knee_z_axis)])

        # Put both axis in array
        # Add the origin back to the vector
        ry_axis = r_knee_y_axis + r_knee_jc
        rz_axis = r_knee_z_axis + r_knee_jc
        rx_axis = r_knee_x_axis + r_knee_jc

        # Add the origin back to the vector
        ly_axis = l_knee_y_axis + l_knee_jc
        lz_axis = l_knee_z_axis + l_knee_jc
        lx_axis = l_knee_x_axis + l_knee_jc

        return np.array([r_knee_jc, rx_axis, ry_axis, rz_axis, l_knee_jc, lx_axis, ly_axis, lz_axis])

    @staticmethod
    def ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements):
        """Ankle Axis Calculation function

        Calculates the right and left ankle joint center and axis and returns them.

        Markers used: RTIB, LTIB, RANK, LANK
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

        Parameters
        ----------
        rtib, ltib, rank, lank : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        knee_origin : ndarray
            A 2x3 ndarray of the right and left knee origin vectors (joint centers).
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rtib, ltib, rank, lank = np.array([[433, 211, 273 ],
        ...                                    [50 , 235, 364],
        ...                                    [422, 217, 92 ],
        ...                                    [58 , 208, 86 ]])
        >>> knee_origin = np.array([[364, 292, 515],
        ...                         [143, 279, 524]])
        >>> measurements = {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0,
        ...                 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0}
        >>> StaticCGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)
        array([[393.36370389, 247.2931075 ,  86.87260464],
               [394.09205433, 247.97797298,  86.85104297],
               [392.68188731, 248.01437196,  86.7505238 ],
               [393.2956466 , 247.39672623,  87.86489057],
               [ 98.11999534, 219.11196211,  80.44029928],
               [ 97.84148846, 220.06709428,  80.33952006],
               [ 97.16475735, 218.84739159,  80.57267311],
               [ 98.21976663, 219.24509728,  81.42636253]])
        """

        # Get Global Values
        r_ankle_width = measurements['RightAnkleWidth']
        l_ankle_width = measurements['LeftAnkleWidth']
        r_torsion = measurements['RightTibialTorsion']
        l_torsion = measurements['LeftTibialTorsion']
        mm = 7.0
        r_delta = (r_ankle_width / 2.0) + mm
        l_delta = (l_ankle_width / 2.0) + mm

        # REQUIRED MARKERS:
        # tib_R
        # tib_L
        # ank_R
        # ank_L
        # knee_JC

        knee_jc_r, knee_jc_l = knee_origin

        # This is Torsioned Tibia and this describes the ankle angles
        # Tibial frontal plane is being defined by ANK, TIB, and KJC

        # Determine the position of ankleJointCenter using CGM.find_joint_center function
        r_ankle_jc = CGM.find_joint_center(rtib, knee_jc_r, rank, r_delta)
        l_ankle_jc = CGM.find_joint_center(ltib, knee_jc_l, lank, l_delta)

        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        # Right axis calculation

        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = knee_jc_r - r_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK, TIB, and KJC markers.
        # and calculated by each point's vector np.cross vector.
        # tib_ank_r vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_r = rtib - rank
        axis_x = np.cross(axis_z, tib_ank_r)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation

        # Z axis is shank bone calculated by the ankleJC and kneeJC
        axis_z = knee_jc_l - l_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK, TIB, and KJC markers.
        # and calculated by each point's vector np.cross vector.
        # tib_ank_l vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_l = ltib - lank
        axis_x = np.cross(tib_ank_l, axis_z)

        # Y axis is determined by np.cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then normalize it.
        r_ankle_x_axis = r_axis[0]
        r_ankle_x_axis_div = np.linalg.norm(r_ankle_x_axis)
        r_ankle_x_axis = np.array([r_ankle_x_axis[0] / r_ankle_x_axis_div, r_ankle_x_axis[1] / r_ankle_x_axis_div,
                                   r_ankle_x_axis[2] / r_ankle_x_axis_div])

        r_ankle_y_axis = r_axis[1]
        r_ankle_y_axis_div = np.linalg.norm(r_ankle_y_axis)
        r_ankle_y_axis = np.array([r_ankle_y_axis[0] / r_ankle_y_axis_div, r_ankle_y_axis[1] / r_ankle_y_axis_div,
                                   r_ankle_y_axis[2] / r_ankle_y_axis_div])

        r_ankle_z_axis = r_axis[2]
        r_ankle_z_axis_div = np.linalg.norm(r_ankle_z_axis)
        r_ankle_z_axis = np.array([r_ankle_z_axis[0] / r_ankle_z_axis_div, r_ankle_z_axis[1] / r_ankle_z_axis_div,
                                   r_ankle_z_axis[2] / r_ankle_z_axis_div])

        l_ankle_x_axis = l_axis[0]
        l_ankle_x_axis_div = np.linalg.norm(l_ankle_x_axis)
        l_ankle_x_axis = np.array([l_ankle_x_axis[0] / l_ankle_x_axis_div, l_ankle_x_axis[1] / l_ankle_x_axis_div,
                                   l_ankle_x_axis[2] / l_ankle_x_axis_div])

        l_ankle_y_axis = l_axis[1]
        l_ankle_y_axis_div = np.linalg.norm(l_ankle_y_axis)
        l_ankle_y_axis = np.array([l_ankle_y_axis[0] / l_ankle_y_axis_div, l_ankle_y_axis[1] / l_ankle_y_axis_div,
                                   l_ankle_y_axis[2] / l_ankle_y_axis_div])

        l_ankle_z_axis = l_axis[2]
        l_ankle_z_axis_div = np.linalg.norm(l_ankle_z_axis)
        l_ankle_z_axis = np.array([l_ankle_z_axis[0] / l_ankle_z_axis_div, l_ankle_z_axis[1] / l_ankle_z_axis_div,
                                   l_ankle_z_axis[2] / l_ankle_z_axis_div])

        # Put both axis in array
        r_axis = np.array([r_ankle_x_axis, r_ankle_y_axis, r_ankle_z_axis])
        l_axis = np.array([l_ankle_x_axis, l_ankle_y_axis, l_ankle_z_axis])

        # Rotate the axes about the tibia torsion.
        r_torsion = np.radians(r_torsion)
        l_torsion = np.radians(l_torsion)

        r_axis = np.array([[cos(r_torsion) * r_axis[0][0] - sin(r_torsion) * r_axis[1][0],
                            cos(r_torsion) * r_axis[0][1] - sin(r_torsion) * r_axis[1][1],
                            cos(r_torsion) * r_axis[0][2] - sin(r_torsion) * r_axis[1][2]],
                           [sin(r_torsion) * r_axis[0][0] + cos(r_torsion) * r_axis[1][0],
                            sin(r_torsion) * r_axis[0][1] + cos(r_torsion) * r_axis[1][1],
                            sin(r_torsion) * r_axis[0][2] + cos(r_torsion) * r_axis[1][2]],
                           [r_axis[2][0], r_axis[2][1], r_axis[2][2]]])

        l_axis = np.array([[cos(l_torsion) * l_axis[0][0] - sin(l_torsion) * l_axis[1][0],
                            cos(l_torsion) * l_axis[0][1] - sin(l_torsion) * l_axis[1][1],
                            cos(l_torsion) * l_axis[0][2] - sin(l_torsion) * l_axis[1][2]],
                           [sin(l_torsion) * l_axis[0][0] + cos(l_torsion) * l_axis[1][0],
                            sin(l_torsion) * l_axis[0][1] + cos(l_torsion) * l_axis[1][1],
                            sin(l_torsion) * l_axis[0][2] + cos(l_torsion) * l_axis[1][2]],
                           [l_axis[2][0], l_axis[2][1], l_axis[2][2]]])

        # Add the origin back to the vector
        rx_axis = r_axis[0] + r_ankle_jc
        ry_axis = r_axis[1] + r_ankle_jc
        rz_axis = r_axis[2] + r_ankle_jc

        lx_axis = l_axis[0] + l_ankle_jc
        ly_axis = l_axis[1] + l_ankle_jc
        lz_axis = l_axis[2] + l_ankle_jc

        return np.array([r_ankle_jc, rx_axis, ry_axis, rz_axis, l_ankle_jc, lx_axis, ly_axis, lz_axis])

    # Note: this function is equivalent to uncorrect_footaxis() in pycgmStatic.py
    # the only difference is the order in which operations were calculated.
    @staticmethod
    def foot_axis_calc(rtoe, ltoe, ankle_axis):
        """Foot Axis Calculation function

        Calculates the right and left foot joint axis by rotating uncorrect foot joint axes about offset angle.

        In case of foot joint center, we've already make 2 kinds of axis for static offset angle.
        and then, Call this static offset angle as an input of this function for dynamic trial.

        Special Cases:

        (anatomical uncorrect foot axis)
        If foot flat is true, then make the reference markers instead of HEE marker
        which height is as same as TOE marker's height.
        otherwise, foot flat is false, use the HEE marker for making Z axis.

        Markers used: RTOE, LTOE
        Other landmarks used: ANKLE_FLEXION_AXIS

        Parameters
        ----------
        rtoe, ltoe : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Modifies
        --------
        Axis changes following to the static info.

        you can set the static_info by the button. and this will calculate the offset angles
        the first setting, the foot axis show foot uncorrected anatomical
        reference axis(Z_axis point to the AJC from TOE)

        if press the static_info button so if static_info is not None,
        and then the static offsets angles are applied to the reference axis.
        the reference axis is Z axis point to HEE from TOE

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rtoe, ltoe = np.array([[442, 381, 42],
        ...                        [39 , 382, 41]])
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394 , 248, 87  ],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [98 , 219, 80 ],
        ...                        [98 , 220, 80],
        ...                        [97 , 219, 80],
        ...                        [98 , 219, 81]])
        >>> StaticCGM.foot_axis_calc(rtoe, ltoe, ankle_axis)  # doctest: +NORMALIZE_WHITESPACE
        array([[442.        , 381.        ,  42.        ],
               [442.676405  , 381.        ,  42.73652989],
               [441.34030115, 381.44468887,  42.60584588],
               [441.67247336, 380.10431489,  42.30078977],
               [ 39.        , 382.        ,  41.        ],
               [ 39.        , 382.2326959 ,  41.97254954],
               [ 38.05673939, 381.67706168,  41.07726745],
               [ 39.33205333, 381.08263232,  41.21949288]])
        """
        ankle_jc_r = ankle_axis[0]
        ankle_jc_l = ankle_axis[4]
        ankle_flexion_r = ankle_axis[2]
        ankle_flexion_l = ankle_axis[6]

        # Foot axis's origin is marker position of TOE

        # z axis is from Toe to AJC and normalized.
        r_axis_z = [ankle_jc_r[0] - rtoe[0], ankle_jc_r[1] - rtoe[1], ankle_jc_r[2] - rtoe[2]]
        r_axis_z_div = np.linalg.norm(r_axis_z)
        r_axis_z = [r_axis_z[0] / r_axis_z_div, r_axis_z[1] / r_axis_z_div, r_axis_z[2] / r_axis_z_div]

        # Bring y flexion axis from ankle axis.
        y_flex_r = [ankle_flexion_r[0] - ankle_jc_r[0], ankle_flexion_r[1] - ankle_jc_r[1],
                    ankle_flexion_r[2] - ankle_jc_r[2]]
        y_flex_r_div = np.linalg.norm(y_flex_r)
        y_flex_r = [y_flex_r[0] / y_flex_r_div, y_flex_r[1] / y_flex_r_div, y_flex_r[2] / y_flex_r_div]

        # Calculate x axis by np.cross-product of ankle flexion axis and z axis.
        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x_div = np.linalg.norm(r_axis_x)
        r_axis_x = [r_axis_x[0] / r_axis_x_div, r_axis_x[1] / r_axis_x_div, r_axis_x[2] / r_axis_x_div]

        # Calculate y axis by np.cross-product of z axis and x axis.
        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y_div = np.linalg.norm(r_axis_y)
        r_axis_y = [r_axis_y[0] / r_axis_y_div, r_axis_y[1] / r_axis_y_div, r_axis_y[2] / r_axis_y_div]

        # Attach each axes to origin.
        r_axis_x = [r_axis_x[0] + rtoe[0], r_axis_x[1] + rtoe[1], r_axis_x[2] + rtoe[2]]
        r_axis_y = [r_axis_y[0] + rtoe[0], r_axis_y[1] + rtoe[1], r_axis_y[2] + rtoe[2]]
        r_axis_z = [r_axis_z[0] + rtoe[0], r_axis_z[1] + rtoe[1], r_axis_z[2] + rtoe[2]]

        # Left

        # z axis is from Toe to AJC and normalized.
        l_axis_z = [ankle_jc_l[0] - ltoe[0], ankle_jc_l[1] - ltoe[1], ankle_jc_l[2] - ltoe[2]]
        l_axis_z_div = np.linalg.norm(l_axis_z)
        l_axis_z = [l_axis_z[0] / l_axis_z_div, l_axis_z[1] / l_axis_z_div, l_axis_z[2] / l_axis_z_div]

        # Bring y flexion axis from ankle axis.
        y_flex_l = [ankle_flexion_l[0] - ankle_jc_l[0], ankle_flexion_l[1] - ankle_jc_l[1],
                    ankle_flexion_l[2] - ankle_jc_l[2]]
        y_flex_l_div = np.linalg.norm(y_flex_l)
        y_flex_l = [y_flex_l[0] / y_flex_l_div, y_flex_l[1] / y_flex_l_div, y_flex_l[2] / y_flex_l_div]

        # Calculate x axis by np.cross-product of ankle flexion axis and z axis.
        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x_div = np.linalg.norm(l_axis_x)
        l_axis_x = [l_axis_x[0] / l_axis_x_div, l_axis_x[1] / l_axis_x_div, l_axis_x[2] / l_axis_x_div]

        # Calculate y axis by np.cross-product of z axis and x axis.
        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y_div = np.linalg.norm(l_axis_y)
        l_axis_y = [l_axis_y[0] / l_axis_y_div, l_axis_y[1] / l_axis_y_div, l_axis_y[2] / l_axis_y_div]

        # Attach each axis to origin.
        l_axis_x = [l_axis_x[0] + ltoe[0], l_axis_x[1] + ltoe[1], l_axis_x[2] + ltoe[2]]
        l_axis_y = [l_axis_y[0] + ltoe[0], l_axis_y[1] + ltoe[1], l_axis_y[2] + ltoe[2]]
        l_axis_z = [l_axis_z[0] + ltoe[0], l_axis_z[1] + ltoe[1], l_axis_z[2] + ltoe[2]]

        return np.array([rtoe, r_axis_x, r_axis_y, r_axis_z, ltoe, l_axis_x, l_axis_y, l_axis_z])

    @staticmethod
    def non_flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis):
        """Non-Flat Foot Axis Calculation function

        Calculate the anatomical correct foot axis for non-foot flat.

        Markers used: RTOE, LTOE, RHEE, LHEE

        Parameters
        ----------
        rtoe, ltoe, rhee, lhee : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rhee, lhee, rtoe, ltoe = np.array([[374, 181, 49],
        ...                                    [105, 180, 47],
        ...                                    [442, 381, 42],
        ...                                    [39, 382, 41]])
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394 , 248, 87  ],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [98 , 219, 80 ],
        ...                        [98 , 220, 80],
        ...                        [97 , 219, 80],
        ...                        [98 , 219, 81]])
        >>> [np.around(arr,8) for arr in StaticCGM.non_flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis)] #doctest: +NORMALIZE_WHITESPACE
        [array([442., 381.,  42.]), array([442.10240005, 381.        ,  42.9947433 ]),
        array([441.05872081, 381.3234263 ,  42.09689639]),
        array([441.67827386, 380.05374664,  42.03311887]),
        array([ 39., 382.,  41.]),
        array([ 39.        , 382.02968988,  41.99955916]),
        array([ 38.04941082, 381.68968524,  41.00921727]),
        array([ 39.31045162, 381.04982988,  41.02822287])]
        """
        # REQUIRED MARKERS:
        # RTOE
        # LTOE
        # ankle_JC

        ankle_jc_r = ankle_axis[0]
        ankle_jc_l = ankle_axis[4]
        ankle_flexion_r = ankle_axis[2]
        ankle_flexion_l = ankle_axis[6]

        # Toe axis's origin is marker position of TOE
        ankle_jc_r = [ankle_jc_r[0], ankle_jc_r[1], ankle_jc_r[2]]
        ankle_jc_l = [ankle_jc_l[0], ankle_jc_l[1], ankle_jc_l[2]]

        # in case of non foot flat we just use the HEE marker
        r_axis_z = [rhee[0] - rtoe[0], rhee[1] - rtoe[1], rhee[2] - rtoe[2]]
        r_axis_z = r_axis_z / np.array([np.linalg.norm(r_axis_z)])

        y_flex_r = [ankle_flexion_r[0] - ankle_jc_r[0], ankle_flexion_r[1] - ankle_jc_r[1],
                    ankle_flexion_r[2] - ankle_jc_r[2]]
        y_flex_r = y_flex_r / np.array([np.linalg.norm(y_flex_r)])

        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x = r_axis_x / np.array([np.linalg.norm(r_axis_x)])

        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y = r_axis_y / np.array([np.linalg.norm(r_axis_y)])

        r_axis_x = [r_axis_x[0] + rtoe[0], r_axis_x[1] + rtoe[1], r_axis_x[2] + rtoe[2]]
        r_axis_y = [r_axis_y[0] + rtoe[0], r_axis_y[1] + rtoe[1], r_axis_y[2] + rtoe[2]]
        r_axis_z = [r_axis_z[0] + rtoe[0], r_axis_z[1] + rtoe[1], r_axis_z[2] + rtoe[2]]

        # Left

        ankle_jc_r = [ankle_jc_r[0], ankle_jc_r[1], ankle_jc_r[2]]
        ankle_jc_l = [ankle_jc_l[0], ankle_jc_l[1], ankle_jc_l[2]]

        l_axis_z = [lhee[0] - ltoe[0], lhee[1] - ltoe[1], lhee[2] - ltoe[2]]
        l_axis_z = l_axis_z / np.array([np.linalg.norm(l_axis_z)])

        y_flex_l = [ankle_flexion_l[0] - ankle_jc_l[0], ankle_flexion_l[1] - ankle_jc_l[1],
                    ankle_flexion_l[2] - ankle_jc_l[2]]
        y_flex_l = y_flex_l / np.array([np.linalg.norm(y_flex_l)])

        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x = l_axis_x / np.array([np.linalg.norm(l_axis_x)])

        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y = l_axis_y / np.array([np.linalg.norm(l_axis_y)])

        l_axis_x = [l_axis_x[0] + ltoe[0], l_axis_x[1] + ltoe[1], l_axis_x[2] + ltoe[2]]
        l_axis_y = [l_axis_y[0] + ltoe[0], l_axis_y[1] + ltoe[1], l_axis_y[2] + ltoe[2]]
        l_axis_z = [l_axis_z[0] + ltoe[0], l_axis_z[1] + ltoe[1], l_axis_z[2] + ltoe[2]]

        return np.array([rtoe, r_axis_x, r_axis_y, r_axis_z, ltoe, l_axis_x, l_axis_y, l_axis_z])

    @staticmethod
    def flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis, measurements):
        """Flat Foot Axis Calculation function

        Calculate the anatomical correct foot axis for non-foot flat.

        Markers used: RTOE, LTOE, RHEE, LHEE

        Parameters
        ----------
        rtoe, ltoe, rhee, lhee : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : ndarray
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rtoe, ltoe, rhee, lhee = np.array([[442, 381, 42],
        ...                                    [39, 382, 41],
        ...                                    [374, 181, 49],
        ...                                    [105, 180, 47]])
        >>> ankle_axis = np.array([[393, 247, 87],
        ...                        [394 , 248, 87  ],
        ...                        [393, 248, 87],
        ...                        [393, 247, 88],
        ...                        [98 , 219, 80],
        ...                        [98 , 220, 80],
        ...                        [97 , 219, 80],
        ...                        [98 , 219, 81]])
        >>> measurements = { 'RightSoleDelta': 0.45, 'LeftSoleDelta': 0.35}
        >>> [np.around(arr,8) for arr in StaticCGM.flat_foot_axis_calc(rtoe, ltoe, rhee, lhee, ankle_axis, measurements)] #doctest: +NORMALIZE_WHITESPACE
        [array([442., 381.,  42.]),
        array([441.22996328, 381.26181249,  42.58180552]),
        array([441.44916239, 381.18728479,  41.18667206]),
        array([441.67809727, 380.05322726,  42.        ]),
        array([ 39., 382.,  41.]),
        array([ 38.67155725, 381.89268702,  41.93840785]),
        array([ 38.10799758, 381.70855366,  40.65447039]),
        array([ 39.31057534, 381.04945123,  41.        ])]
        """
        r_sole_delta = measurements['RightSoleDelta']
        l_sole_delta = measurements['LeftSoleDelta']

        # REQUIRED MARKERS:
        # RTOE
        # LTOE
        # ankle_JC

        ankle_jc_r = ankle_axis[0]
        ankle_jc_l = ankle_axis[4]
        ankle_flexion_r = ankle_axis[2]
        ankle_flexion_l = ankle_axis[6]

        # Toe axis's origin is marker position of TOE

        ankle_jc_r = [ankle_jc_r[0], ankle_jc_r[1], ankle_jc_r[2] + r_sole_delta]
        ankle_jc_l = [ankle_jc_l[0], ankle_jc_l[1], ankle_jc_l[2] + l_sole_delta]

        # this is the way to calculate the z axis
        r_axis_z = [ankle_jc_r[0] - rtoe[0], ankle_jc_r[1] - rtoe[1], ankle_jc_r[2] - rtoe[2]]
        r_axis_z = r_axis_z / np.array([np.linalg.norm(r_axis_z)])

        # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
        hee2_toe = [rhee[0] - rtoe[0], rhee[1] - rtoe[1], rtoe[2] - rtoe[2]]
        hee2_toe = hee2_toe / np.array([np.linalg.norm(hee2_toe)])
        a = np.cross(hee2_toe, r_axis_z)
        a = a / np.array([np.linalg.norm(a)])
        b = np.cross(a, hee2_toe)
        b = b / np.array([np.linalg.norm(b)])
        c = np.cross(b, a)
        r_axis_z = c / np.array([np.linalg.norm(c)])

        # Bring flexion axis from ankle axis.
        y_flex_r = [ankle_flexion_r[0] - ankle_jc_r[0], ankle_flexion_r[1] - ankle_jc_r[1],
                    ankle_flexion_r[2] - ankle_jc_r[2]]
        y_flex_r = y_flex_r / np.array([np.linalg.norm(y_flex_r)])

        # Calculate each x,y,z axis of foot using np.cross-product and make sure x,y,z axis is orthogonal each other.
        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x = r_axis_x / np.array([np.linalg.norm(r_axis_x)])

        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y = r_axis_y / np.array([np.linalg.norm(r_axis_y)])

        r_axis_z = np.cross(r_axis_x, r_axis_y)
        r_axis_z = r_axis_z / np.array([np.linalg.norm(r_axis_z)])

        # Attach each axis to origin.
        r_axis_x = [r_axis_x[0] + rtoe[0], r_axis_x[1] + rtoe[1], r_axis_x[2] + rtoe[2]]
        r_axis_y = [r_axis_y[0] + rtoe[0], r_axis_y[1] + rtoe[1], r_axis_y[2] + rtoe[2]]
        r_axis_z = [r_axis_z[0] + rtoe[0], r_axis_z[1] + rtoe[1], r_axis_z[2] + rtoe[2]]

        # Left

        # this is the way to calculate the z axis of foot flat.
        l_axis_z = [ankle_jc_l[0] - ltoe[0], ankle_jc_l[1] - ltoe[1], ankle_jc_l[2] - ltoe[2]]
        l_axis_z = l_axis_z / np.array([np.linalg.norm(l_axis_z)])

        # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
        hee2_toe = [lhee[0] - ltoe[0], lhee[1] - ltoe[1], ltoe[2] - ltoe[2]]
        hee2_toe = hee2_toe / np.array([np.linalg.norm(hee2_toe)])
        a = np.cross(hee2_toe, l_axis_z)
        a = a / np.array([np.linalg.norm(a)])
        b = np.cross(a, hee2_toe)
        b = b / np.array([np.linalg.norm(b)])
        c = np.cross(b, a)
        l_axis_z = c / np.array([np.linalg.norm(c)])

        # Bring flexion axis from ankle axis.
        y_flex_l = [ankle_flexion_l[0] - ankle_jc_l[0], ankle_flexion_l[1] - ankle_jc_l[1],
                    ankle_flexion_l[2] - ankle_jc_l[2]]
        y_flex_l = y_flex_l / np.array([np.linalg.norm(y_flex_l)])

        # Calculate each x,y,z axis of foot using np.cross-product and make sure x,y,z axis is orthogonal each other.
        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x = l_axis_x / np.array([np.linalg.norm(l_axis_x)])

        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y = l_axis_y / np.array([np.linalg.norm(l_axis_y)])

        l_axis_z = np.cross(l_axis_x, l_axis_y)
        l_axis_z = l_axis_z / np.array([np.linalg.norm(l_axis_z)])

        # Attach each axis to origin.
        l_axis_x = [l_axis_x[0] + ltoe[0], l_axis_x[1] + ltoe[1], l_axis_x[2] + ltoe[2]]
        l_axis_y = [l_axis_y[0] + ltoe[0], l_axis_y[1] + ltoe[1], l_axis_y[2] + ltoe[2]]
        l_axis_z = [l_axis_z[0] + ltoe[0], l_axis_z[1] + ltoe[1], l_axis_z[2] + ltoe[2]]

        return np.array([rtoe, r_axis_x, r_axis_y, r_axis_z, ltoe, l_axis_x, l_axis_y, l_axis_z])

    @staticmethod
    def head_axis_calc(rfhd, lfhd, rbhd, lbhd):
        """Head Axis Calculation function

        Calculates the head joint center and axis and returns them.

        Markers used: LFHD, RFHD, LBHD, RBHD

        Parameters
        ----------
        lfhd, rfhd, lbhd, rbhd : ndarray
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the head origin and the
            head x, y, and z axis components.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> rfhd, lfhd, rbhd, lbhd = np.array([[325, 402, 1722],
        ...                                   [184, 409, 1721],
        ...                                   [304, 242, 1694],
        ...                                   [197, 251, 1696]])
        >>> [np.around(arr,8) for arr in StaticCGM.head_axis_calc(rfhd, lfhd, rbhd, lbhd)] #doctest: +NORMALIZE_WHITESPACE
        [array([ 254.5,  405.5, 1721.5]), 
        array([ 254.5248073 ,  406.48609036, 1721.66434839]),
        array([ 253.50032966,  405.52555761, 1721.49754796]),
        array([ 254.49338171,  405.33576661, 1722.48639931])]
        """
        # get the midpoints of the head to define the sides
        front = [(lfhd[0] + rfhd[0]) / 2.0, (lfhd[1] + rfhd[1]) / 2.0, (lfhd[2] + rfhd[2]) / 2.0]
        back = [(lbhd[0] + rbhd[0]) / 2.0, (lbhd[1] + rbhd[1]) / 2.0, (lbhd[2] + rbhd[2]) / 2.0]
        left = [(lfhd[0] + lbhd[0]) / 2.0, (lfhd[1] + lbhd[1]) / 2.0, (lfhd[2] + lbhd[2]) / 2.0]
        right = [(rfhd[0] + rbhd[0]) / 2.0, (rfhd[1] + rbhd[1]) / 2.0, (rfhd[2] + rbhd[2]) / 2.0]
        origin = front

        # Get the vectors from the sides with primary x axis facing front
        # First get the x direction
        x_vec = [front[0] - back[0], front[1] - back[1], front[2] - back[2]]
        x_vec_div = np.linalg.norm(x_vec)
        x_vec = [x_vec[0] / x_vec_div, x_vec[1] / x_vec_div, x_vec[2] / x_vec_div]

        # get the direction of the y axis
        y_vec = [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
        y_vec_div = np.linalg.norm(y_vec)
        y_vec = [y_vec[0] / y_vec_div, y_vec[1] / y_vec_div, y_vec[2] / y_vec_div]

        # get z axis by np.cross-product of x axis and y axis.
        z_vec = np.cross(x_vec, y_vec)
        z_vec_div = np.linalg.norm(z_vec)
        z_vec = [z_vec[0] / z_vec_div, z_vec[1] / z_vec_div, z_vec[2] / z_vec_div]

        # make sure all x,y,z axis is orthogonal each other by np.cross-product
        y_vec = np.cross(z_vec, x_vec)
        y_vec_div = np.linalg.norm(y_vec)
        y_vec = [y_vec[0] / y_vec_div, y_vec[1] / y_vec_div, y_vec[2] / y_vec_div]
        x_vec = np.cross(y_vec, z_vec)
        x_vec_div = np.linalg.norm(x_vec)
        x_vec = [x_vec[0] / x_vec_div, x_vec[1] / x_vec_div, x_vec[2] / x_vec_div]

        # Add the origin back to the vector to get it in the right position
        x_axis = [x_vec[0] + origin[0], x_vec[1] + origin[1], x_vec[2] + origin[2]]
        y_axis = [y_vec[0] + origin[0], y_vec[1] + origin[1], y_vec[2] + origin[2]]
        z_axis = [z_vec[0] + origin[0], z_vec[1] + origin[1], z_vec[2] + origin[2]]

        # Return the three axes and origin
        return np.array([origin, x_axis, y_axis, z_axis])

    @staticmethod
    def ankle_angle_calc(axis_p, axis_d):
        """Static angle calculation function.

        This function takes in two axis and returns three angles.
        and It use inverse Euler rotation matrix in YXZ order.
        the output shows the angle in degrees.

        As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2
        but in case of calculate static offset angle it is in boundary under pi/2, it doesn't matter.

        Parameters
        ----------
        axis_p : ndarray
            A 3x3 ndarray containing the unit vectors of axisP, the proximal axis.
        axis_d : ndarray
            A 3x3 ndarray containing the unit vectors of axisD, the distal axis.

        Returns
        -------
        angle : ndarray
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding ndarray.

        Examples
        --------
        >>> import numpy as np
        >>> from pycgm.pycgm import StaticCGM
        >>> axis_p = [[ 0.59, 0.15, 0.15],
        ...         [-0.13, -0.16, -0.93],
        ...         [0.93, -0.04, 0.75]]
        >>> axis_d = [[0.16, 0.69, -0.37],
        ...         [0.14, -0.39, 0.94],
        ...         [-0.15, -0.53, -0.61]]
        >>> np.around(StaticCGM.ankle_angle_calc(axis_p,axis_d),8)
        array([ 0.54405441,  0.99687718, -1.4287752 ])
        """
        # make inverse matrix of axisP
        axis_p_i = np.linalg.inv(axis_p)

        # M is multiply of axis_d and axis_p_i
        m = np.matmul(axis_d, axis_p_i)

        # This is the angle calculation in YXZ Euler angle
        get_a = m[2][1] / sqrt((m[2][0] * m[2][0]) + (m[2][2] * m[2][2]))
        get_b = -1 * m[2][0] / m[2][2]
        get_g = -1 * m[0][1] / m[1][1]

        gamma = np.arctan(get_g)
        alpha = np.arctan(get_a)
        beta = np.arctan(get_b)

        angle = np.array([alpha, beta, gamma])
        return angle

    def get_static(self, flat_foot=False, gcs=None):
        """ Get Static Offset function

        Calculate the static offset angle values and return the values in radians

        Parameters
        ----------
        flat_foot : boolean, optional
            A boolean indicating if the feet are flat or not.
            The default value is False.
        gcs : ndarray, optional
            An array containing the Global Coordinate System.
            If not provided, the default will be set to: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].

        Returns
        -------
        cal_sm : dict
            Dictionary containing the calibrated subject measurements.

        Examples
        --------
        >>> from pycgm.pycgm import StaticCGM
        >>> import pycgm
        >>> static_trial, dynamic_trial, measurements = pycgm.get_rom_data()
        >>> static = StaticCGM(static_trial, measurements)
        >>> static.measurements['HeadOffset']
        0.0
        >>> cal_sm = static.get_static()
        >>> np.around(cal_sm['HeadOffset'],8)
        0.25719905
        """
        static_offset = []
        head_offset = []
        iad = []
        cal_sm = {}
        left_leg_length = self.measurements['LeftLegLength']
        right_leg_length = self.measurements['RightLegLength']
        cal_sm['MeanLegLength'] = (left_leg_length + right_leg_length) / 2.0
        cal_sm['Bodymass'] = self.measurements['Bodymass']

        # Define the global coordinate system
        if gcs is None:
            cal_sm['GCS'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        if self.measurements['LeftAsisTrocanterDistance'] != 0 and self.measurements['RightAsisTrocanterDistance'] != 0:
            cal_sm['L_AsisToTrocanterMeasure'] = self.measurements['LeftAsisTrocanterDistance']
            cal_sm['R_AsisToTrocanterMeasure'] = self.measurements['RightAsisTrocanterDistance']
        else:
            cal_sm['R_AsisToTrocanterMeasure'] = (0.1288 * right_leg_length) - 48.56
            cal_sm['L_AsisToTrocanterMeasure'] = (0.1288 * left_leg_length) - 48.56

        if self.measurements['InterAsisDistance'] != 0:
            cal_sm['InterAsisDistance'] = self.measurements['InterAsisDistance']
        else:
            for frame in self.motion_data:
                rasi, lasi = frame[self.mapping['RASI']], frame[self.mapping['LASI']]
                iad_calc = StaticCGM.iad_calculation(rasi, lasi)
                iad.append(iad_calc)
            inter_asis_distance = np.average(iad)
            cal_sm['InterAsisDistance'] = inter_asis_distance

        try:
            cal_sm['RightKneeWidth'] = self.measurements['RightKneeWidth']
            cal_sm['LeftKneeWidth'] = self.measurements['LeftKneeWidth']

        except KeyError:
            # no knee width
            cal_sm['RightKneeWidth'] = 0
            cal_sm['LeftKneeWidth'] = 0

        if cal_sm['RightKneeWidth'] == 0:
            if 'RMKN' in self.mapping:
                # medial knee markers are available
                rwidth = []
                lwidth = []
                # average each frame
                for frame in self.motion_data:
                    rmkn = frame[self.mapping['RMKN']]
                    lmkn = frame[self.mapping['LMKN']]

                    rkne = frame[self.mapping['RKNE']]
                    lkne = frame[self.mapping['LKNE']]

                    rdst = np.linalg.norm(rkne - rmkn)
                    ldst = np.linalg.norm(lkne - lmkn)
                    rwidth.append(rdst)
                    lwidth.append(ldst)

                cal_sm['RightKneeWidth'] = sum(rwidth) / len(rwidth)
                cal_sm['LeftKneeWidth'] = sum(lwidth) / len(lwidth)
        try:
            cal_sm['RightAnkleWidth'] = self.measurements['RightAnkleWidth']
            cal_sm['LeftAnkleWidth'] = self.measurements['LeftAnkleWidth']

        except KeyError:
            # no knee width
            cal_sm['RightAnkleWidth'] = 0
            cal_sm['LeftAnkleWidth'] = 0

        if cal_sm['RightAnkleWidth'] == 0:
            if 'RMKN' in self.mapping:
                # medial knee markers are available
                rwidth = []
                lwidth = []
                # average each frame
                for frame in self.motion_data:
                    rmma = frame[self.mapping['RMMA']]
                    lmma = frame[self.mapping['LMMA']]

                    rank = frame[self.mapping['RANK']]
                    lank = frame[self.mapping['LANK']]

                    rdst = np.linalg.norm(rmma - rank)
                    ldst = np.linalg.norm(lmma - lank)
                    rwidth.append(rdst)
                    lwidth.append(ldst)

                cal_sm['RightAnkleWidth'] = sum(rwidth) / len(rwidth)
                cal_sm['LeftAnkleWidth'] = sum(lwidth) / len(lwidth)

        cal_sm['RightTibialTorsion'] = self.measurements['RightTibialTorsion']
        cal_sm['LeftTibialTorsion'] = self.measurements['LeftTibialTorsion']

        cal_sm['RightShoulderOffset'] = self.measurements['RightShoulderOffset']
        cal_sm['LeftShoulderOffset'] = self.measurements['LeftShoulderOffset']

        cal_sm['RightElbowWidth'] = self.measurements['RightElbowWidth']
        cal_sm['LeftElbowWidth'] = self.measurements['LeftElbowWidth']
        cal_sm['RightWristWidth'] = self.measurements['RightWristWidth']
        cal_sm['LeftWristWidth'] = self.measurements['LeftWristWidth']

        cal_sm['RightHandThickness'] = self.measurements['RightHandThickness']
        cal_sm['LeftHandThickness'] = self.measurements['LeftHandThickness']

        for frame in self.motion_data:
            rasi, lasi = frame[self.mapping['RASI']], frame[self.mapping['LASI']]
            rpsi, lpsi, sacr = None, None, None
            try:
                sacr = frame[self.mapping['SACR']]
            except KeyError:
                rpsi = frame[self.mapping['RPSI']]
                lpsi = frame[self.mapping['LPSI']]
            pelvis = StaticCGM.pelvis_axis_calc(rasi, lasi, rpsi, lpsi, sacr)

            hip_axis = StaticCGM.hip_axis_calc(pelvis, cal_sm)
            hip_origin = [hip_axis[0], hip_axis[1]]

            rthi, lthi, rkne, lkne = frame[self.mapping['RTHI']], frame[self.mapping['LTHI']], frame[
                self.mapping['RKNE']], frame[
                                         self.mapping['LKNE']]
            knee_axis = StaticCGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, cal_sm)
            knee_origin = np.array([knee_axis[0], knee_axis[4]])

            rtib, ltib, rank, lank = frame[self.mapping['RTIB']], frame[self.mapping['LTIB']], frame[
                self.mapping['RANK']], frame[
                                         self.mapping['LANK']]
            ankle_axis = StaticCGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, cal_sm)

            rtoe, ltoe, rhee, lhee = frame[self.mapping['RTOE']], frame[self.mapping['LTOE']], frame[
                self.mapping['RHEE']], frame[
                                         self.mapping['LHEE']]
            angles = StaticCGM.static_calculation(rtoe, ltoe, rhee, lhee, ankle_axis, flat_foot, cal_sm)

            rfhd, lfhd, rbhd, lbhd = frame[self.mapping['RFHD']], frame[self.mapping['LFHD']], frame[
                self.mapping['RBHD']], frame[self.mapping['LBHD']]
            head_axis = StaticCGM.head_axis_calc(rfhd, lfhd, rbhd, lbhd)

            head_angle = StaticCGM.static_calculation_head(head_axis)

            static_offset.append(angles)
            head_offset.append(head_angle)

        static = np.average(static_offset, axis=0)
        static_head = np.average(head_offset)

        cal_sm['RightStaticRotOff'] = static[0][0] * -1
        cal_sm['RightStaticPlantFlex'] = static[0][1]
        cal_sm['LeftStaticRotOff'] = static[1][0]
        cal_sm['LeftStaticPlantFlex'] = static[1][1]
        cal_sm['HeadOffset'] = static_head

        return cal_sm
