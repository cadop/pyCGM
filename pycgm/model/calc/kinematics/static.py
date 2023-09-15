# -*- coding: utf-8 -*-
"""
This file provides helper functions for static calculations.

Created on Tue Jul 28 16:55:25 2015

@author: cadop
"""
import math

import numpy as np
import numpy.lib.recfunctions as rfn

from ..function import Function
from .shared import CalcUtils as CalcUtils

class CalcStatic():
    def __init__(self):
        self.funcs = [ self.calibrate_mean_leg_length,
                       self.calibrate_asis_to_troc_measure,
                       self.calibrate_inter_asis_distance,
                       self.calc_axis_pelvis,
                       self.calc_joint_center_hip,
                       self.calc_axis_knee,
                       self.calc_axis_ankle,
                       self.calc_axis_uncorrect_foot,
                       self.calc_axis_head,
                       self.calc_static_head, 
                       self.calc_axis_foot,
                       self.calc_static_ankle_offsets ]


    @Function.info(measurements=['RightLegLength', 'LeftLegLength'],
           returns_measurements=['MeanLegLength'])
    def calibrate_mean_leg_length(right_leg_length, left_leg_length):
        return (left_leg_length + right_leg_length) / 2.0


    @Function.info(measurements=['RightAsisTrocanterDistance', 'LeftAsisTrocanterDistance', 'RightLegLength', 'LeftLegLength', ('FlatFoot', np.bool_)],
           returns_measurements=['R_AsisToTrocanterMeasure', 'L_AsisToTrocanterMeasure'])
    def calibrate_asis_to_troc_measure(right_asis_to_trochanter, left_asis_to_trochanter, right_leg_length, left_leg_length, flat_foot):
        if left_asis_to_trochanter != 0 and right_asis_to_trochanter != 0:
            return np.array(right_asis_to_trochanter, left_asis_to_trochanter)
        else:
            right_asis_to_trochanter_calibrated = ( 0.1288 * right_leg_length ) - 48.56
            left_asis_to_trochanter_calibrated = ( 0.1288 * left_leg_length ) - 48.56
            return np.array([right_asis_to_trochanter_calibrated, left_asis_to_trochanter_calibrated])


    @Function.info(markers=['RASI', 'LASI'],
              measurements=['InterAsisDistance'],
      returns_measurements=['InterAsisDistance'])
    def calibrate_inter_asis_distance(rasi, lasi, inter_asis_distance):
        if inter_asis_distance != 0:
            return inter_asis_distance
        else:
            inter_asis_distance = np.linalg.norm(rasi - lasi, axis=1)
            return np.average(inter_asis_distance)


    @Function.info(markers=['RASI', 'LASI', 'RPSI', 'LPSI', 'SACR'],
              returns_axes=['Pelvis'])
    def calc_axis_pelvis(rasi, lasi, rpsi, lpsi, sacr=None):
        """
        Make the Pelvis Axis.
        """

        # Get the Pelvis Joint Centre
        if sacr is None:
            sacr = (rpsi + lpsi) / 2.0

        # Origin is Midpoint between RASI and LASI
        o = (rasi+lasi)/2.0


        b1 = o - sacr
        b2 = lasi - rasi

        # y is normalized b2
        y = b2 / np.linalg.norm(b2,axis=1)[:, np.newaxis]

        b3 = b1 - ( y * np.sum(b1*y,axis=1)[:, np.newaxis] )
        x = b3/np.linalg.norm(b3,axis=1)[:, np.newaxis]

        # Z-axis is cross product of x and y vectors.
        z = np.cross(x, y)

        num_frames = rasi.shape[0]
        pelvis_stack = np.column_stack([x,y,z,o])
        pelvis_matrix = pelvis_stack.reshape(num_frames,4,3).transpose(0,2,1)

        return pelvis_matrix


    @Function.info(measurements=['MeanLegLength', 'R_AsisToTrocanterMeasure', 'L_AsisToTrocanterMeasure', 'InterAsisDistance'],
                           axes=['Pelvis'],
                   returns_axes=['RHipJC', 'LHipJC'])
    def calc_joint_center_hip(mean_leg_length, right_asis_to_trochanter, left_asis_to_trochanter, inter_asis_distance, pelvis):
        u"""Calculate the right and left hip joint center.

        Takes in a 4x4 affine matrix of pelvis axis and subject measurements
        dictionary. Calculates and returns the right and left hip joint centers.

        Subject Measurement values used:
            MeanLegLength

            R_AsisToTrocanterMeasure

            InterAsisDistance

            L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation [1]_.

        Parameters
        ----------
        pelvis : array
            A 4x4 affine matrix with pelvis x, y, z axes and pelvis origin.
        subject : dict
            A dictionary containing subject measurements.

        Returns
        -------
        hip_jc : array
            A 2x3 array that contains two 1x3 arrays
            containing the x, y, z components of the right and left hip joint
            centers.

        References
        ----------
        .. [1] Davis, R. B., III, Õunpuu, S., Tyburski, D. and Gage, J. R. (1991).
                A gait analysis data collection and reduction technique.
                Human Movement Science 10 575–87.

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> from .pyCGM import calc_joint_center_hip
        >>> mean_leg_length = 940.0 
        >>> right_asis_to_trochanter = 72.51
        >>> left_asis_to_trochanter = 72.51
        >>> inter_asis_distance = 215.90
        >>> pelvis_axis = np.array([[
        ...                            [ 0.14, 0.98, -0.11,  251.60],
        ...                            [-0.99, 0.13, -0.02,  391.74],
        ...                            [ 0,    0.1,   0.99, 1032.89],
        ...                         ],
        ...                         [
        ...                            [ 0.14, 0.98, -0.11,  251.60],
        ...                            [-0.99, 0.13, -0.02,  391.74],
        ...                            [ 0,    0.1,   0.99, 1032.89],
        ...                        ]])
        >>> np.around(calc_joint_center_hip(pelvis_axis, mean_leg_length, right_asis_to_trochanter, left_asis_to_trochanter, inter_asis_distance), 2) #doctest: +NORMALIZE_WHITESPACE
        array([[307.36, 323.83, 938.72],
               [181.71, 340.33, 936.18]])
        """

        # Requires
        # pelvis axis

        # Model's eigen value
        #
        # LegLength
        # MeanLegLength
        # mm (marker radius)
        # interAsisMeasure

        # Set the variables needed to calculate the joint angle
        # Half of marker size
        mm = 7.0

        C = (mean_leg_length * 0.115) - 15.3
        theta = 0.500000178813934
        beta = 0.314000427722931
        aa = inter_asis_distance/2.0
        S = -1

        # Hip Joint Center Calculation (ref. Davis_1991)

        # Left: Calculate the distance to translate along the pelvis axis
        L_Xh = (-left_asis_to_trochanter - mm) * \
            math.cos(beta) + C * math.cos(theta) * math.sin(beta)
        L_Yh = S*(C*math.sin(theta) - aa)
        L_Zh = (-left_asis_to_trochanter - mm) * \
            math.sin(beta) - C * math.cos(theta) * math.cos(beta)

        # Right:  Calculate the distance to translate along the pelvis axis
        R_Xh = (-right_asis_to_trochanter - mm) * \
            math.cos(beta) + C * math.cos(theta) * math.sin(beta)
        R_Yh = (C*math.sin(theta) - aa)
        R_Zh = (-right_asis_to_trochanter - mm) * \
            math.sin(beta) - C * math.cos(theta) * math.cos(beta)

        # get the unit pelvis axis
        pelvis = np.array(pelvis)

        pelvis_xaxis = pelvis[:, :, 0]
        pelvis_yaxis = pelvis[:, :, 1]
        pelvis_zaxis = pelvis[:, :, 2]
        pel_origin   = pelvis[:, :, 3]
        pelvis_axis = np.array([pelvis_xaxis, pelvis_yaxis, pelvis_zaxis])

        # multiply the distance to the unit pelvis axis
        left_hip_jc_x = pelvis_xaxis * L_Xh
        left_hip_jc_y = pelvis_yaxis * L_Yh
        left_hip_jc_z = pelvis_zaxis * L_Zh


        left_hip_jc = np.array([left_hip_jc_x, left_hip_jc_y, left_hip_jc_z])
        left_hip_jc = np.matmul(pelvis_axis.T, np.array([L_Xh, L_Yh, L_Zh])).T

        right_hip_jc_x = pelvis_xaxis * R_Xh
        right_hip_jc_y = pelvis_yaxis * R_Yh
        right_hip_jc_z = pelvis_zaxis * R_Zh

        right_hip_jc = np.array([right_hip_jc_x, right_hip_jc_y, right_hip_jc_z])
        right_hip_jc = np.matmul(pelvis_axis.T, np.array([R_Xh, R_Yh, R_Zh])).T


        left_hip_jc = left_hip_jc+pel_origin
        right_hip_jc = right_hip_jc+pel_origin

        num_frames = pelvis.shape[0]

        right = np.zeros((num_frames, 4, 3))
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = right_hip_jc

        right_stack = np.column_stack([x, y, z, o])
        right_hip_jc_matrix = right_stack.reshape(num_frames, 4, 3).transpose(0,2,1)

        left = np.zeros((num_frames, 4, 3))
        x = np.zeros((num_frames, 3))
        y = np.zeros((num_frames, 3))
        z = np.zeros((num_frames, 3))
        o = left_hip_jc

        left_stack = np.column_stack([x, y, z, o])
        left_hip_jc_matrix = left_stack.reshape(num_frames, 4, 3).transpose(0,2,1)

        hip_jc = np.array([right_hip_jc_matrix, left_hip_jc_matrix])

        return hip_jc


    @Function.info(markers=['RTHI', 'LTHI', 'RKNE', 'LKNE'],
              measurements=['RightKneeWidth', 'LeftKneeWidth'],
                      axes=['RHipJC', 'LHipJC'],
              returns_axes=['RKnee', 'LKnee'])
    def calc_axis_knee(rthi, lthi, rkne, lkne, rkne_width, lkne_width, r_hip_jc, l_hip_jc):
        """Calculate the knee joint center and axis.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame, the hip joint center, and knee widths.

        Markers used: RTHI, LTHI, RKNE, LKNE, r_hip_jc, l_hip_jc

        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation [1]_.

        Parameters
        ----------
        rthi : array
            1x3 RTHI marker
        lthi : array
            1x3 LTHI marker
        rkne : array
            1x3 RKNE marker
        lkne : array
            1x3 LKNE marker
        r_hip_jc : array
            4x4 affine matrix containing the right hip joint center.
        l_hip_jc : array
            4x4 affine matrix containing the left hip joint center.
        rkne_width : float
            The width of the right knee
        lkne_width : float
            The width of the left knee

        Returns
        -------
        [r_axis, l_axis] : array
            An array of two 4x4 affine matrices representing the right and left
            knee axes and joint centers.

        References
        ----------
        .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
                analysis. Mac Keith Press.

        Notes
        -----
        Delta is changed suitably to knee.

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> rthi = np.array([426.50, 262.65, 673.66])
        >>> lthi = np.array([51.93, 320.01, 723.03])
        >>> rkne = np.array([416.98, 266.22, 524.04])
        >>> lkne = np.array([84.62, 286.69, 529.39])
        >>> l_hip_jc = [182.57, 339.43, 935.52]
        >>> r_hip_jc = [309.38, 322.80, 937.98]
        >>> rkne_width = 105.0
        >>> lkne_width = 105.0
        >>> [arr.round(2) for arr in calc_axis_knee(rthi, lthi, rkne, lkne, l_hip_jc, r_hip_jc, rkne_width, lkne_width)] #doctest: +NORMALIZE_WHITESPACE
        [array([[  0.3 ,   0.95,   0.  , 365.09],
            [ -0.87,   0.28,  -0.4 , 282.84],
            [ -0.38,   0.12,   0.92, 500.13],
            [  0.  ,   0.  ,   0.  ,   1.  ]]),
         array([[  0.11,   0.98,  -0.15, 139.57],
            [ -0.92,   0.16,   0.35, 277.13],
            [  0.37,   0.1 ,   0.93, 508.67],
            [  0.  ,   0.  ,   0.  ,   1.  ]])]
        """
        # Get Global Values
        mm = 7.0
        r_delta = (rkne_width/2.0) + mm
        l_delta = (lkne_width/2.0) + mm
        r_hip_jc = r_hip_jc[:, :, 3]
        l_hip_jc = l_hip_jc[:, :, 3]

        # Determine the position of kneeJointCenter using calc_joint_center function
        r_knee_o = CalcUtils.calc_joint_center(rthi, r_hip_jc, rkne, r_delta)
        l_knee_o = CalcUtils.calc_joint_center(lthi, l_hip_jc, lkne, l_delta)

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = r_hip_jc-r_knee_o

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        # axis_x = cross(axis_z,thi_kne_R)
        axis_x = np.cross(axis_z, rkne-r_hip_jc)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = l_hip_jc-l_knee_o

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        # axis_x = cross(thi_kne_L,axis_z)
        # using hipjc instead of thigh marker
        axis_x = np.cross(lkne-l_hip_jc, axis_z)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        r_knee_x = r_axis[0]/np.linalg.norm(r_axis[0], axis=1)[:, np.newaxis] 
        r_knee_y = r_axis[1]/np.linalg.norm(r_axis[1], axis=1)[:, np.newaxis]
        r_knee_z = r_axis[2]/np.linalg.norm(r_axis[2], axis=1)[:, np.newaxis]

        l_knee_x = l_axis[0]/np.linalg.norm(l_axis[0], axis=1)[:, np.newaxis]
        l_knee_y = l_axis[1]/np.linalg.norm(l_axis[1], axis=1)[:, np.newaxis]
        l_knee_z = l_axis[2]/np.linalg.norm(l_axis[2], axis=1)[:, np.newaxis]

        num_frames = rthi.shape[0]

        r_knee_axis = np.column_stack([r_knee_x, r_knee_y, r_knee_z, r_knee_o])
        l_knee_axis = np.column_stack([l_knee_x, l_knee_y, l_knee_z, l_knee_o])

        r_axis_matrix = r_knee_axis.reshape(num_frames, 4, 3).transpose(0, 2, 1)
        l_axis_matrix = l_knee_axis.reshape(num_frames, 4, 3).transpose(0, 2, 1)

        return np.asarray([r_axis_matrix, l_axis_matrix])


    @Function.info(markers=['RTIB', 'LTIB', 'RANK', 'LANK'],
              measurements=['RightAnkleWidth', 'LeftAnkleWidth', 'RightTibialTorsion', 'LeftTibialTorsion'],
                      axes=['RKnee',  'LKnee'],
              returns_axes=['RAnkle', 'LAnkle'])
    def calc_axis_ankle(rtib, ltib, rank, lank, rank_width, lank_width, rtib_torsion, ltib_torsion, r_knee_jc, l_knee_jc):
        """Calculate the ankle joint center and axis.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame, the knee joint centers, ankle widths, and tibial torsions.

        Markers used: RTIB, LTIB, RANK, LANK, r_knee_JC, l_knee_JC

        Subject Measurement values used:
            RightKneeWidth

            LeftKneeWidth

            RightTibialTorsion

            LeftTibialTorsion

        Ankle Axis: Computed using Ankle Axis Calculation [1]_.

        Parameters
        ----------
        rtib : array
            1x3 RTIB marker
        ltib : array
            1x3 LTIB marker
        rank : array
            1x3 RANK marker
        lank : array
            1x3 LANK marker
        r_knee_JC : array
            The (x,y,z) position of the right knee joint center.
        l_knee_JC : array
            The (x,y,z) position of the left knee joint center.
        rank_width : float
            The width of the right ankle
        lank_width : float
            The width of the left ankle
        rtib_torsion : float
            Right tibial torsion angle
        ltib_torsion : float
            Left tibial torsion angle

        Returns
        -------
        [r_axis, l_axis] : array
            An array of two 4x4 affine matrices representing the right and left
            ankle axes and joint centers.

        References
        ----------
        .. [1] Baker, R. (2013). Measuring walking : a handbook of clinical gait
                analysis. Mac Keith Press.

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> rank_width = 70.0
        >>> lank_width = 70.0
        >>> rtib_torsion = 0.0
        >>> ltib_torsion = 0.0
        >>> rtib = np.array([433.97, 211.93, 273.30])
        >>> ltib = np.array([50.04, 235.90, 364.32])
        >>> rank = np.array([422.77, 217.74, 92.86])
        >>> lank = np.array([58.57, 208.54, 86.16])
        >>> knee_JC = np.array([[365.09, 282.84, 500.13],
        ...                     [139.57, 277.13, 508.67]])
        >>> [np.around(arr, 2) for arr in calc_axis_ankle(rtib, ltib, rank, lank, knee_JC[0], knee_JC[1], rank_width, lank_width, rtib_torsion, ltib_torsion)] #doctest: +NORMALIZE_WHITESPACE
                    [array([[  0.69,   0.73,  -0.02, 392.33],
                            [ -0.72,   0.68,  -0.11, 246.32],
                            [ -0.07,   0.09,   0.99,  88.31],
                            [  0.  ,   0.  ,   0.  ,   1.  ]]),
                     array([[ -0.28,   0.96,  -0.1 ,  98.76],
                            [ -0.96,  -0.26,   0.13, 219.53],
                            [  0.09,   0.13,   0.99,  80.85],
                            [  0.  ,   0.  ,   0.  ,   1.  ]])]
        """
        # Get Global Values
        mm = 7.0
        r_delta = (rank_width/2.0) + mm
        l_delta = (lank_width/2.0) + mm
        r_knee_jc = r_knee_jc[:, :, 3]
        l_knee_jc = l_knee_jc[:, :, 3]

        # This is Torsioned Tibia and this describe the ankle angles
        # Tibial frontal plane being defined by ANK,TIB and KJC

        # Determine the position of ankleJointCenter using calc_joint_center function
        r_ankle_jc = CalcUtils.calc_joint_center(rtib, r_knee_jc, rank, r_delta)
        l_ankle_jc = CalcUtils.calc_joint_center(ltib, l_knee_jc, lank, l_delta)

        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        # Right axis calculation

        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = r_knee_jc - r_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector.
        # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_R = rtib - rank
        axis_x = np.cross(axis_z, tib_ank_R)

        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation

        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = l_knee_jc - l_ankle_jc

        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector.
        # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_L = ltib - lank
        axis_x = np.cross(tib_ank_L, axis_z)

        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then normalize it.
        r_ankle_x = r_axis[0]/np.linalg.norm(r_axis[0], axis=1)[:, np.newaxis]
        r_ankle_y = r_axis[1]/np.linalg.norm(r_axis[1], axis=1)[:, np.newaxis]
        r_ankle_z = r_axis[2]/np.linalg.norm(r_axis[2], axis=1)[:, np.newaxis]

        l_ankle_x = l_axis[0]/np.linalg.norm(l_axis[0], axis=1)[:, np.newaxis]
        l_ankle_y = l_axis[1]/np.linalg.norm(l_axis[1], axis=1)[:, np.newaxis]
        l_ankle_z = l_axis[2]/np.linalg.norm(l_axis[2], axis=1)[:, np.newaxis]

        # Put both axis in array
        r_axis = np.array([r_ankle_x, r_ankle_y, r_ankle_z])
        l_axis = np.array([l_ankle_x, l_ankle_y, l_ankle_z])


        # Rotate the axes about the tibia torsion.
        rtib_torsion = np.radians(rtib_torsion)
        ltib_torsion = np.radians(ltib_torsion)

        r_axis_x, r_axis_y, r_axis_z = np.cos(rtib_torsion) * r_axis - np.sin(rtib_torsion) * r_axis
        l_axis_x, l_axis_y, l_axis_z = np.cos(ltib_torsion) * l_axis - np.sin(ltib_torsion) * l_axis

        r_ankle_axis = np.column_stack([r_axis_x, r_axis_y, r_axis_z, r_ankle_jc])
        l_ankle_axis = np.column_stack([l_axis_x, l_axis_y, l_axis_z, l_ankle_jc])

        num_frames = rtib.shape[0]
        r_axis_matrix = r_ankle_axis.reshape(num_frames, 4, 3).transpose(0, 2, 1)
        l_axis_matrix = l_ankle_axis.reshape(num_frames, 4, 3).transpose(0, 2, 1)


        return np.asarray([r_axis_matrix, l_axis_matrix])


    @Function.info(axes=['Head'],
   returns_measurements=['HeadOffset'])
    def calc_static_head(head_axis):
        """Static Head Calculation

        This function converts the head axis to a numpy array,
        and then calculates the offset of the head using the calc_head_offset function.

        Parameters
        ----------
        head_axis : array
            4x4 affine matrix containing the head (x, y, z) axes and origin

        Returns
        -------
        offset : float
            The head offset angle for static calibration.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgmStatic import calc_static_head
        >>> head_axis = np.array([[ 0.75,    0.6 ,    0.28,   99.58],
        ...                       [-0.61,    0.79,   -0.03,   82.79],
        ...                       [-0.23,   -0.15,    0.96, 1483.8 ],
        ...                       [ 0.  ,    0.  ,    0.  ,    0.  ]])
        >>> np.around(calc_static_head(head_axis), 2)
        0.28
        """

        def calc_head_offset(axis_p, axis_d):
            """Head Offset Calculation

            Calculate head offset angle for static calibration.
            This function is only called in static trial.
            Output will be used later in the dynamic trial.

            Parameters
            ----------
            axisP : array
                4x4 affine matrix representing the position of the proximal axis.
            axisD : array
                4x4 affine matrix representing the position of the distal axis.

            Returns
            -------
            angle : float
                The beta angle of the head offset.

            Examples
            --------
            >>> import numpy as np
            >>> from .pycgmStatic import calc_head_offset
            >>> axisP = np.array([[0.96, 0.81, 0.82, 0],
            ...                   [0.24, 0.72, 0.38, 0],
            ...                   [0.98, 0.21, 0.68, 0],
            ...                   [0,    0,    0,    1]])
            >>> axisD = np.array([[0.21, 0.25, 0.94, 0],
            ...                   [0.8,  0.45, 0.91, 0],
            ...                   [0.17, 0.67, 0.85, 0],
            ...                   [0,    0,    0,    1]])
            >>> np.around(calc_head_offset(axisP,axisD), 2)
            0.95
            """
            axis_p_inv = np.linalg.inv(axis_p)
            
            # Repeat GCS to match shape of vectorized head axis
            axis_p_inv_stack = np.repeat(axis_p_inv[None,...], axis_d.shape[0], axis=0)

            # rotation matrix is in order XYZ
            M = np.matmul(axis_d, axis_p_inv_stack)

            # get y angle from rotation matrix using inverse trigonometry.
            b = np.divide(M[:, 0, 2], M[:, 2, 2])

            angle = np.arctan(b)

            return angle

        head_axis = head_axis[:, :, :3].transpose(0,2,1)

        global_axis = np.array([[ 0, 1, 0],
                                [-1, 0, 0],
                                [ 0, 0, 1]])

        offset = calc_head_offset(global_axis, head_axis)

        return np.average(offset)


    @Function.info(markers=['LFHD', 'RFHD', 'LBHD', 'RBHD'],
              returns_axes=['Head'])
    def calc_axis_head(lfhd, rfhd, lbhd, rbhd):
        """Calculate the head joint center and axis.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame, and the head offset. 

        Calculates the head joint center and axis.

        Markers used: LFHD, RFHD, LBHD, RBHD

        Parameters
        ----------
        lfhd : array
            1x3 LFHD marker
        rfhd : array
            1x3 RFHD marker
        lbhd : array
            1x3 LBHD marker
        rbhd : array
            1x3 RBHD marker

        Returns
        -------
        head_axis : array
            4x4 affine matrix with head (x, y, z) axes and origin.


        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> from .pycgmStatic import calc_axis_head
        >>> rfhd = np.array([325.82, 402.55, 1722.49])
        >>> lfhd = np.array([184.55, 409.68, 1721.34])
        >>> rbhd = np.array([304.39, 242.91, 1694.97])
        >>> lbhd = np.array([197.86, 251.28, 1696.90])
        >>> [np.around(arr, 2) for arr in calc_axis_head(lfhd, rfhd, lbhd, rbhd)] #doctest: +NORMALIZE_WHITESPACE
        [array([  0.03,   0.99,  0.16,  255.18]),
         array([ -1.  ,   0.03, -0.  ,  406.12]),
         array([ -0.01,  -0.16,  0.99, 1721.92]),
           array([0.,     0.,    0.,      1.])]
        """

        # get the midpoints of the head to define the sides
        front = (lfhd + rfhd) / 2.0
        back  = (lbhd + rbhd) / 2.0
        left  = (lfhd + lbhd) / 2.0
        right = (rfhd + rbhd) / 2.0

        # Get the vectors from the sides with primary x axis facing front
        # First get the x direction
        x_axis = np.subtract(front, back)
        x_axis_norm = np.linalg.norm(x_axis, axis=1)[:, np.newaxis]
        x_axis = np.divide(x_axis, x_axis_norm)

        # get the direction of the y axis
        y_axis = np.subtract(left, right)
        y_axis_norm = np.linalg.norm(y_axis, axis=1)[:, np.newaxis]
        y_axis = np.divide(y_axis, y_axis_norm)

        # get z axis by cross-product of x axis and y axis.
        z_axis = np.cross(x_axis, y_axis)
        z_axis_norm = np.linalg.norm(z_axis, axis=1)[:, np.newaxis]
        z_axis = np.divide(z_axis, z_axis_norm)

        # make sure all x,y,z axis is orthogonal each other by cross-product
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis, axis=1)[:, np.newaxis]
        y_axis = np.divide(y_axis, y_axis_norm)

        x_axis = np.cross(y_axis, z_axis)
        x_axis_norm = np.linalg.norm(x_axis, axis=1)[:, np.newaxis]
        x_axis = np.divide(x_axis, x_axis_norm)

        # Create the return matrix
        num_frames = rfhd.shape[0]
        head_axis = np.column_stack([x_axis, y_axis, z_axis, front])
        head_axis = head_axis.reshape(num_frames,4,3).transpose(0,2,1)

        return head_axis


    @Function.info(markers=['RTOE', 'LTOE'],
                      axes=['RAnkle', 'LAnkle'],
              returns_axes=['RFootUncorrected', 'LFootUncorrected'])
    def calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis):
        """Calculate the foot joint center and axis.

        Takes in markers that correspond to (x, y, z) positions of the current
        frame, the right and left ankle axes, right and left static rotation
        offset angles, and the right and left static plantar flexion angles.

        Calculates the foot joint axis by rotating incorrect foot joint axes about
        offset angle.

        Markers used: RTOE, LTOE

        Other landmarks used: ankle axis

        Subject Measurement values used: 
            RightStaticRotOff

            RightStaticPlantFlex

            LeftStaticRotOff

            LeftStaticPlantFlex

        Parameters
        ----------
        rtoe : array
            1x3 RTOE marker
        ltoe : array
            1x3 LTOE marker
        r_ankle_axis : array
            4x4 affine matrix with right ankle x, y, z axes and origin.
        l_ankle_axis : array
            4x4 affine matrix with left ankle x, y, z axes and origin.
        r_static_rot_off : float
            Right static offset angle.
        l_static_rot_off : float
            Left static offset angle.
        r_static_plant_flex : float
            Right static plantar flexion angle.
        l_static_plant_flex : float
            Left static plantar flexion angle.

        Returns
        -------
        [r_axis, l_axis] : array
            A list of two 4x4 affine matrices representing the right and left
            foot axes and origins.

        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> from .pycgmStatic import calc_axis_uncorrect_foot
        >>> rtoe = np.array([442.81, 381.62, 42.66])
        >>> ltoe = np.array([39.43, 382.44, 41.78])
        >>> r_ankle_axis = np.array([[  0.69,   0.73,  -0.02, 392.33],
        ...                          [ -0.72,   0.68,  -0.11, 246.32],
        ...                          [ -0.07,   0.09,   0.99,  88.31],
        ...                          [  0.  ,   0.  ,   0.  ,   1.  ]])
        >>> l_ankle_axis = np.array([[ -0.28,   0.96,  -0.1 ,  98.76],
        ...                         [  -0.96,  -0.26,   0.13, 219.53],
        ...                         [   0.09,   0.13,   0.99,  80.85],
        ...                         [   0.  ,   0.  ,   0.  ,   1.  ]])
        >>> [np.around(arr, 2) for arr in calc_axis_uncorrect_foot(rtoe, ltoe, r_ankle_axis, l_ankle_axis)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ 0.12,   0.28,   0.95, 442.81],
               [ -0.94,   0.35,   0.01, 381.62],
               [ -0.33,  -0.89,   0.3 ,  42.66],
               [  0.  ,   0.  ,   0.  ,   1.  ]]), 
        array([[  0.06,   0.25,   0.97,  39.43],
               [ -0.94,  -0.31,   0.14, 382.44],
               [  0.33,  -0.92,   0.22,  41.78],
               [  0.  ,   0.  ,   0.  ,   1.  ]])]
        """

        # REQUIRE JOINT CENTER & AXIS
        # KNEE JOINT CENTER
        # ANKLE JOINT CENTER
        # ANKLE FLEXION AXIS
        r_ankle_axis = np.asarray(r_ankle_axis)
        l_ankle_axis = np.asarray(l_ankle_axis)

        ankle_jc_r = r_ankle_axis[:, :, 3]
        ankle_jc_l = l_ankle_axis[:, :, 3]

        ankle_flexion_r = r_ankle_axis[:, :, 1] + ankle_jc_r
        ankle_flexion_l = l_ankle_axis[:, :, 1] + ankle_jc_l

        # Toe axis's origin is marker position of TOE
        r_origin = rtoe
        l_origin = ltoe

        # Right

        # z axis is from TOE marker to AJC. and normalized it.
        r_axis_z = ankle_jc_r - rtoe
        r_axis_z_div = np.linalg.norm(r_axis_z, axis=1)[:, np.newaxis]
        r_axis_z = r_axis_z / r_axis_z_div

        # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
        y_flex_r = ankle_flexion_r - ankle_jc_r
        y_flex_r_div = np.linalg.norm(y_flex_r, axis=1)[:, np.newaxis]
        y_flex_r = y_flex_r / y_flex_r_div

        # x axis is calculated as a cross product of z axis and ankle flexion axis.
        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x_div = np.linalg.norm(r_axis_x, axis=1)[:, np.newaxis]
        r_axis_x = r_axis_x / r_axis_x_div

        # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y_div = np.linalg.norm(r_axis_y, axis=1)[:, np.newaxis]
        r_axis_y = r_axis_y / r_axis_y_div

        r_foot_axis = np.column_stack([r_axis_x, r_axis_y, r_axis_z, r_origin])

        # Left

        # z axis is from TOE marker to AJC. and normalized it.
        l_axis_z = ankle_jc_l - ltoe
        l_axis_z_div = np.linalg.norm(l_axis_z, axis=1)[:, np.newaxis]
        l_axis_z = l_axis_z / l_axis_z_div

        # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
        y_flex_l = ankle_flexion_l - ankle_jc_l
        y_flex_l_div = np.linalg.norm(y_flex_l, axis=1)[:, np.newaxis]
        y_flex_l = y_flex_l / y_flex_l_div

        # x axis is calculated as a cross product of z axis and ankle flexion axis.
        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x_div = np.linalg.norm(l_axis_x, axis=1)[:, np.newaxis]
        l_axis_x = l_axis_x / l_axis_x_div

        # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y_div = np.linalg.norm(l_axis_y, axis=1)[:, np.newaxis]
        l_axis_y = l_axis_y / l_axis_y_div

        l_foot_axis = np.column_stack([l_axis_x, l_axis_y, l_axis_z, l_origin])

        num_frames = rtoe.shape[0]
        r_foot = r_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
        l_foot = l_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

        foot_axis = np.array([r_foot, l_foot])

        return foot_axis


    @Function.info(markers=['RTOE', 'LTOE', 'RHEE', 'LHEE'],
              measurements=[('FlatFoot', np.bool_), 'RightSoleDelta', 'LeftSoleDelta'],
                      axes=['RAnkle', 'LAnkle'],
              returns_axes=['RFoot', 'LFoot'])
    def calc_axis_foot(rtoe, ltoe, rhee, lhee, flat_foot, r_sole_delta, l_sole_delta, r_ankle_axis, l_ankle_axis):
        """Calculate the anatomically correct foot joint center and axis for a non-flat foot.

        Takes in the RTOE, LTOE, RHEE and LHEE marker positions
        as well as the ankle axes. Calculates the anatomically
        correct foot axis for non-flat feet.

        Markers used: RTOE, LTOE, RHEE, LHEE

        Parameters
        ----------
        rtoe : array
            1x3 RTOE marker
        ltoe : array
            1x3 LTOE marker
        rhee : array
            1x3 RHEE marker
        lhee : array
            1x3 LHEE marker
        ankle_axis : array
            array of two 4x4 affine matrices representing the right and left ankle axes and origins

        Returns
        -------
        axis : array
            An array of two 4x4 affine matrices representing the right and left non-flat foot
            axes and origins


        Examples
        --------
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> from .pycgmStatic import calc_axis_nonflatfoot
        >>> rhee = [374.01, 181.58, 49.51]
        >>> lhee = [105.30, 180.21, 47.16]
        >>> rtoe = [442.82, 381.62, 42.66]
        >>> ltoe = [39.44, 382.45, 41.79]
        >>> ankle_axis = np.array([[[ 0.72,   0.69,  -0.02, 393.76],
        ...                        [ -0.69,   0.71,  -0.12, 247.68],
        ...                        [ -0.07,   0.1 ,   0.99,  87.74],
        ...                        [  0.  ,   0.  ,   0.  ,   1.  ]],
        ...                       [[ -0.27,   0.96,  -0.1 ,  98.75],
        ...                        [ -0.96,  -0.26,   0.13, 219.47],
        ...                        [  0.1 ,   0.13,   0.99,  80.63],
        ...                        [  0.  ,   0.  ,   0.  ,   1.  ]]])
        >>> [np.around(arr, 2) for arr in calc_axis_nonflatfoot(rtoe, ltoe, rhee, lhee, ankle_axis)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ -0.1 ,   0.07,   0.99, 442.82],
                [ -0.94,   0.32,  -0.12, 381.62],
                [ -0.33,  -0.95,   0.03,  42.66],
                [  0.  ,   0.  ,   0.  ,   1.  ]]), 
         array([[  0.12,   0.06,   0.99,  39.44],
                [ -0.94,  -0.3 ,   0.13, 382.45],
                [  0.31,  -0.95,   0.03,  41.79],
                [  0.  ,   0.  ,   0.  ,   1.  ]])]
        """
        #REQUIRED MARKERS:
        # RTOE
        # LTOE
        # RHEE
        # LHEE
        # ankle_axis

        ankle_jc_right = r_ankle_axis[:, :, 3]
        ankle_jc_left = l_ankle_axis[:, :, 3]
        ankle_flexion_right = r_ankle_axis[:, :, 1]  + ankle_jc_right
        ankle_flexion_left = l_ankle_axis[:, :, 1]  + ankle_jc_left

        # Toe axis's origin is marker position of TOE
        right_origin = rtoe
        left_origin = ltoe

        if not flat_foot:
            # in case of non foot flat we just use the HEE marker
            right_axis_z = rhee - rtoe
            right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

            y_flex_R = ankle_flexion_right - ankle_jc_right
            y_flex_R = np.divide(y_flex_R, np.linalg.norm(y_flex_R, axis=1)[:, np.newaxis])

            right_axis_x = np.cross(y_flex_R, right_axis_z)
            right_axis_x = np.divide(right_axis_x, np.linalg.norm(right_axis_x, axis=1)[:, np.newaxis])

            right_axis_y = np.cross(right_axis_z, right_axis_x)
            right_axis_y = np.divide(right_axis_y, np.linalg.norm(right_axis_y, axis=1)[:, np.newaxis])

            r_foot_axis = np.column_stack([right_axis_x, right_axis_y, right_axis_z, right_origin])

            # Left
            left_axis_z = lhee - ltoe
            left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z))

            y_flex_L = ankle_flexion_left - ankle_jc_left
            y_flex_L = np.divide(y_flex_L, np.linalg.norm(y_flex_L, axis=1)[:, np.newaxis])

            left_axis_x = np.cross(y_flex_L, left_axis_z)
            left_axis_x = np.divide(left_axis_x, np.linalg.norm(left_axis_x, axis=1)[:, np.newaxis])

            left_axis_y = np.cross(left_axis_z, left_axis_x)
            left_axis_y = np.divide(left_axis_y, np.linalg.norm(left_axis_y, axis=1)[:, np.newaxis])

            l_foot_axis = np.column_stack([left_axis_x, left_axis_y, left_axis_z, left_origin])

            num_frames = rtoe.shape[0]
            r_foot = r_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
            l_foot = l_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

            axis = np.array([r_foot, l_foot])

            return axis
        
        elif flat_foot:
            ankle_jc_right[:, 2] += r_sole_delta
            ankle_jc_left[:, 2]  += l_sole_delta

            # Calculate the z axis
            right_axis_z = ankle_jc_right - rtoe
            right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

            # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
            heel_to_toe = rhee - rtoe
            heel_to_toe[:, 2] = 0
            heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe, axis=1)[:, np.newaxis])

            A = np.cross(heel_to_toe, right_axis_z)
            A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
            B = np.cross(A, heel_to_toe)
            B /= np.linalg.norm(B, axis=1)[:, np.newaxis]
            C = np.cross(B, A)
            right_axis_z = C / np.linalg.norm(C, axis=1)[:, np.newaxis]

            # Bring flexion axis from ankle axis
            right_y_flex = ankle_flexion_right - ankle_jc_right
            right_y_flex = np.divide(right_y_flex, np.linalg.norm(right_y_flex, axis=1)[:, np.newaxis])

            # Calculate each x,y,z axis of foot using np.cross-product and make sure x,y,z axis is orthogonal each other.
            right_axis_x = np.cross(right_y_flex,right_axis_z)
            right_axis_x = np.divide(right_axis_x, np.linalg.norm(right_axis_x, axis=1)[:, np.newaxis])

            right_axis_y = np.cross(right_axis_z,right_axis_x)
            right_axis_y = np.divide(right_axis_y, np.linalg.norm(right_axis_y, axis=1)[:, np.newaxis])

            right_axis_z = np.cross(right_axis_x,right_axis_y)
            right_axis_z = np.divide(right_axis_z, np.linalg.norm(right_axis_z, axis=1)[:, np.newaxis])

            right_foot_axis = np.column_stack([right_axis_x, right_axis_y, right_axis_z, right_origin])

            # Left

            # Calculate the z axis of foot flat.
            left_axis_z = ankle_jc_left - ltoe
            left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z, axis=1)[:, np.newaxis])

            # For foot flat, Z axis pointing same height of TOE marker from TOE to AJC
            heel_to_toe = lhee - ltoe
            heel_to_toe[:, 2] = 0
            heel_to_toe = np.divide(heel_to_toe, np.linalg.norm(heel_to_toe, axis=1)[:, np.newaxis])
            A = np.cross(heel_to_toe, left_axis_z)
            A = A / np.linalg.norm(A)
            B = np.cross(A, heel_to_toe)
            B = B / np.linalg.norm(B)
            C = np.cross(B, A)
            left_axis_z = C / np.linalg.norm(C)

            # Bring flexion axis from ankle axis
            left_y_flex = ankle_flexion_left - ankle_jc_left
            left_y_flex = np.divide(left_y_flex, np.linalg.norm(left_y_flex, axis=1)[:, np.newaxis])

            # Calculate each x,y,z axis of foot using np.cross-product and make sure (x, y, z) axes are orthogonal to each other
            left_axis_x = np.cross(left_y_flex,left_axis_z)
            left_axis_x = np.divide(left_axis_x, np.linalg.norm(left_axis_x, axis=1)[:, np.newaxis])

            left_axis_y = np.cross(left_axis_z,left_axis_x)
            left_axis_y = np.divide(left_axis_y, np.linalg.norm(left_axis_y, axis=1)[:, np.newaxis])

            left_axis_z = np.cross(left_axis_x,left_axis_y)
            left_axis_z = np.divide(left_axis_z, np.linalg.norm(left_axis_z, axis=1)[:, np.newaxis])

            left_foot_axis = np.column_stack([left_axis_x, left_axis_y, left_axis_z, left_origin])

            num_frames = rtoe.shape[0]
            r_foot = right_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)
            l_foot = left_foot_axis.reshape(num_frames,4,3).transpose(0,2,1)

            axis = np.array([r_foot, l_foot])

            return axis


    @Function.info(axes=['RFootUncorrected', 'RFoot', 'LFootUncorrected', 'LFoot'],
   returns_measurements=['RightStaticRotOff', 'RightStaticPlantFlex', 'LeftStaticRotOff', 'LeftStaticPlantFlex'])
    def calc_static_ankle_offsets(r_foot_axis_uncorrect, r_foot_axis, l_foot_axis_uncorrect, l_foot_axis):
        """Static angle calculation function.

        Takes in two axes and returns the rotation, flexion,
        and abduction angles in degrees.
        Uses the inverse Euler rotation matrix in YXZ order.

        Since we use arcsin we must check if the angle is in area between -pi/2 and pi/2
        but because the static offset angle is less than pi/2, it doesn't matter.

        Parameters
        ----------
        r_foot_axis_uncorrect : array
            4x4 affine matrix representing the position of the proximal axis.
        r_foot_axis : array
            4x4 affine matrix representing the position of the distal axis.

        Returns
        -------
        angle : array
            1x3 array representing the rotation, flexion, and abduction angles in degrees

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgmStatic import calc_static_angle_ankle
        >>> r_foot_axis_uncorrect = np.array([[ 0.59,  0.11,  0.16, 0],
        ...                    [-0.13, -0.10, -0.90, 0],
        ...                    [ 0.94, -0.05,  0.75, 0],
        ...                    [ 0,     0,     0,    0]])
        >>> r_foot_axis = np.array([[ 0.17,  0.69, -0.37, 0],
        ...                    [ 0.14, -0.39,  0.94, 0],
        ...                    [-0.16, -0.53, -0.60, 0],
        ...                    [ 0,     0,     0,    0]])
        >>> np.around(calc_static_angle_ankle(r_foot_axis_uncorrect, r_foot_axis), 2)
        array([0.48, 1.  , 1.56])
        """
        # make inverse matrix of r_foot_axis_uncorrect
        r_foot_axis_uncorrect = np.asarray(r_foot_axis_uncorrect)
        r_foot_axis = np.asarray(r_foot_axis)

        r_foot_axis_uncorrect = r_foot_axis_uncorrect[:, :, :3]
        r_foot_axis = r_foot_axis[:, :, :3]

        r_foot_axis_uncorrect_inv = np.linalg.inv(r_foot_axis_uncorrect)

        # M is multiply of r_foot_axis and r_foot_axis_uncorrect_inv
        r_foot_axis = np.transpose(r_foot_axis, axes=(0,2,1))
        r_foot_axis_uncorrect_inv = np.transpose(r_foot_axis_uncorrect_inv, axes=(0,2,1))
        M = np.matmul(r_foot_axis, r_foot_axis_uncorrect_inv)

        # This is the angle calculation in YXZ Euler angle
        a = np.divide(M[:, 2, 1], np.sqrt((M[:, 2, 0] * M[:, 2, 0]) + (M[:, 2, 2] * M[:, 2, 2])))
        b = -1 * M[:, 2, 0] / M[:, 2, 2]
        g = -1 * M[:, 0, 1] / M[:, 1, 1]

        gamma =np.arctan(g)
        alpha = np.arctan(a)
        beta = np.arctan(b)

        right_static_rot_off    = np.average(alpha) * -1
        right_static_plant_flex = np.average(beta)

        l_foot_axis_uncorrect = np.asarray(l_foot_axis_uncorrect)
        l_foot_axis = np.asarray(l_foot_axis)

        l_foot_axis_uncorrect = l_foot_axis_uncorrect[:, :, :3]
        l_foot_axis = l_foot_axis[:, :, :3]

        l_foot_axis_uncorrect_inv = np.linalg.inv(l_foot_axis_uncorrect)

        # M is multiply of l_foot_axis and l_foot_axis_uncorrect_inv
        l_foot_axis = np.transpose(l_foot_axis, axes=(0,2,1))
        l_foot_axis_uncorrect_inv = np.transpose(l_foot_axis_uncorrect_inv, axes=(0,2,1))
        M = np.matmul(l_foot_axis, l_foot_axis_uncorrect_inv)

        # This is the angle calculation in YXZ Euler angle
        a = np.divide(M[:, 2, 1], np.sqrt((M[:, 2, 0] * M[:, 2, 0]) + (M[:, 2, 2] * M[:, 2, 2])))
        b = -1 * M[:, 2, 0] / M[:, 2, 2]
        g = -1 * M[:, 0, 1] / M[:, 1, 1]

        gamma =np.arctan(g)
        alpha = np.arctan(a)
        beta = np.arctan(b)

        left_static_rot_off     = np.average(alpha)
        left_static_plant_flex  = np.average(beta)

        return np.array([right_static_rot_off, right_static_plant_flex, left_static_rot_off, left_static_plant_flex])
