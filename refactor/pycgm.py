from refactor.io import IO
from math import cos, sin, acos
import numpy as np


class CGM:

    def __init__(self, path_static, path_dynamic, path_measurements, path_results=None,
                 write_axes=True, write_angles=True, write_com=True,
                 static=None, cores=1):
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
        path_results : str, optional
            File path of the output file in csv or c3d form
        write_axes : bool or list, optional
            Boolean option to enable or disable writing of axis results to output file, or list
            of axis names to write
        write_angles : bool or list, optional
            Boolean option to enable or disable writing of angle results to output file, or list
            of angle names to write
        write_com : bool, optional
            Boolean option to enable or disable writing of center of mass results to output file

        Examples
        --------
        >>> from .pycgm import CGM
        >>> dir = "SampleData/59993_Frame/"
        >>> static_trial = dir + "59993_Frame_Static.c3d"
        >>> dynamic_trial = dir + "59993_Frame_Dynamic.c3d"
        >>> measurements = dir + "59993_Frame_SM.vsk"
        >>> subject1 = CGM(static_trial, dynamic_trial, measurements)
        SampleData/59993_Frame/59993_Frame_Dynamic.c3d
        >>> subject1.marker_data[0][0]  # doctest: +NORMALIZE_WHITESPACE
        array([  54.67363358,  156.26828003, 1474.328125  ])
        """
        self.path_static = path_static
        self.path_dynamic = path_dynamic
        self.path_measurements = path_measurements
        self.path_results = path_results
        self.write_axes = write_axes
        self.write_angles = write_angles
        self.write_com = write_com
        self.static = static if static else StaticCGM(path_static, path_measurements)
        self.cores = cores
        self.angle_results = None
        self.axis_results = None
        self.com_results = None
        self.marker_map = {marker: marker for marker in IO.marker_keys()}
        self.marker_data, self.marker_idx = IO.load_marker_data(path_dynamic)

    def run(self):
        """Execute the CGM calculations function

        Loads in appropriate data from IO using paths.
        Performs any necessary prep on data.
        Runs the static calibration trial.
        Runs the dynamic trial to calculate all axes and angles.
        """
        pass

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

    @staticmethod
    def find_joint_center(a, b, c, delta):
        """Calculate the Joint Center function.

        This function is based on physical markers, a, b, and c, and joint center, which will be
        calculated in this function. All are in the same plane.

        Parameters
        ----------
        a, b, c : array
            A 1x3 ndarray representing x, y, and z coordinates of the marker.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        mr : array
            Returns the Joint Center x, y, z positions in a 1x3 ndarray.
        """
        # Make the two vector using 3 markers, which is on the same plane.
        v1 = (a[0] - c[0], a[1] - c[1], a[2] - c[2])
        v2 = (b[0] - c[0], b[1] - c[1], b[2] - c[2])

        # v3 is cross vector of v1, v2
        # and then it is normalized.
        # v3 = cross(v1,v2)
        v3 = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]
        v3_div = np.linalg.norm(v3)
        v3 = [v3[0] / v3_div, v3[1] / v3_div, v3[2] / v3_div]

        m = [(b[0] + c[0]) / 2.0, (b[1] + c[1]) / 2.0, (b[2] + c[2]) / 2.0]
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

        r = [r[0, 0], r[1, 0], r[2, 0]]
        mr = np.array([r[0] + m[0], r[1] + m[1], r[2] + m[2]])

        return mr

    @staticmethod
    def rotation_matrix(x=0, y=0, z=0):
        """Rotation Matrix function

        This function creates and returns a rotation matrix.

        Parameters
        ----------
        x, y, z : float, optional
            Angle, which will be converted to radians, in
            each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.

        Returns
        -------
        rxyz : list
            The product of the matrix multiplication as a 3x3 ndarray.
        """

    @staticmethod
    def wand_marker(rsho, lsho, thorax_axis):
        """Wand Marker Calculation function

        Takes in a dictionary of x,y,z positions and marker names.
        and takes the thorax axis.
        Calculates the wand marker for calculating the clavicle.

        Markers used: RSHO, LSHO

        Parameters
        ----------
        rsho, lsho : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_axis : array
            A 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.

        Returns
        -------
        wand : array
            Returns a 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.
        """

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
        rasi, lasi : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        sacr, rpsi, lpsi : array, optional
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
        >>> from .pycgm import CGM
        >>> rasi, lasi, rpsi, lpsi = np.array([[ 395.36532593,  428.09790039, 1036.82763672],
        ...                                    [ 183.18504333,  422.78927612, 1033.07299805],
        ...                                    [ 341.41815186,  246.72117615, 1055.99145508],
        ...                                    [ 255.79994202,  241.42199707, 1057.30065918]])
        >>> CGM.pelvis_axis_calc(rasi, lasi, rpsi=rpsi, lpsi=lpsi)
        array([[ 289.27518463,  425.44358826, 1034.95031739],
               [ 289.25243803,  426.43632163, 1034.8321521 ],
               [ 288.27565385,  425.41858059, 1034.93263018],
               [ 289.25467091,  425.56129577, 1035.94315379]])
        >>> rasi, lasi, sacr = np.array([[ 395.36532593,  428.09790039, 1036.82763672],
        ...                              [ 183.18504333,  422.78927612, 1033.07299805],
        ...                              [ 294.60904694,  242.07158661, 1049.64605713]])
        >>> CGM.pelvis_axis_calc(rasi, lasi, sacr=sacr)
        array([[ 289.27518463,  425.44358826, 1034.95031739],
               [ 289.25166321,  426.44012508, 1034.87056085],
               [ 288.27565385,  425.41858059, 1034.93263018],
               [ 289.25556415,  425.52289134, 1035.94697483]])
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

        # Z-axis is cross product of x_axis and y_axis
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
        >>> from .pycgm import CGM
        >>> pelvis_axis = np.array([[ 251.60830688, 391.74131775, 1032.89349365],
        ...                         [ 251.74063624, 392.72694721, 1032.78850073],
        ...                         [ 250.61711554, 391.87232862, 1032.8741063 ],
        ...                         [ 251.60295336, 391.84795134, 1033.88777762]])
        >>> measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
        ...                 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
        >>> CGM.hip_axis_calc(pelvis_axis, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[245.47574075, 331.11787116, 936.75939614],
               [245.60807011, 332.10350062, 936.65440322],
               [244.48454941, 331.24888203, 936.74000879],
               [245.47038723, 331.22450475, 937.75368011]])
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

        return np.array([hip_axis_center, x_axis, y_axis, z_axis])

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements):
        """Knee Axis Calculation function

        Calculates the right and left knee joint center and axis and returns them.

        Markers used: RTHI, LTHI, RKNE, LKNE
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        hip_origin : array
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
        >>> from .pycgm import CGM
        >>> rthi, lthi, rkne, lkne = np.array([[426.50338745, 262.65310669, 673.66247559],
        ...                                    [51.93867874 , 320.01849365, 723.03186035],
        ...                                    [416.98687744, 266.22558594, 524.04089355],
        ...                                    [84.62355804 , 286.69122314, 529.39819336]])
        >>> hip_origin = np.array([[309.38050472, 32280342417, 937.98979061],
        ...                        [182.57097863, 339.43231855, 935.52900126]])
        >>> measurements = {'RightKneeWidth': 105.0, 'LeftKneeWidth': 105.0 }
        >>> CGM.knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[413.21007973, 266.22558784, 464.66088466],
               [414.20806312, 266.22558785, 464.59740907],
               [413.14660414, 266.22558786, 463.66290127],
               [413.21007973, 267.22558784, 464.66088468],
               [143.55478579, 279.90370346, 524.78408753],
               [143.65611281, 280.88685896, 524.63197541],
               [142.56434499, 280.01777942, 524.86163553],
               [143.64837987, 280.0465038 , 525.76940383]])
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

        # Determine the position of kneeJointCenter using findJointC function
        r_knee_jc = CGM.find_joint_center(rthi, r_hip_jc, rkne, r_delta)
        l_knee_jc = CGM.find_joint_center(lthi, l_hip_jc, lkne, l_delta)

        # Right axis calculation

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = r_hip_jc - r_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        axis_x = np.cross(axis_z, rkne - r_hip_jc)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        r_axis = np.array([axis_x, axis_y, axis_z])

        # Left axis calculation

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = l_hip_jc - l_knee_jc

        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector.
        # the axis is then normalized.
        axis_x = np.cross(lkne - l_hip_jc, axis_z)

        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = np.cross(axis_z, axis_x)

        l_axis = np.array([axis_x, axis_y, axis_z])

        # Clear the name of axis and then nomalize it.
        r_knee_x_axis = r_axis[0]
        r_knee_x_axis = r_knee_x_axis / np.array([np.linalg.norm(r_knee_x_axis)])
        r_knee_y_axis = r_axis[1]
        r_knee_y_axis = r_knee_y_axis / np.array([np.linalg.norm(r_knee_y_axis)])
        r_knee_z_axis = r_axis[2]
        r_knee_z_axis = r_knee_z_axis / np.array([np.linalg.norm(r_knee_z_axis)])
        l_knee_x_axis = l_axis[0]
        l_knee_x_axis = l_knee_x_axis / np.array([np.linalg.norm(l_knee_x_axis)])
        l_knee_y_axis = l_axis[1]
        l_knee_y_axis = l_knee_y_axis / np.array([np.linalg.norm(l_knee_y_axis)])
        l_knee_z_axis = l_axis[2]
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
        rtib, ltib, rank, lank : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        knee_origin : array
            A 2x3 ndarray of the right and left knee origin vectors (joint centers).
        measurements : dict, optional
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
        """

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
        rtoe, ltoe : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : array
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
        """

    @staticmethod
    def head_axis_calc(lfhd, rfhd, lbhd, rbhd, measurements):
        """Head Axis Calculation function

        Calculates the head joint center and axis and returns them.

        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        lfhd, rfhd, lbhd, rbhd : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the head origin and the
            head x, y, and z axis components.
        """

    @staticmethod
    def thorax_axis_calc(clav, c7, strn, t10):
        """Thorax Axis Calculation function

        Calculates the thorax joint center and axis and returns them.

        Markers used: CLAV, C7, STRN, T10

        Parameters
        ----------
        clav, c7, strn, t10 : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.
        """

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
        thorax_origin : array
            A 1x3 ndarray of the thorax origin vector (joint center).
        wand : array
            A 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right shoulder origin, right shoulder x, y, and z
            axis components, left shoulder origin, and left shoulder x, y, and z axis components.
        """

    @staticmethod
    def elbow_wrist_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb,
                              thorax_axis, shoulder_origin, wand, measurements):
        """Elbow and Wrist Axis Calculation function

        Calculates the right and left elbow joint center and axis, and the
        right and left wrist just center and axis, and returns them.

        Markers used: RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB
        Subject Measurement values used: RightElbowWidth, LeftElbowWidth

        Parameters
        ----------
        rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        thorax_axis : array
            A 4x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.
        shoulder_origin : array
            A 2x3 ndarray of the right and left shoulder origin vectors (joint centers).
        wand : array
            A 2x3 ndarray containing the right wand marker x, y, and z positions and the
            left wand marker x, y, and z positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 2x8x3 ndarray, where the first index contains the right elbow origin, right elbow x, y, and z
            axis components, left elbow origin, and left elbow x, y, and z axis components, and the second index
            contains the right wrist origin, right wrist x, y, and z axis components, left wrist origin, and left
            wrist x, y, and z axis components.
        """

    # @staticmethod
    # def wrist_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, elbow_axis, wand):
    #     """Wrist Axis Calculation function
    #
    #     Calculates the right and left wrist joint center and axis and returns them.
    #
    #     Markers used: RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB
    #
    #     Parameters
    #     ----------
    #     rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb : array
    #         A 1x3 ndarray of each respective marker containing the XYZ positions.
    #     elbow_axis : array
    #         An 8x3 ndarray that contains the right elbow origin, right elbow x, y, and z
    #         axis components, left elbow origin, and left elbow x, y, and z axis components.
    #     wand : array
    #         A 2x3 ndarray containing the right wand marker x, y, and z positions and the
    #         left wand marker x, y, and z positions.
    #
    #     Returns
    #     --------
    #     array
    #         Returns an 8x3 ndarray that contains the right wrist origin, right wrist x, y, and z
    #         axis components, left wrist origin, and left wrist x, y, and z axis components.
    #     """

    @staticmethod
    def hand_axis_calc(rwra, wrb, lwra, lwrb, rfin, lfin, wrist_jc, measurements):
        """Hand Axis Calculation function

        Calculates the right and left hand joint center and axis and returns them.

        Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN
        Subject Measurement values used: RightHandThickness, LeftHandThickness

        Parameters
        ----------
        rwra, wrb, lwra, lwrb, rfin, lfin : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        wrist_jc : array
            A 2x3 array containing the x,y,z position of the right and left wrist joint center.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x3 ndarray that contains the right hand origin, right hand x, y, and z
            axis components, left hand origin, and left hand x, y, and z axis components.
        """

    @staticmethod
    def pelvis_angle_calc():
        pass

    @staticmethod
    def hip_angle_calc():
        pass

    @staticmethod
    def knee_angle_calc():
        pass

    @staticmethod
    def ankle_angle_calc():
        pass

    @staticmethod
    def foot_angle_calc():
        pass

    @staticmethod
    def head_angle_calc():
        pass

    @staticmethod
    def thorax_angle_calc():
        pass

    @staticmethod
    def spine_angle_calc():
        pass

    @staticmethod
    def neck_angle_calc():
        pass

    @staticmethod
    def shoulder_angle_calc():
        pass

    @staticmethod
    def elbow_angle_calc():
        pass

    @staticmethod
    def wrist_angle_calc():
        pass

    @staticmethod
    def multi_calc(data, methods, mappings, measurements, cores=1):
        """Multiprocessing calculation handler function

        Takes in the necessary information for performing each frame's calculation as parameters
        and distributes frames along available cores.

        Parameters
        ----------
        data : array
            3d ndarray consisting of each frame by each marker by x, y, and z positions.
        methods : list
            List containing the calculation methods to be used.
        mappings : list
            List containing dictionary mappings for marker names and input and output indices.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
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
        pass

    @staticmethod
    def calc(data, methods, mappings, measurements):
        """Overall axis and angle calculation function

        Uses the data and methods passed in to distribute the appropriate inputs to each
        axis and angle calculation function (generally markers and axis results) and
        store and return their output, all in the context of a single frame.

        Parameters
        ----------
        data : array
            3d ndarray consisting of each frame by each marker by x, y, and z positions.
        methods : list
            List containing the calculation methods to be used.
        mappings : list
            List containing dictionary mappings for marker names and input and output indices.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        results : tuple
            A tuple consisting of the angle results and axis results. Angle results are stored
            as a 2d ndarray of each angle by x, y, and z. Axis results are stored as a 3d ndarray
            of each joint by origin and xyz unit vectors by x, y, and z location.
        """
        pass


class StaticCGM:

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
        pass

    @staticmethod
    def rotation_matrix(x=0, y=0, z=0):
        """Rotation Matrix function

        This function creates and returns a rotation matrix.

        Parameters
        ----------
        x, y, z : float, optional
            Angle, which will be converted to radians, in
            each respective axis to describe the rotations.
            The default is 0 for each unspecified angle.

        Returns
        -------
        rxyz : list
            The product of the matrix multiplication.
        """

    @staticmethod
    def get_dist(p0, p1):
        """Get Distance function

        This function calculates the distance between two 3-D positions.

        Parameters
        ----------
        p0 : array
            Position of first x,y,z coordinate.
        p1 : array
            Position of second x,y,z coordinate.
        Returns
        -------
        float
            The distance between positions p0 and p1.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import StaticCGM
        >>> p0 = [0,1,2]
        >>> p1 = [1,2,3]
        >>> np.around(StaticCGM.getDist(p0,p1),8)
        1.73205081
        >>> p0 = np.array([991.44611381, 741.95103792, 321.35500969])
        >>> p1 = np.array([117.08710839, 142.23917057, 481.95268411])
        >>> np.around(StaticCGM.getDist(p0,p1),8)
        1072.35703347
        """

    @staticmethod
    def average(list):
        """The Average Calculation function

        Calculates the average of the values in a given list or array.

        Parameters
        ----------
        list : list
            List or array of values.

        Returns
        -------
        float
            The mean of the list.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import StaticCGM
        >>> list = [1,2,3,4,5]
        >>> StaticCGM.average(list)
        3.0
        >>> list = np.array([93.81607046, 248.95632028, 782.61762769])
        >>> np.around(StaticCGM.average(list),8)
        375.13000614
        """

    @staticmethod
    def find_joint_c(a, b, c, delta):
        """Calculate the Joint Center function.

        This function is based on physical markers, a,b,c and joint center which will be
        calulcated in this function are all in the same plane.

        Parameters
        ----------
        a,b,c : list
            Three markers x,y,z position of a, b, c.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        mr : array
            Returns the Joint C x, y, z positions in a 1x3 array.
        """

    @staticmethod
    def get_static(motion_data, ):
        """ Get Static Offset function
        
        Calculate the static offset angle values and return the values in radians

        Parameters
        ----------
        motionData : dict
            Dictionary of marker lists.
        measurements : dict, optional
            A dictionary containing the subject measurements given from the file input.
        flat_foot : boolean, optional
            A boolean indicating if the feet are flat or not.
            The default value is False.
        GCS : array, optional
            An array containing the Global Coordinate System.
            If not provided, the default will be set to: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
        
        Returns
        -------
        calSM : dict
            Dictionary containing various marker lists of offsets.
        """

    @staticmethod
    def iad_calculation():
        pass

    @staticmethod
    def static_calculation_head():
        pass

    @staticmethod
    def static_calculation(rtoe, ltoe, rhee, lhee, ankle_JC, knee_JC, flat_foot, measurements):
        """The Static Angle Calculation function
        
        Takes in anatomical uncorrect axis and anatomical correct axis. 
        Correct axis depends on foot flat options.

        Calculates the offset angle between that two axis.

        It is rotated from uncorrect axis in YXZ order.
        
        Parameters
        ----------
        rtoe, ltoe, rhee, lhee : dict 
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_JC : array
            An ndarray containing the x,y,z axes marker positions of the ankle joint centers. 
        knee_JC : array
            An ndarray containing the x,y,z axes marker positions of the knee joint centers. 
        flat_foot : boolean
            A boolean indicating if the feet are flat or not.
        measurements : dict, optional
            A dictionary containing the subject measurements given from the file input.
                
        Returns
        ------- 
        angle : list
            Returns the offset angle represented by a 2x3x3 array. 
            The array contains the right flexion, abduction, rotation angles (1x3x3)
            followed by the left flexion, abduction, rotation angles (1x3x3).

        Modifies
        --------
        The correct axis changes following to the foot flat option. 
        
        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import StaticCGM
        >>> rtoe = np.array([427.95211792, 437.99603271,  41.77342987])
        >>> ltoe = np.array([175.78988647, 379.49987793,  42.61193085])
        >>> rhee = np.array([406.46331787, 227.56491089,  48.75952911])
        >>> lhee = np.array([223.59848022, 173.42980957,  47.92973328])
        >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
        ...            np.array([98.74901939, 219.46930221, 80.6306816]),
        ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
        ...            np.array([393.07114384, 248.39110006, 87.61575574]),
        ...            np.array([393.69314056, 247.78157916, 88.73002876])],
        ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
        ...            np.array([97.79246671, 219.20927275, 80.76255901]),
        ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
        >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
        ...           np.array([143.55478579, 279.90370346, 524.78408753]),
        ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
        ...           [363.29019771, 292.60656648, 515.04309095],
        ...           [364.04724541, 292.24216264, 516.18067112]],
        ...           [[143.65611282, 280.88685896, 524.63197541],
        ...           [142.56434499, 280.01777943, 524.86163553],
        ...           [143.64837987, 280.04650381, 525.76940383]]])]
        >>> flat_foot = True      
        >>> measurements = { 'RightSoleDelta': 0.4532,'LeftSoleDelta': 0.4532 }
        >>> np.around(StaticCGM.staticCalculation(rtoe, ltoe, rhee, lhee, ankle_JC, knee_JC, flat_foot, measurements),8)
        array([[-0.08036968,  0.23192796, -0.66672181],
            [-0.67466613,  0.21812578, -0.30207993]])
        >>> flat_foot = False # Using the same variables and switching the flat_foot flag. 
        >>> np.around(StaticCGM.staticCalculation(rtoe, ltoe, rhee, lhee, ankle_JC, knee_JC, flat_foot, measurements),8)
        array([[-0.07971346,  0.19881323, -0.15319313],
            [-0.67470483,  0.18594096,  0.12287455]])
        """

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
        rasi, lasi : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        rpsi, lpsi, sacr : array, optional
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
        """

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
        pelvis_axis : array
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
        """

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_origin, delta, measurements):
        """Knee Axis Calculation function

        Calculates the right and left knee joint center and axis and returns them.

        Markers used: RTHI, LTHI, RKNE, LKNE
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        hip_origin : array
            A 2x3 ndarray of the right and left hip origin vectors (joint centers).
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
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
        """

    @staticmethod
    def ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, delta, measurements):
        """Ankle Axis Calculation function

        Calculates the right and left ankle joint center and axis and returns them.

        Markers used: RTIB, LTIB, RANK, LANK
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

        Parameters
        ----------
        rtib, ltib, rank, lank : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        knee_origin : array
            A 2x3 ndarray of the right and left knee origin vectors (joint centers).
        delta : float
            The length from marker to joint center, retrieved from subject measurement file
        measurements : dict, optional
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
        """

    @staticmethod
    def foot_axis_calc(rtoe, ltoe, ankle_axis, delta, measurements):
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
        rtoe, ltoe : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_axis : array
            An 8x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
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
        """

    @staticmethod
    def head_axis_calc(lfhd, rfhd, lbhd, rbhd, measurements):
        """Head Axis Calculation function

        Calculates the head joint center and axis and returns them.

        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        lfhd, rfhd, lbhd, rbhd : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x3 ndarray that contains the head origin and the
            head x, y, and z axis components.
        """

    @staticmethod
    def ankle_angle_calc(axis_p, axis_d):
        """Static angle calculation function.
        
        This function takes in two axis and returns three angles.
        and It use inverse Euler rotation matrix in YXZ order.
        the output shows the angle in degrees.
        
        As we use arc sin we have to care about if the angle is in area between -pi/2 to pi/2
        but in case of calculate static offset angle it is in boundry under pi/2, it doesn't matter.
        
        Parameters
        ----------
        axis_p : list
            Shows the unit vector of axisP, the position of the proximal axis.
        axis_d : list
            Shows the unit vector of axisD, the position of the distal axis.
            
        Returns
        -------
        angle : list
            Returns the gamma, beta, alpha angles in degrees in a 1x3 corresponding list. 
        
        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import StaticCGM
        >>> axis_p = [[ 0.59327576, 0.10572786, 0.15773334],
        ...         [-0.13176004, -0.10067464, -0.90325703],
        ...         [0.9399765, -0.04907387, 0.75029827]]
        >>> axis_d = [[0.16701015, 0.69080381, -0.37358145],
        ...         [0.1433922, -0.3923507, 0.94383974],
        ...         [-0.15507695, -0.5313784, -0.60119402]]
        >>> np.around(StaticCGM.getankleangle(axis_p,axis_d),8)
        array([0.47919763, 0.99019921, 1.51695461])
        """
        pass
