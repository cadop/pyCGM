from refactor.io import IO
from math import cos, sin
import numpy as np
import os
import sys

if sys.version_info[0]==2:
    pyver = 2
else:
    pyver = 3

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
        self.measurements = IO.load_sm(path_measurements)

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
    def subtract_origin(axis_vectors):
        """Subtract origin from axis vectors.

        Parameters
        ----------
        axis_vectors : array
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
        >>> from refactor.pycgm import CGM
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
        origin, x_axis, y_axis, z_axis = axis_vectors
        return np.vstack([np.subtract(x_axis, origin),
                          np.subtract(y_axis, origin),
                          np.subtract(z_axis, origin)])

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

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
        ...                 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
        >>> pelvis_axis = np.array([[ 251.60830688, 391.74131775, 1032.89349365],
        ...                         [ 251.74063624, 392.72694721, 1032.78850073],
        ...                         [ 250.61711554, 391.87232862, 1032.8741063 ],
        ...                         [ 251.60295336, 391.84795134, 1033.88777762]])
        >>> CGM.hip_axis_calc(pelvis_axis, measurements) #doctest: +NORMALIZE_WHITESPACE
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

    #Center of Mass / kinetics calculation Methods:
    @staticmethod
    def point_to_line(point, start, end):
        """Finds the distance from a point to a line.

        Calculates the distance from the point `point` to the line formed
        by the points `start` and `end`.

        Parameters
        ----------
        point, start, end : array
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
        >>> from refactor.pycgm import CGM
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
        point_vector_scaled = point_vector * (1.0/line_length)
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
        lhjc, rhjc : array
            1x3 ndarray giving the XYZ coordinates of the LHJC and RHJC
            markers respectively.
        axis : array
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
        >>> from refactor.pycgm import CGM
        >>> lhjc = np.array([308.38050472, 322.80342417, 937.98979061])
        >>> rhjc = np.array([182.57097863, 339.43231855, 935.529000126])
        >>> axis = np.array([[251.60830688, 391.74131775, 1032.89349365],
        ...                  [251.74063624, 392.72694721, 1032.78850073],
        ...                  [250.61711554, 391.87232862, 1032.8741063 ],
        ...                  [251.60295336, 391.84795134, 1033.88777762]])
        >>> np.around(CGM.find_l5(lhjc, rhjc, axis), 8)
        array([[ 245.47574168,  331.11787136,  936.75939537],
               [ 271.52716019,  371.69050709, 1043.80997977]])
        """
        #The L5 position is estimated as (LHJC + RHJC)/2 + 
        #(0.0, 0.0, 0.828) * Length(LHJC - RHJC), where the value 0.828 
        #is a ratio of the distance from the hip joint centre level to the 
        #top of the lumbar 5: this is calculated as in the vertical (z) axis
        mid_hip = (lhjc + rhjc) / 2

        offset = np.linalg.norm(lhjc - rhjc) * 0.925
        origin, x_axis, y_axis, z_axis = axis
        norm_dir = z_axis / np.linalg.norm(z_axis) #Create unit vector
        l5 = mid_hip + offset * norm_dir

        return mid_hip, l5
    
    @staticmethod
    def get_kinetics(joint_centers, jc_mapping, body_mass):
        """Estimate center of mass values in the global coordinate system.

        Estimates whole body CoM in global coordinate system using PiG scaling 
        factors for determining individual segment CoM.

        Parameters
        ----------
        joint_centers : 3darray
            3D numpy array where each index corresponds to a frame of trial.
            Each index contains an array of joint center or marker values that
            are used to estimate the center of mass for that frame. Each value
            is a 1x3 array indicating the XYZ coordinate of that marker or joint
            center.
        jc_mapping : dict
            Dictionary where keys are joint center or marker names, and values are
            indices that indicate which index in `joint_centers` correspond to that
            joint center or marker name.
        body_mass : int, float
            Total bodymass (kg) of the subject.
        
        Returns
        -------
        com_coords : 2darray
            Numpy array containing center of mass coordinates for each frame of
            trial. Each coordinate is a 1x3 array of the XYZ position of the center
            of mass.
        
        Notes
        -----
        The PiG scaling factors are taken from Dempster -- they are available at:
        http://www.c-motion.com/download/IORGaitFiles/pigmanualver1.pdf
        """

        #Get PlugInGait scaling table from segments.csv
        seg_scale = {}
        with open(os.path.dirname(os.path.abspath(__file__)) + os.sep +'segments.csv','r') as f:
            header = False
            for line in f:
                if header == False:
                    header = line.rstrip('\n').split(',')
                    header = True
                else:
                    row = line.rstrip('\n').split(',')
                    seg_scale[row[0]] = {'com':float(row[1]),'mass':float(row[2]),'x':row[3],'y':row[4],'z':row[5]}

        #Define names of segments
        sides = ['L', 'R']
        segments = ['Foot','Tibia','Femur','Pelvis','Radius','Hand','Humerus','Head','Thorax']

        #Create empty numpy array for center of mass outputs
        com_coords = np.empty([len(joint_centers), 3])

        #Iterate through each frame of joint_centers
        for idx, frame in enumerate(joint_centers):

            #Find distal and proximal joint centers
            seg_temp = {}
            for s in sides:
                for seg in segments:
                    if seg != 'Pelvis' and seg != 'Thorax' and seg != 'Head':
                        seg_temp[s+seg] = {}
                    else:
                        seg_temp[seg] = {}

                    if seg == 'Foot':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Foot']]
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'HEE']]

                    if seg == 'Tibia':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Knee']]
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'Ankle']]

                    if seg == 'Femur':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Hip']]
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'Knee']]

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
                        #The thorax length is taken as the distance between an 
                        #approximation to the C7 vertebra and the L5 vertebra in the 
                        #Thorax reference frame. C7 is estimated from the C7 marker, 
                        #and offset by half a marker diameter in the direction of 
                        #the X axis. L5 is estimated from the L5 provided from the 
                        #pelvis segment, but localised to the thorax.

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

                        upper = np.array([(clav[0]+c7[0])/2.0,(clav[1]+c7[1])/2.0,(clav[2]+c7[2])/2.0])
                        lower = np.array([(strn[0]+t10[0])/2.0,(strn[1]+t10[1])/2.0,(strn[2]+t10[2])/2.0])

                        #Get the direction of the primary axis Z (facing down)
                        z_vec = upper - lower
                        z_dir = z_vec / np.linalg.norm(z_vec)
                        new_start = upper + (z_dir * 300)
                        new_end = lower - (z_dir * 300)

                        _,new_l5,_ = CGM.point_to_line(l5, new_start, new_end)
                        _,new_c7,_ = CGM.point_to_line(c7, new_start, new_end)

                        seg_temp[seg]['Prox'] = np.array(new_c7)
                        seg_temp[seg]['Dist'] = np.array(new_l5)

                    if seg == 'Humerus':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Shoulder']] 
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'Humerus']]

                    if seg == 'Radius':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Humerus']] 
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'Radius']] 
                        
                    if seg == 'Hand':
                        seg_temp[s+seg]['Prox'] = frame[jc_mapping[s+'Radius']]  
                        seg_temp[s+seg]['Dist'] = frame[jc_mapping[s+'Hand']] 

                    if seg == 'Head':
                        seg_temp[seg]['Prox'] = frame[jc_mapping['Back_Head']]
                        seg_temp[seg]['Dist'] = frame[jc_mapping['Front_Head']]
                    
                    #Iterate through scaling values
                    for row in list(seg_scale.keys()):
                        scale = seg_scale[row]['com']
                        mass = seg_scale[row]['mass']
                        if seg == row:
                            if seg!='Pelvis' and seg!='Thorax' and seg!='Head':
                                prox = seg_temp[s+seg]['Prox']
                                dist = seg_temp[s+seg]['Dist']
                            
                                #segment length
                                length = prox - dist
                                
                                #segment center of mass
                                com = dist + length * scale

                                seg_temp[s+seg]['CoM'] = com

                                #segment mass (kg)
                                mass = body_mass * mass #row[2] contains mass corrections
                                seg_temp[s+seg]['Mass'] = mass

                                #segment torque
                                torque = com * mass
                                seg_temp[s+seg]['Torque'] = torque

                                #vector
                                vector = np.array(com) - np.array([0, 0, 0])
                                val = vector * mass
                                seg_temp[s+seg]['val'] = val
                            
                            #no side allocation
                            else:
                                prox = seg_temp[seg]['Prox']
                                dist = seg_temp[seg]['Dist']
                                
                                #segment length
                                length = prox - dist
                                
                                #segment CoM
                                com = dist + length * scale
                                
                                seg_temp[seg]['CoM'] = com
                                
                                #segment mass (kg)
                                mass = body_mass*mass #row[2] is mass correction factor
                                seg_temp[seg]['Mass'] = mass
                                
                                #segment torque
                                torque = com * mass
                                seg_temp[seg]['Torque'] = torque
                                
                                #vector
                                vector = np.array(com) - np.array([0, 0, 0])
                                val = vector*mass
                                seg_temp[seg]['val'] = val
                    
                    vals = []

                    if pyver == 2:
                        for_iter = seg_temp.iteritems()
                    elif pyver == 3:
                        for_iter = seg_temp.items()
                    
                    for attr, value in for_iter:
                        vals.append(value['val'])
                    
                    com_coords[idx,:] = sum(vals) / body_mass

        return com_coords

                

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
    def iad_calculation(rasi, lasi):
        """Calculates the Inter ASIS Distance.

        Given the markers RASI and LASI, the Inter ASIS Distance is defined as:
        .. math::
            InterASISDist = \sqrt{(RASI_x-LASI_x)^2 + (RASI_y-LASI_y)^2 + (RASI_z-LASI_z)^2}
        where :math:`RASI_x` is the x-coordinate of the RASI marker in frame.

        Markers used: RASI, LASI

        Parameters
        ----------
        rasi, lasi : array
            A 1x3 ndarray of each respective marker containing the XYZ positions.

        Returns
        -------
        iad : float
            The Inter ASIS distance as a float.
        
        Examples
        --------
        >>> from numpy import around, array
        >>> from refactor import pycgm
        >>> lasi = array([ 183.18504333,  422.78927612, 1033.07299805])
        >>> rasi = array([ 395.36532593,  428.09790039, 1036.82763672])
        >>> around(pycgm.StaticCGM.iad_calculation(rasi, lasi), 2)
        212.28
        """
        x_diff = rasi[0] - lasi[0]
        y_diff = rasi[1] - lasi[1]
        z_diff = rasi[2] - lasi[2]
        iad = np.sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff)
        return iad

    @staticmethod
    def static_calculation_head(head_axis):
        """Calculates the offset angle of the head.

        Uses the x,y,z axes of the head and the head origin to calculate
        the head offset angle. Uses the global axis.

        Parameters
        ----------
        head : array
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
        >>> from refactor import pycgm
        >>> head = array([[99.58366584777832, 82.79330825805664, 1483.7968139648438],
        ...               [100.33272997128863, 83.39303060995121, 1484.078302933558], 
        ...               [98.9655145897623, 83.57884461044797, 1483.7681493301013], 
        ...               [99.34535520789223, 82.64077714742746, 1484.7559501904173]])
        >>> around(pycgm.StaticCGM.static_calculation_head(head), 8)
        0.28546606
        """
        head_axis = CGM.subtract_origin(head_axis)
        global_axis = [[0,1,0],[-1,0,0],[0,0,1]]

        #Global axis is the proximal axis
        #Head axis is the distal axis
        axis_p = global_axis
        axis_d = head_axis

        axis_p_inverse = np.linalg.inv(axis_p)
        rotation_matrix = np.matmul(axis_d, axis_p_inverse)
        offset = np.arctan(rotation_matrix[0][2]/rotation_matrix[2][2])
        
        return offset

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
