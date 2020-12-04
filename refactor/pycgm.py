from refactor.io import IO
from math import cos, sin, acos, degrees, radians, pi
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
        SampleData/59993_Frame/59993_Frame_Static.c3d
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

    # Utility functions
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

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> a, b, c = np.array([[468.14532471, 325.09780884, 673.12591553],
        ...                     [355.90861996, 365.38260964, 940.6974861 ],
        ...                     [452.35180664, 329.0609436 , 524.77893066]])
        >>> delta = 59.5
        >>> CGM.find_joint_center(a, b, c, delta)
        array([396.25286248, 347.91367254, 518.63620527])
        """
        # Make the two vector using 3 markers, which is on the same plane.
        v1 = (a[0] - c[0], a[1] - c[1], a[2] - c[2])
        v2 = (b[0] - c[0], b[1] - c[1], b[2] - c[2])

        # v3 is np.cross vector of v1, v2
        # and then it is normalized.
        # v3 = np.cross(v1,v2)
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

        This function creates and returns a rotation matrix about axes x, y, z.

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
        # Convert the x, y, z rotation angles from degrees to radians
        x = radians(x)
        y = radians(y)
        z = radians(z)

        # Making elemental rotations about each of the x, y, z axes
        Rx = [[1, 0, 0], [0, cos(x), sin(x) * -1], [0, sin(x), cos(x)]]
        Ry = [[cos(y), 0, sin(y)], [0, 1, 0], [sin(y) * -1, 0, cos(y)]]
        Rz = [[cos(z), sin(z) * -1, 0], [sin(z), cos(z), 0], [0, 0, 1]]

        # Making the rotation matrix around x, y, z axes using matrix multiplication
        Rxy = np.matmul(Rx, Ry)
        Rxyz = np.matmul(Rxy, Rz)

        return Rxyz

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
        >>> from .pycgm import CGM
        >>> axis_p = np.array([[ 0.0464229 ,   0.99648672,  0.06970743],
        ...                    [ 0.99734011,  -0.04231089, -0.05935067],
        ...                    [-0.05619277,   0.07227725, -0.99580037]])
        >>> axis_d = np.array([[-0.18067218,  -0.98329158, -0.02225371],
        ...                    [ 0.71383942,  -0.1155303 , -0.69071415],
        ...                    [ 0.67660243,  -0.1406784 ,  0.7227854 ]])
        >>> CGM.get_angle(axis_p, axis_d)
        array([-175.65183483,  -39.6322192 ,  100.2668477 ])
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

        angle = np.array([180.0 * beta / pi, 180.0 * alpha / pi, 180.0 * gamma / pi])

        return angle

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
           1990;8(3):383-92.

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
        >>> from .pycgm import CGM
        >>> pelvis_axis = np.array([[ 251.60830688, 391.74131775, 1032.89349365],
        ...                         [ 251.74063624, 392.72694721, 1032.78850073],
        ...                         [ 250.61711554, 391.87232862, 1032.8741063 ],
        ...                         [ 251.60295336, 391.84795134, 1033.88777762]])
        >>> measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
        ...                 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
        >>> CGM.hip_axis_calc(pelvis_axis, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[308.38050352, 322.80342433, 937.98979092],
               [182.57097799, 339.43231799, 935.52900136],
               [245.47574075, 331.11787116, 936.75939614],
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
        >>> from .pycgm import CGM
        >>> rtib, ltib, rank, lank = np.array([[433.97537231, 211.93408203, 273.3008728 ],
        ...                                    [50.04016495 , 235.90718079, 364.32226562],
        ...                                    [422.77005005, 217.74053955, 92.86152649 ],
        ...                                    [58.57380676 , 208.54806519, 86.16953278 ]])
        >>> knee_origin = np.array([[364.17774614, 292.17051722, 515.19181496],
        ...                         [143.55478579, 279.90370346, 524.78408753]])
        >>> measurements = {'RightAnkleWidth': 70.0, 'LeftAnkleWidth': 70.0,
        ...                 'RightTibialTorsion': 0.0, 'LeftTibialTorsion': 0.0}
        >>> CGM.ankle_axis_calc(rtib, ltib, rank, lank, knee_origin, measurements)
        array([[393.76181609, 247.67829633,  87.73775041],
               [394.48171575, 248.37201349,  87.715368  ],
               [393.07114385, 248.39110006,  87.61575574],
               [393.69314056, 247.78157916,  88.73002876],
               [ 98.74901939, 219.46930221,  80.63068161],
               [ 98.47494966, 220.42553804,  80.52821783],
               [ 97.79246671, 219.20927276,  80.76255902],
               [ 98.84848169, 219.60345781,  81.61663776]])
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

        # Determine the position of ankleJointCenter using findJointC function
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

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> rtoe, ltoe = np.array([[442.81997681, 381.62280273, 42.66047668],
        ...                        [39.43652725 , 382.44522095, 41.78911591]])
        >>> ankle_axis = np.array([[393.76181608, 247.67829633, 87.73775041],
        ...                        [394.4817575 , 248.37201348, 87.715368  ],
        ...                        [393.07114384, 248.39110006, 87.61575574],
        ...                        [393.69314056, 247.78157916, 88.73002876],
        ...                        [98.74901939 , 219.46930221, 80.6306816 ],
        ...                        [98.47494966 , 220.42553803, 80.52821783],
        ...                        [97.79246671 , 219.20927275, 80.76255901],
        ...                        [98.84848169 , 219.60345781, 81.61663775]])
        >>> measurements = {'RightStaticRotOff': 0.015683497632642047, 'LeftStaticRotOff': 0.009402910292403012,
        ...                 'RightStaticPlantFlex': 0.2702417907002758, 'LeftStaticPlantFlex': 0.20251085737834015}
        >>> CGM.foot_axis_calc(rtoe, ltoe, ankle_axis, measurements)  # doctest: +NORMALIZE_WHITESPACE
        array([[442.81997681, 381.62280273,  42.66047668],
               [442.84624127, 381.6513024 ,  43.65972537],
               [441.87735057, 381.9563035 ,  42.67574106],
               [442.48716163, 380.68048378,  42.69610043],
               [ 39.43652725, 382.44522095,  41.78911591],
               [ 39.56652626, 382.50901001,  42.77857597],
               [ 38.49313328, 382.14606841,  41.93234851],
               [ 39.74166341, 381.4931502 ,  41.81040459]])
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
        r_axis_z = [ankle_jc_r[0] - rtoe[0], ankle_jc_r[1] - rtoe[1], ankle_jc_r[2] - rtoe[2]]
        r_axis_z_div = np.linalg.norm(r_axis_z)
        r_axis_z = [r_axis_z[0] / r_axis_z_div, r_axis_z[1] / r_axis_z_div, r_axis_z[2] / r_axis_z_div]

        # Bring the flexion axis of ankle axes from ankle_axis, and normalize it.
        y_flex_r = [ankle_flexion_r[0] - ankle_jc_r[0], ankle_flexion_r[1] - ankle_jc_r[1],
                    ankle_flexion_r[2] - ankle_jc_r[2]]
        y_flex_r_div = np.linalg.norm(y_flex_r)
        y_flex_r = [y_flex_r[0] / y_flex_r_div, y_flex_r[1] / y_flex_r_div, y_flex_r[2] / y_flex_r_div]

        # X axis is calculated as a cross product of Z axis and ankle flexion axis.
        r_axis_x = np.cross(y_flex_r, r_axis_z)
        r_axis_x_div = np.linalg.norm(r_axis_x)
        r_axis_x = [r_axis_x[0] / r_axis_x_div, r_axis_x[1] / r_axis_x_div, r_axis_x[2] / r_axis_x_div]

        # Y axis is then perpendicularly calculated from Z axis and X axis, and normalized.
        r_axis_y = np.cross(r_axis_z, r_axis_x)
        r_axis_y_div = np.linalg.norm(r_axis_y)
        r_axis_y = [r_axis_y[0] / r_axis_y_div, r_axis_y[1] / r_axis_y_div, r_axis_y[2] / r_axis_y_div]

        r_foot_axis = [r_axis_x, r_axis_y, r_axis_z]

        # Left
        # Z axis is from TOE marker to AJC, normalized.
        l_axis_z = [ankle_jc_l[0] - ltoe[0], ankle_jc_l[1] - ltoe[1], ankle_jc_l[2] - ltoe[2]]
        l_axis_z_div = np.linalg.norm(l_axis_z)
        l_axis_z = [l_axis_z[0] / l_axis_z_div, l_axis_z[1] / l_axis_z_div, l_axis_z[2] / l_axis_z_div]

        # Bring the flexion axis of ankle axes from ankle_axis, and normalize it.
        y_flex_l = [ankle_flexion_l[0] - ankle_jc_l[0], ankle_flexion_l[1] - ankle_jc_l[1],
                    ankle_flexion_l[2] - ankle_jc_l[2]]
        y_flex_l_div = np.linalg.norm(y_flex_l)
        y_flex_l = [y_flex_l[0] / y_flex_l_div, y_flex_l[1] / y_flex_l_div, y_flex_l[2] / y_flex_l_div]

        # X axis is calculated as a cross product of Z axis and ankle flexion axis.
        l_axis_x = np.cross(y_flex_l, l_axis_z)
        l_axis_x_div = np.linalg.norm(l_axis_x)
        l_axis_x = [l_axis_x[0] / l_axis_x_div, l_axis_x[1] / l_axis_x_div, l_axis_x[2] / l_axis_x_div]

        # Y axis is then perpendicularly calculated from Z axis and X axis, and normalized.
        l_axis_y = np.cross(l_axis_z, l_axis_x)
        l_axis_y_div = np.linalg.norm(l_axis_y)
        l_axis_y = [l_axis_y[0] / l_axis_y_div, l_axis_y[1] / l_axis_y_div, l_axis_y[2] / l_axis_y_div]

        l_foot_axis = [l_axis_x, l_axis_y, l_axis_z]

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

        r_axis = [[(r_foot_axis[0][0]), (r_foot_axis[0][1]), (r_foot_axis[0][2])],
                  [(r_foot_axis[1][0]), (r_foot_axis[1][1]), (r_foot_axis[1][2])],
                  [(r_foot_axis[2][0]), (r_foot_axis[2][1]), (r_foot_axis[2][2])]]

        l_axis = [[(l_foot_axis[0][0]), (l_foot_axis[0][1]), (l_foot_axis[0][2])],
                  [(l_foot_axis[1][0]), (l_foot_axis[1][1]), (l_foot_axis[1][2])],
                  [(l_foot_axis[2][0]), (l_foot_axis[2][1]), (l_foot_axis[2][2])]]

        # Rotate incorrect foot axis around y axis first.

        # Right
        r_rotmat = [[(cos(r_beta) * r_axis[0][0] + sin(r_beta) * r_axis[2][0]),
                     (cos(r_beta) * r_axis[0][1] + sin(r_beta) * r_axis[2][1]),
                     (cos(r_beta) * r_axis[0][2] + sin(r_beta) * r_axis[2][2])],
                    [r_axis[1][0], r_axis[1][1], r_axis[1][2]],
                    [(-1 * sin(r_beta) * r_axis[0][0] + cos(r_beta) * r_axis[2][0]),
                     (-1 * sin(r_beta) * r_axis[0][1] + cos(r_beta) * r_axis[2][1]),
                     (-1 * sin(r_beta) * r_axis[0][2] + cos(r_beta) * r_axis[2][2])]]
        # Left
        l_rotmat = [[(cos(l_beta) * l_axis[0][0] + sin(l_beta) * l_axis[2][0]),
                     (cos(l_beta) * l_axis[0][1] + sin(l_beta) * l_axis[2][1]),
                     (cos(l_beta) * l_axis[0][2] + sin(l_beta) * l_axis[2][2])],
                    [l_axis[1][0], l_axis[1][1], l_axis[1][2]],
                    [(-1 * sin(l_beta) * l_axis[0][0] + cos(l_beta) * l_axis[2][0]),
                     (-1 * sin(l_beta) * l_axis[0][1] + cos(l_beta) * l_axis[2][1]),
                     (-1 * sin(l_beta) * l_axis[0][2] + cos(l_beta) * l_axis[2][2])]]

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
        r_axis_x = r_rotmat[0]
        r_axis_y = r_rotmat[1]
        r_axis_z = r_rotmat[2]
        l_axis_x = l_rotmat[0]
        l_axis_y = l_rotmat[1]
        l_axis_z = l_rotmat[2]

        # Attach each axis to the origin
        rx_axis = [r_axis_x[0] + toe_jc_r[0], r_axis_x[1] + toe_jc_r[1], r_axis_x[2] + toe_jc_r[2]]
        ry_axis = [r_axis_y[0] + toe_jc_r[0], r_axis_y[1] + toe_jc_r[1], r_axis_y[2] + toe_jc_r[2]]
        rz_axis = [r_axis_z[0] + toe_jc_r[0], r_axis_z[1] + toe_jc_r[1], r_axis_z[2] + toe_jc_r[2]]

        lx_axis = [l_axis_x[0] + toe_jc_l[0], l_axis_x[1] + toe_jc_l[1], l_axis_x[2] + toe_jc_l[2]]
        ly_axis = [l_axis_y[0] + toe_jc_l[0], l_axis_y[1] + toe_jc_l[1], l_axis_y[2] + toe_jc_l[2]]
        lz_axis = [l_axis_z[0] + toe_jc_l[0], l_axis_z[1] + toe_jc_l[1], l_axis_z[2] + toe_jc_l[2]]

        return np.array([toe_jc_r, rx_axis, ry_axis, rz_axis, toe_jc_l, lx_axis, ly_axis, lz_axis])

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
        >>> from .pycgm import CGM
        >>> global_axis = np.array([[1,0,0],[0,1,0],[0,0,1]])
        >>> pelvis_axis = np.array([[251.60830688, 391.74131774, 1032.89349365],
        ...                         [251.74063624, 392.72694720, 1032.78850073],
        ...                         [250.61711554, 391.87232862, 1032.87410630],
        ...                         [251.60295335, 391.84795133, 1033.88777762]])
        >>> CGM.pelvis_angle_calc(global_axis, pelvis_axis)
        array([-0.30849508, -6.12129284,  7.57143134])
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
        >>> from .pycgm import CGM
        >>> hip_axis = np.array([[np.nan, np.nan, np.nan],
        ...                      [np.nan, np.nan, np.nan],
        ...                      [245.47574167, 331.11787135, 936.75939593],
        ...                      [245.60807102, 332.10350081, 936.65440301],
        ...                      [244.48455032, 331.24888223, 936.74000858],
        ...                      [245.47038814, 331.22450494, 937.75367990]])
        >>> knee_axis = np.array([[364.17774613, 292.17051731, 515.19181496],
        ...                       [364.61959153, 293.06758353, 515.18513093],
        ...                       [363.29019771, 292.60656648, 515.04309095],
        ...                       [364.04724540, 292.24216263, 516.18067111],
        ...                       [143.55478579, 279.90370346, 524.78408753],
        ...                       [143.65611281, 280.88685896, 524.63197541],
        ...                       [142.56434499, 280.01777942, 524.86163553],
        ...                       [143.64837986, 280.04650380, 525.76940383]])
        >>> CGM.hip_angle_calc(hip_axis, knee_axis)
        array([[  2.91422854,  -6.86706805, -18.82100186],
               [ -2.86020432,  -5.34565068,  -1.80256237]])
        """

        hip_axis_mod = CGM.subtract_origin(hip_axis[2:])
        r_knee_axis_mod = CGM.subtract_origin(knee_axis[:4])
        l_knee_axis_mod = CGM.subtract_origin(knee_axis[4:])

        r_hip_angle = CGM.get_angle(hip_axis_mod, r_knee_axis_mod)
        l_hip_angle = CGM.get_angle(hip_axis_mod, l_knee_axis_mod)

        # GCS fix
        r_hip_angle = np.array([r_hip_angle[0] * -1, r_hip_angle[1], r_hip_angle[2] * -1 + 90])
        l_hip_angle = np.array([l_hip_angle[0] * -1, l_hip_angle[1] * -1, l_hip_angle[2] - 90])

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
        >>> from .pycgm import CGM
        >>> knee_axis = np.array([[364.17774614, 292.17051722, 515.19181496],
        ...                       [364.61959153, 293.06758353, 515.18513093],
        ...                       [363.29019771, 292.60656648, 515.04309095],
        ...                       [364.04724541, 292.24216264, 516.18067112],
        ...                       [143.55478579, 279.90370346, 524.78408753],
        ...                       [143.65611282, 280.88685896, 524.63197541],
        ...                       [142.56434499, 280.01777943, 524.86163553],
        ...                       [143.64837987, 280.04650381, 525.76940383]])
        >>> ankle_axis = np.array([[393.76181608, 247.67829633, 87.73775041],
        ...                        [394.48171575, 248.37201348, 87.71536800],
        ...                        [393.07114384, 248.39110006, 87.61575574],
        ...                        [393.69314056, 247.78157916, 88.73002876],
        ...                        [ 98.74901939, 219.46930221, 80.63068160],
        ...                        [ 98.47494966, 220.42553803, 80.52821783],
        ...                        [ 97.79246671, 219.20927275, 80.76255901],
        ...                        [ 98.84848169, 219.60345781, 81.61663775]])
        >>> CGM.knee_angle_calc(knee_axis, ankle_axis)
        array([[  3.19436865,   2.38341045, -19.47591616],
               [ -0.45848726,  -0.3866728 , -21.87580851]])
        """
        r_knee_axis_mod = CGM.subtract_origin(knee_axis[:4])
        l_knee_axis_mod = CGM.subtract_origin(knee_axis[4:])
        r_ankle_axis_mod = CGM.subtract_origin(ankle_axis[:4])
        l_ankle_axis_mod = CGM.subtract_origin(ankle_axis[4:])

        r_knee_angle = CGM.get_angle(r_knee_axis_mod, r_ankle_axis_mod)
        l_knee_angle = CGM.get_angle(l_knee_axis_mod, l_ankle_axis_mod)

        # GCS fix
        r_knee_angle = np.array([r_knee_angle[0], r_knee_angle[1], r_knee_angle[2] * -1 + 90])
        l_knee_angle = np.array([l_knee_angle[0], l_knee_angle[1] * -1, l_knee_angle[2] - 90])

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
        >>> from .pycgm import CGM
        >>> ankle_axis = np.array([[393.76181608, 247.67829633, 87.73775041],
        ...                        [394.48171575, 248.37201348, 87.71536800],
        ...                        [393.07114384, 248.39110006, 87.61575574],
        ...                        [393.69314056, 247.78157916, 88.73002876],
        ...                        [ 98.74901939, 219.46930221, 80.63068160],
        ...                        [ 98.47494966, 220.42553803, 80.52821783],
        ...                        [ 97.79246671, 219.20927275, 80.76255901],
        ...                        [ 98.84848169, 219.60345781, 81.61663775]])
        >>> foot_axis = np.array([[442.81997681, 381.62280273, 42.66047668],
        ...                       [442.84624127, 381.65130240, 43.65972538],
        ...                       [441.87735056, 381.95630350, 42.67574106],
        ...                       [442.48716163, 380.68048378, 42.69610044],
        ...                       [39.43652725 , 382.44522095, 41.78911591],
        ...                       [39.56652626 , 382.50901000, 42.77857597],
        ...                       [38.49313328 , 382.14606841, 41.93234850],
        ...                       [39.74166342 , 381.49315020, 41.81040458]])
        >>> CGM.knee_angle_calc(ankle_axis, foot_axis)
        array([[-92.50533765,  26.4981019 ,  -7.68822002],
               [-94.38467038,  -2.37873795,   0.59929708]])
        """
        r_ankle_axis_mod = CGM.subtract_origin(ankle_axis[:4])
        l_ankle_axis_mod = CGM.subtract_origin(ankle_axis[4:])
        r_foot_axis_mod = CGM.subtract_origin(foot_axis[:4])
        l_foot_axis_mod = CGM.subtract_origin(foot_axis[4:])

        r_ankle_angle = CGM.get_angle(r_ankle_axis_mod, r_foot_axis_mod)
        l_ankle_angle = CGM.get_angle(l_ankle_axis_mod, l_foot_axis_mod)

        # GCS fix
        r_ankle_angle = np.array([r_ankle_angle[0] * -1 - 90, r_ankle_angle[1] * -1 + 90, r_ankle_angle[2]])
        l_ankle_angle = np.array([l_ankle_angle[0] * -1 - 90, l_ankle_angle[1] - 90, l_ankle_angle[2] * -1])

        return np.array([r_ankle_angle, l_ankle_angle])

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

    # Input and output handlers
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
    def get_static(motion_data, measurements, flat_foot, gcs):
        """ Get Static Offset function

        Calculate the static offset angle values and return the values in radians

        Parameters
        ----------
        motion_data : dict
            Dictionary of marker lists.
        measurements : dict, optional
            A dictionary containing the subject measurements given from the file input.
        flat_foot : boolean, optional
            A boolean indicating if the feet are flat or not.
            The default value is False.
        gcs : array, optional
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
    def static_calculation(rtoe, ltoe, rhee, lhee, ankle_jc, knee_jc, flat_foot, measurements):
        """The Static Angle Calculation function

        Takes in anatomical uncorrect axis and anatomical correct axis.
        Correct axis depends on foot flat options.

        Calculates the offset angle between that two axis.

        It is rotated from uncorrect axis in YXZ order.

        Parameters
        ----------
        rtoe, ltoe, rhee, lhee : dict
            A 1x3 ndarray of each respective marker containing the XYZ positions.
        ankle_jc : array
            An ndarray containing the x,y,z axes marker positions of the ankle joint centers.
        knee_jc : array
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
        """
