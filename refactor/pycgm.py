import refactor.io as io
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
        write_axes : bool, optional
            Boolean option to enable or disable writing of axis results to output file
        write_angles : bool, optional
            Boolean option to enable or disable writing of angle results to output file
        write_com : bool, optional
            Boolean option to enable or disable writing of center of mass results to output file
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
        self.marker_map = {marker:marker for marker in io.marker_keys()}
        self.marker_data, self.marker_idx = io.load_marker_data(path_dynamic)
        

    def run(self):
        """Execute the CGM calculations function

        Load in appropriate data from IO using paths.
        Perform any necessary prep on data.
        Run the static calibration trial.
        Run the dynamic trial to calculate all axes and angles.
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
    def cross(a, b):
        """Cross Product function

        Given vectors a and b, calculate the cross product.

        Parameters
        ----------
        a : list
            First 3D vector.
        b : list
            Second 3D vector.

        Returns
        -------
        list
            The cross product of vector a and vector b.
        """

    @staticmethod
    def norm2d(v):
        """2D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        float
            The normalization of the vector as a float.
        """

    @staticmethod
    def norm3d(v):
        """3D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        list
            The normalization of the vector returned as a float in an array.
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
    def matrix_mult(a, b):
        """Matrix multiplication function

        This function returns the product of a matrix multiplication given two matrices.

        Let the dimension of the matrix A be: m by n,
        let the dimension of the matrix B be: p by q,
        multiplication will only possible if n = p,
        creating a matrix of m by q size.

        Parameters
        ----------
        a : list
            First matrix, in a 2D array format.
        b : list
            Second matrix, in a 2D array format.

        Returns
        -------
        c : list
            The product of the matrix multiplication.
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
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        """The Pelvis Axis Calculation function

        Calculates the pelvis joint center and axis and returns both.

        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: CGM.cross product of x_axis and y_axis.

        Parameters
        ----------
        rasi, lasi : array
            A 1x3 array of each respective marker containing the XYZ positions.
        rpsi, lpsi, sacr : array, optional
            A 1x3 array of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x1x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.

        References
        ----------
        .. [12] Kadaba MP, Ramakrishnan HK, Wootten ME.
           Measurement of lower extremity kinematics during level walking.
           Journal of orthopaedic research: official publication of the Orthopaedic Research Society.
           1990;8(3):383–92.
        """

    @staticmethod
    def hip_axis_calc(pel, measurements):
        """The Hip Axis Calculation function

        Calculates the hip joint center and returns the hip joint center.

        Other landmarks used: origin, sacrum
        Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure, 
        InterAsisDistance, L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

        Parameters
        ----------
        pel : array
            A array containing a 1x3 array representing the pelvis origin, followed by
            a 3x1x3 array representing the x, y, z components of the pelvis axis.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right hip origin, right hip x, y, and z
            axis components, left hip origin, and left hip x, y, and z axis components.

        References
        ----------
        .. [20]  Davis RB, Ounpuu S, Tyburski D, Gage JR.
           A gait analysis data collection and reduction technique. Human Movement Science.
           1991;10(5):575–587.
        """

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_jc, delta, measurements):
        """The Knee Axis Calculation function

        Calculates the knee joint axis and returns the knee origin and axis

        Markers used: RTHI, LTHI, RKNE, LKNE, hip_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : array
            A 1x3 array of each respective marker containing the XYZ positions.
        hip_jc : array
            An array of hip_JC containing the x,y,z axes marker positions of the hip joint center.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right knee origin, right knee x, y, and z
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
    def ankle_axis_calc(rtib, ltib, rank, lank, knee_jc, delta, measurements):
        """The Ankle Axis Calculation function

        Calculates the ankle joint axis and returns the ankle origin and axis

        Markers used: RTIB, LTIB, RANK, LANK, knee_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

        Parameters
        ----------
        rtib, ltib, rank, lank : array
            A 1x3 array of each respective marker containing the XYZ positions.
        knee_jc : array
            A numpy array representing the knee joint center origin and axis.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file
        measurements : dict, optional
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.
        """

    @staticmethod
    def foot_axis_calc(rtoe, ltoe, measurements, ankle_jc, knee_jc, delta):
        """The Ankle Axis Calculation function

        Calculate the foot joint axis by rotating incorrect foot joint axes about offset angle.
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
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        ankle_jc : array
            An array containing the x,y,z axes marker positions of the ankle joint center.
        knee_jc : array
            An array containing the x,y,z axes marker positions of the knee joint center.
        delta
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Modifies
        --------
        Axis changes following to the static info.

        you can set the static_info by the button. and this will calculate the offset angles
        the first setting, the foot axis show foot uncorrected anatomical reference axis(Z_axis point to the AJC from TOE)

        if press the static_info button so if static_info is not None,
        and then the static offsets angles are applied to the reference axis.
        the reference axis is Z axis point to HEE from TOE
        """

    @staticmethod
    def head_axis_calc(lfhd, rfhd, lbhd, rbhd, measurements):
        """The Head Axis Calculation function

        Calculates the head joint center and returns the head joint center and axis.

        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        lfhd, rfhd, lbhd, rbhd : array
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x1x3 ndarray that contains the head origin and the
            head x, y, and z axis components.
        """

    @staticmethod
    def thorax_axis_calc(clav, c7, strn, t10):
        """The Thorax Axis Calculation function

        Calculates the thorax joint center and returns the thorax joint center and axis.

        Markers used: CLAV, C7, STRN, T10

        Parameters
        ----------
        clav, c7, strn, t10 : array
            A 1x3 array of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x1x3 ndarray that contains the thorax origin and the
            thorax x, y, and z axis components.
        """

    @staticmethod
    def shoulder_axis_calc(rsho, lsho, thorax, wand, measurements):
        """The Shoulder Axis Calculation function
        Calculate each shoulder joint center and returns it.

        Markers used: RSHO, LSHO
        Subject Measurement values used: RightShoulderOffset, LeftShoulderOffset

        Parameters
        ----------
        rsho, lsho : dict
            A 1x3 array of each respective marker containing the XYZ positions.
        thorax : array
            An array containing several x,y,z markers for the thorax.
        wand : array
            An array containing two x,y,z markers for wand.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right shoulder origin, right shoulder x, y, and z
            axis components, left shoulder origin, and left shoulder x, y, and z axis components.
        """

    @staticmethod
    def elbow_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, thorax, shoulder_jc, wand, measurements):
        """The Elbow Axis Calculation function

        Calculate each elbow joint axis and returns it.

        Markers used: RSHO, LSHO, RELB, LELB, RWRA , RWRB, LWRA, LWRB
        Subject Measurement values used: RightElbowWidth, LeftElbowWidth

        Parameters
        ----------
        rsho, lsho, relb, lelb, rwra , rwrb, lwra, lwrb : array
            A 1x3 array of each respective marker containing the XYZ positions.
        thorax : array
            An array containing the thorax joint center origin and axis.
        shoulder_jc : array
            A 2x3 array containing the x,y,z position of the right and left shoulder joint center.
        wand : array
            A 2x3 array containing the x,y,z position of the right and left wand marker.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right elbow origin, right elbow x, y, and z
            axis components, left elbow origin, and left elbow x, y, and z axis components.
        """

    @staticmethod
    def wrist_axis_calc(rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb, shoulder_jc, elbow_jc, wand):
        """The Wrist Axis Calculation function

        Calculate each wrist joint axis and returns it.

        Markers used: RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB

        Parameters
        ----------
        rsho, lsho, relb, lelb, rwra, rwrb, lwra, lwrb : array
            A 1x3 array of each respective marker containing the XYZ positions.
        shoulder_jc : array
            A 2x3 array containing the x,y,z position of the right and left shoulder joint center.
        elbow_jc : array
            An array containing position of the left and right elbow joint centers.
        wand : array
            A 2x3 array containing the x,y,z position of the right and left wand marker.

        Returns
        --------
        array
            Returns an 8x1x3 ndarray that contains the right wrist origin, right wrist x, y, and z
            axis components, left wrist origin, and left wrist x, y, and z axis components.
        """

    @staticmethod
    def hand_axis_calc(rwra, wrb, lwra, lwrb, rfin, lfin, elbow_jc, wrist_jc, measurements):
        """Calculate the Hand joint axis ( Hand) function.

        Takes in a dictionary of x,y,z positions and marker names.
        and takes the elbow axis and wrist axis.
        Calculate each Hand joint axis and returns it.

        Markers used: RWRA, RWRB, LWRA, LWRB, RFIN, LFIN
        Subject Measurement values used: RightHandThickness, LeftHandThickness

        Parameters
        ----------
        rwra, wrb, lwra, lwrb, rfin, lfin : array
            A 1x3 array of each respective marker containing the XYZ positions.
        elbow_jc : array
            A 2x3 array containing the x,y,z position of the right and left elbow joint center.
        wrist_jc : array
            A 2x3 array containing the x,y,z position of the right and left wrist joint center.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right hand origin, right hand x, y, and z
            axis components, left hand origin, and left hand x, y, and z axis components.
        """
        pass

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
        pass

    @staticmethod
    def norm2d(v):
        """2D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        float
            The normalization of the vector as a float.
        """

    @staticmethod
    def norm3d(v):
        """3D Vector normalization function

        This function calculates the normalization of a 3-dimensional vector.

        Parameters
        ----------
        v : list
            A 3D vector.

        Returns
        -------
        list
            The normalization of the vector returned as a float in an array.
        """

    @staticmethod
    def matrix_mult(a, b):
        """Matrix multiplication function

        This function returns the product of a matrix multiplication given two matrices.

        Let the dimension of the matrix A be: m by n,
        let the dimension of the matrix B be: p by q,
        multiplication will only possible if n = p,
        creating a matrix of m by q size.

        Parameters
        ----------
        a : list
            First matrix, in a 2D array format.
        b : list
            Second matrix, in a 2D array format.

        Returns
        -------
        c : list
            The product of the matrix multiplication.
        """

    @staticmethod
    def cross(a, b):
        """Cross Product function

        Given vectors a and b, calculate the cross product.

        Parameters
        ----------
        a : list
            First 3D vector.
        b : list
            Second 3D vector.

        Returns
        -------
        list
            The cross product of vector a and vector b.
        """

    @staticmethod
    def average(list):
        pass

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
    def get_static():
        pass

    @staticmethod
    def iad_calculation():
        pass

    @staticmethod
    def static_calculation_head():
        pass

    @staticmethod
    def static_calculation():
        pass

@staticmethod
    def pelvis_axis_calc(rasi, lasi, rpsi=None, lpsi=None, sacr=None):
        """The Pelvis Axis Calculation function

        Calculates the pelvis joint center and axis and returns both.

        Markers used: RASI, LASI, RPSI, LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: CGM.cross product of x_axis and y_axis.

        Parameters
        ----------
        rasi, lasi : array
            A 1x3 array of each respective marker containing the XYZ positions.
        rpsi, lpsi, sacr : array, optional
            A 1x3 array of each respective marker containing the XYZ positions.

        Returns
        -------
        array
            Returns a 4x1x3 ndarray that contains the pelvis origin and the
            pelvis x, y, and z axis components.

        References
        ----------
        .. [12] Kadaba MP, Ramakrishnan HK, Wootten ME.
           Measurement of lower extremity kinematics during level walking.
           Journal of orthopaedic research: official publication of the Orthopaedic Research Society.
           1990;8(3):383–92.
        """

    @staticmethod
    def hip_axis_calc(pel, measurements):
        """The Hip Axis Calculation function

        Calculates the hip joint center and returns the hip joint center.

        Other landmarks used: origin, sacrum
        Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure, 
        InterAsisDistance, L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

        Parameters
        ----------
        pel : array
            A array containing a 1x3 array representing the pelvis origin, followed by
            a 3x1x3 array representing the x, y, z components of the pelvis axis.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right hip origin, right hip x, y, and z
            axis components, left hip origin, and left hip x, y, and z axis components.

        References
        ----------
        .. [20]  Davis RB, Ounpuu S, Tyburski D, Gage JR.
           A gait analysis data collection and reduction technique. Human Movement Science.
           1991;10(5):575–587.
        """

    @staticmethod
    def knee_axis_calc(rthi, lthi, rkne, lkne, hip_jc, delta, measurements):
        """The Knee Axis Calculation function

        Calculates the knee joint axis and returns the knee origin and axis

        Markers used: RTHI, LTHI, RKNE, LKNE, hip_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)

        Parameters
        ----------
        rthi, lthi, rkne, lkne : array
            A 1x3 array of each respective marker containing the XYZ positions.
        hip_jc : array
            An array of hip_JC containing the x,y,z axes marker positions of the hip joint center.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right knee origin, right knee x, y, and z
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
    def ankle_axis_calc(rtib, ltib, rank, lank, knee_jc, delta, measurements):
        """The Ankle Axis Calculation function

        Calculates the ankle joint axis and returns the ankle origin and axis

        Markers used: RTIB, LTIB, RANK, LANK, knee_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).

        Parameters
        ----------
        rtib, ltib, rank, lank : array
            A 1x3 array of each respective marker containing the XYZ positions.
        knee_jc : array
            A numpy array representing the knee joint center origin and axis.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file
        measurements : dict, optional
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right ankle origin, right ankle x, y, and z
            axis components, left ankle origin, and left ankle x, y, and z axis components.

        References
        ----------
        .. [43]  Baker R.
           Measuring walking: a handbook of clinical gait analysis.
           Hart Hilary M, editor. Mac Keith Press; 2013.
        """

    @staticmethod
    def foot_axis_calc(rtoe, ltoe, measurements, ankle_jc, knee_jc, delta):
        """The Ankle Axis Calculation function

        Calculate the foot joint axis by rotating incorrect foot joint axes about offset angle.
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
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        ankle_jc : array
            An array containing the x,y,z axes marker positions of the ankle joint center.
        knee_jc : array
            An array containing the x,y,z axes marker positions of the knee joint center.
        delta
            The length from marker to joint center, retrieved from subject measurement file.

        Returns
        -------
        array
            Returns an 8x1x3 ndarray that contains the right foot origin, right foot x, y, and z
            axis components, left foot origin, and left foot x, y, and z axis components.

        Modifies
        --------
        Axis changes following to the static info.

        you can set the static_info by the button. and this will calculate the offset angles
        the first setting, the foot axis show foot uncorrected anatomical reference axis(Z_axis point to the AJC from TOE)

        if press the static_info button so if static_info is not None,
        and then the static offsets angles are applied to the reference axis.
        the reference axis is Z axis point to HEE from TOE
        """

    @staticmethod
    def head_axis_calc(lfhd, rfhd, lbhd, rbhd, measurements):
        """The Head Axis Calculation function

        Calculates the head joint center and returns the head joint center and axis.

        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        lfhd, rfhd, lbhd, rbhd : array
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        array
            Returns a 4x1x3 ndarray that contains the head origin and the
            head x, y, and z axis components.
        """

    @staticmethod
    def ankle_angle_calc():
        pass
