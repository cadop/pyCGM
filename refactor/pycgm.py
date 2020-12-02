import numpy as np


class CGM:

    def __init__(self, path_static, path_dynamic, path_measurements, path_results=None, path_com=None, cores=1):
        pass

    def run(self):
        """Execute the CGM calculations function

        Load in appropriate data from IO using paths.
        Perform any necessary prep on data.
        Run the static calibration trial.
        Run the dynamic trial to calculate all axes and angles.
        """
        pass

    def map(self, old, new):
        """Remap marker function

        Remaps a single marker from the expected name in CGM to a new one, using `old` and `new`.

        Parameters
        ----------
        old : str
            String containing the marker name that pycgm currently expects.
        new : str
            String containing the marker name to map `old` to.
        """
        pass

    def full_map(self, mapping):
        """Remap all markers function

        Uses the passed dictionary as the mapping for all markers.

        Parameters
        ----------
        mapping: dict
            Dictionary where each key is a string of pycgm's expected marker
            name and each value is a string of the new marker name.
        """
        pass

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
        c : list
            The cross product of vector a and vector b.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> a = [6.25286248, 7.91367254, 18.63620527]
        >>> b = [3.49290439, 4.42038315, 19.23948238]
        >>> np.around(CGM.cross(a, b),8)
        array([ 6.98757956e+01, -5.52073543e+01, -1.65361000e-03])
        """
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

        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> v = [105.141121037153, 101.890788777524, 326.7710280245359]
        >>> np.around(CGM.norm2d(v),8)
        358.07218955
        """
        pass

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

        Examples
        --------
        >>> from .pycgm import CGM
        >>> v = [125.44928201, 143.94301493, 213.49204956]
        >>> CGM.norm3d(v)
        array(286.4192192)
        """
        pass

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

        Examples
        --------
        >>> from .pycgm import CGM
        >>> A = [[11,12,13],[14,15,16]]
        >>> B = [[1,2],[3,4],[5,6]]
        >>> CGM.matrix_mult(A, B)
        [[112, 148], [139, 184]]
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
        r_xyz : list
            The product of the matrix multiplication.

        Examples
        --------
        >>> import numpy as np
        >>> from .pycgm import CGM
        >>> x = 0.5
        >>> y = 0.3
        >>> z = 0.8
        >>> np.around(CGM.rotation_matrix(x, y, z), 8)
        array([[ 0.99988882, -0.01396199,  0.00523596],
               [ 0.01400734,  0.99986381, -0.00872642],
               [-0.00511341,  0.00879879,  0.99994822]])
        >>> x = 0.5
        >>> np.around(CGM.rotation_matrix(x), 8)
        array([[ 1.        ,  0.        ,  0.        ],
               [ 0.        ,  0.99996192, -0.00872654],
               [ 0.        ,  0.00872654,  0.99996192]])
        >>> x = 1
        >>> y = 1
        >>> np.around(CGM.rotation_matrix(x, y), 8)
        array([[ 9.9984770e-01,  0.0000000e+00,  1.7452410e-02],
               [ 3.0459000e-04,  9.9984770e-01, -1.7449750e-02],
               [-1.7449750e-02,  1.7452410e-02,  9.9969541e-01]])
        """
        #TODO add better doctests for this function
        pass

    @staticmethod
    def pelvis_axis_calc(RASI, LASI, RPSI=None, LPSI=None, SACR=None):
        """The Pelvis Axis Calculation function

        Takes in a dictionary of x,y,z positions and marker names, as well as an index
        Calculates the pelvis joint center and axis and returns both.

        Markers used: RASI,LASI,RPSI,LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: Cross product of x_axis and y_axis.

        Parameters
        ---------- 
        RASI, LASI : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        RPSI, LPSI, SACR : array, optional
            A 1x3 array of each respective marker containing the XYZ positions.

        Returns
        -------
        pelvis : array
            Returns an array that contains the pelvis origin in a 1x3 array of xyz values, 
            which is then followed by a 4x1x3 array composed of the pelvis x, y, z 
            axis components, and the sacrum x,y,z position.   
        
        References
        ----------
        .. [12] Kadaba MP, Ramakrishnan HK, Wootten ME. 
           Measurement of lower extremity kinematics during level walking. 
           Journal of orthopaedic research: official publication of the Orthopaedic Research Society. 
           1990;8(3):383–92. 

        Examples
        --------
        >>> import numpy as np 
        >>> from .pyCGM import CGM
        >>> RASI = np.array([ 395.36532593,  428.09790039, 1036.82763672])
        >>> LASI = np.array([ 183.18504333,  422.78927612, 1033.07299805])
        >>> RPSI = np.array([ 341.41815186,  246.72117615, 1055.99145508])
        >>> LPSI = np.array([ 255.79994202,  241.42199707, 1057.30065918])
        >>> CGM.pelvis_axis_calc(RASI, LASI, RPSI, LPSI) #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27518463,  425.44358826, 1034.95031739]), 
        array([[ 289.25243803,  426.43632163, 1034.8321521 ],
        [ 288.27565385,  425.41858059, 1034.93263018],
        [ 289.25467091,  425.56129577, 1035.94315379]]), 
        array([ 298.60904694,  244.07158661, 1056.64605713])]  
        >>> RASI = np.array([ 395.36532593,  428.09790039, 1036.82763672]), 
        >>> LASI = np.array([ 183.18504333,  422.78927612, 1033.07299805]),
        >>> SACR = np.array([ 294.60904694,  242.07158661, 1049.64605713]) }
        >>> CGM.pelvis_axis_calc(RASI, LASI, SACR=SACR) #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27518463,  425.44358826, 1034.95031739]), 
        array([[ 289.25166321,  426.44012508, 1034.87056085],
        [ 288.27565385,  425.41858059, 1034.93263018],
        [ 289.25556415,  425.52289134, 1035.94697483]]), 
        array([ 294.60904694,  242.07158661, 1049.64605713])]
        """
        pass

    @staticmethod
    def hip_axis_calc(pel, measurements):
        """The Hip Axis Calculation function

        Takes in a dictionary of x,y,z positions and marker names, as well as an index.
        Calculates the hip joint center and returns the hip joint center.
        
        Other landmarks used: origin, sacrum
        Subject Measurement values used: MeanLegLength, R_AsisToTrocanterMeasure, InterAsisDistance, L_AsisToTrocanterMeasure

        Hip Joint Center: Computed using Hip Joint Center Calculation (ref. Davis_1991)

        Parameters
        ----------
        pel : array
            A array containing a 1x3 array representing the pelvis origin, followed by 
            a 3x1x3 array representing the x, y, z components of the pelvis axis. 
        measurements : dict
            A dictionary containing the subject measurements given from file input.  

        Returns
        -------
        hip_JC : array
            Returns a 2x3 array that contains the left hip joint center, a 1x3 array containing the x,y,z components
            followed by the right hip joint center, another 1x3 array containing the x,y,z components.
            
        References
        ----------
        .. [20]  Davis RB, Ounpuu S, Tyburski D, Gage JR.
           A gait analysis data collection and reduction technique. Human Movement Science. 
           1991;10(5):575–587. 

        Examples
        --------
        >>> import numpy as np 
        >>> from .pyCGM import CGM
        >>> measurements = { 'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
        ...                'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031 }
        >>> pel_origin = [ 251.60830688, 391.74131775, 1032.89349365]
        >>> pel_x = [251.74063624, 392.72694721, 1032.78850073]
        >>> pel_y = [250.61711554, 391.87232862, 1032.8741063]
        >>> pel_z = [251.60295336, 391.84795134, 1033.88777762]
        >>> pel = [pel_origin, [pel_x, pel_y, pel_z]]
        >>> CGM.hipJointCenter(pel, measurements)
        array([[182.57097799, 339.43231799, 935.52900136],
            [308.38050352, 322.80342433, 937.98979092]])

        """
        # TODO, implement pel = [pel_origin, [pel_x, pel_y, pel_z]]
        # before calling this function
        pass

    @staticmethod
    def knee_axis_calc(RTHI, LTHI, RKNE, LKNE, hip_JC, delta, measurements):
        """The Knee Axis Calculation function

        Takes in a dictionary of xyz positions and marker names, as well as an index.
        and takes the hip axis and pelvis axis.
        Calculates the knee joint axis and returns the knee origin and axis

        Markers used: RTHI, LTHI, RKNE, LKNE, hip_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Knee joint center: Computed using Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        
        Parameters
        ----------
        RTHI, LTHI, RKNE, LKNE : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        hip_JC : array
            An array of hip_JC containing the x,y,z axes marker positions of the hip joint center. 
        delta : float
            The length from marker to joint center, retrieved from subject measurement file.
        measurements : dict
            A dictionary containing the subject measurements given from file input.  

        Returns
        -------
        R, L, axis : array
            Returns an array that contains the knee axis center in a 1x3 array of xyz values, 
            which is then followed by a 2x3x3 array composed of the knee axis center x, y, and z 
            axis components. The xyz axis components are 2x3 arrays consisting of the axis center 
            in the first dimension and the direction of the axis in the second dimension.                                 
            
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
        >>> from .pyCGM import CGM
        >>> measurements = { 'RightKneeWidth' : 105.0, 'LeftKneeWidth' : 105.0 }
        >>> RTHI = np.array([426.50338745, 262.65310669, 673.66247559])
        >>> LTHI = np.array([51.93867874, 320.01849365, 723.03186035])
        >>> RKNE = np.array([416.98687744, 266.22558594, 524.04089355])
        >>> LKNE = np.array([84.62355804, 286.69122314, 529.39819336])
        >>> hip_JC = [[182.57097863, 339.43231855, 935.52900126],
        ...         [309.38050472, 32280342417, 937.98979061]]
        >>> delta = 0
        >>> CGM.knee_axis_calc(RTHI, LTHI, RKNE, LKNE, hip_JC, delta, measurements) #doctest: +NORMALIZE_WHITESPACE
        [array([413.21007973, 266.22558784, 464.66088466]), 
        array([143.55478579, 279.90370346, 524.78408753]), 
        array([[[414.20806312, 266.22558785, 464.59740907],
        [413.14660414, 266.22558786, 463.66290127], 
        [413.21007973, 267.22558784, 464.66088468]],
        [[143.65611281, 280.88685896, 524.63197541],
        [142.56434499, 280.01777942, 524.86163553],
        [143.64837987, 280.0465038 , 525.76940383]]])]
        """
        pass

    @staticmethod
    def ankle_axis_calc(RTIB, LTIB, RANK, LANK, knee_JC, delta, measurements):
        """The Ankle Axis Calculation function

        Takes in a dictionary of xyz positions and marker names, as well as an index.
        and takes the knee axis.
        Calculates the ankle joint axis and returns the ankle origin and axis
        
        Markers used: RTIB, LTIB, RANK, LANK, knee_JC
        Subject Measurement values used: RightKneeWidth, LeftKneeWidth

        Ankle Axis: Computed using Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013).
        
        Parameters
        ----------
        RTIB, LTIB, RANK, LANK : array
            A 1x3 array of each respective marker containing the XYZ positions.
        knee_JC : array
            A numpy array representing the knee joint center origin and axis.
        delta : float
            The length from marker to joint center, retrieved from subject measurement file
        measurements : dict, optional
            A dictionary containing the subject measurements given from file input. 

        Returns
        -------
        R, L, axis : array
            Returns an array that contains the ankle axis origin in a 1x3 array of xyz values, 
            which is then followed by a 3x2x3 array composed of the ankle origin, x, y, and z 
            axis components. The xyz axis components are 2x3 arrays consisting of the origin 
            in the first dimension and the direction of the axis in the second dimension.                         
        
        References 
        ----------
        .. [43]  Baker R. 
           Measuring walking: a handbook of clinical gait analysis. 
           Hart Hilary M, editor. Mac Keith Press; 2013. 

        Examples
        --------
        >>> import numpy as np 
        >>> from .pyCGM import CGM
        >>> measurements = { 'RightAnkleWidth' : 70.0, 'LeftAnkleWidth' : 70.0, 
        ...                'RightTibialTorsion': 0.0, 'LeftTibialTorsion' : 0.0}
        >>> RTIB = np.array([433.97537231, 211.93408203, 273.3008728])
        >>> LTIB = np.array([50.04016495, 235.90718079, 364.32226562])
        >>> RANK = np.array([422.77005005, 217.74053955, 92.86152649])
        >>> LANK = np.array([58.57380676, 208.54806519, 86.16953278])
        >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
        ...           np.array([143.55478579, 279.90370346, 524.78408753]),
        ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
        ...           [363.29019771, 292.60656648, 515.04309095],
        ...           [364.04724541, 292.24216264, 516.18067112]],
        ...           [[143.65611282, 280.88685896, 524.63197541],
        ...           [142.56434499, 280.01777943, 524.86163553],
        ...           [143.64837987, 280.04650381, 525.76940383]]])]
        >>> delta = 0
        >>> CGM.ankle_axis_calc(RTIB, LTIB, RANK, LANK, knee_JC, delta, measurements) #doctest: +NORMALIZE_WHITESPACE
        [array([393.76181609, 247.67829633,  87.73775041]), 
        array([ 98.74901939, 219.46930221,  80.63068161]), 
        [[array([394.48171575, 248.37201349,  87.715368  ]), 
        array([393.07114385, 248.39110006,  87.61575574]), 
        array([393.69314056, 247.78157916,  88.73002876])], 
        [array([ 98.47494966, 220.42553804,  80.52821783]), 
        array([ 97.79246671, 219.20927276,  80.76255902]), 
        array([ 98.84848169, 219.60345781,  81.61663776])]]]
        """
        pass

    @staticmethod
    def foot_axis_calc(RTOE, LTOE, measurements, ankle_JC, knee_JC, delta):
        """The Ankle Axis Calculation function
        
        Takes in a dictionary of xyz positions and marker names.
        and takes the ankle axis and knee axis.
        Calculate the foot joint axis by rotating incorrect foot joint axes about offset angle.
        Returns the foot axis origin and axis.
        
        In case of foot joint center, we've already make 2 kinds of axis for static offset angle. 
        and then, Call this static offset angle as an input of this function for dynamic trial. 

        Special Cases:

        (anatomical uncorrect foot axis)
        If foot flat is true, then make the reference markers instead of HEE marker which height is as same as TOE marker's height.
        otherwise, foot flat is false, use the HEE marker for making Z axis.

        Markers used: RTOE, LTOE
        Other landmarks used: ANKLE_FLEXION_AXIS
        Subject Measurement values used: RightStaticRotOff, RightStaticPlantFlex, LeftStaticRotOff, LeftStaticPlantFlex
            
        Parameters
        ---------- 
        RTOE, LTOE : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from file input. 
        ankle_JC : array
            An array containing the x,y,z axes marker positions of the ankle joint center. 
        knee_JC : array
            An array containing the x,y,z axes marker positions of the knee joint center. 
        delta
            The length from marker to joint center, retrieved from subject measurement file.
            
        Returns
        -------
        R, L, foot_axis : array
            Returns an array that contains the foot axis center in a 1x3 array of xyz values, 
            which is then followed by a 2x3x3 array composed of the foot axis center x, y, and z 
            axis components. The xyz axis components are 2x3 arrays consisting of the axis center 
            in the first dimension and the direction of the axis in the second dimension.   
            This function also saves the static offset angle in a global variable. 
                
        Modifies
        --------   
        Axis changes following to the static info.

        you can set the static_info by the button. and this will calculate the offset angles 
        the first setting, the foot axis show foot uncorrected anatomical reference axis(Z_axis point to the AJC from TOE)

        if press the static_info button so if static_info is not None,
        and then the static offsets angles are applied to the reference axis.
        the reference axis is Z axis point to HEE from TOE

        Examples
        --------
        >>> import numpy as np 
        >>> from .pyCGM import CGM
        >>> measurements = { 'RightStaticRotOff' : 0.015683497632642047, 'LeftStaticRotOff': 0.009402910292403012,
        ...                'RightStaticPlantFlex' : 0.2702417907002758, 'LeftStaticPlantFlex': 0.20251085737834015}
        >>> RTOE = np.array([442.81997681, 381.62280273, 42.66047668])
        >>> LTOE = np.array([39.43652725, 382.44522095, 41.78911591])
        >>> knee_JC = [np.array([364.17774614, 292.17051722, 515.19181496]),
        ...           np.array([143.55478579, 279.90370346, 524.78408753]),
        ...           np.array([[[364.64959153, 293.06758353, 515.18513093],
        ...           [363.29019771, 292.60656648, 515.04309095],
        ...           [364.04724541, 292.24216264, 516.18067112]],
        ...           [[143.65611282, 280.88685896, 524.63197541],
        ...           [142.56434499, 280.01777943, 524.86163553],
        ...           [143.64837987, 280.04650381, 525.76940383]]])]
        >>> ankle_JC = [np.array([393.76181608, 247.67829633, 87.73775041]),
        ...            np.array([98.74901939, 219.46930221, 80.6306816]),
        ...            [[np.array([394.4817575, 248.37201348, 87.715368]),
        ...            np.array([393.07114384, 248.39110006, 87.61575574]),
        ...            np.array([393.69314056, 247.78157916, 88.73002876])],
        ...            [np.array([98.47494966, 220.42553803, 80.52821783]),
        ...            np.array([97.79246671, 219.20927275, 80.76255901]),
        ...            np.array([98.84848169, 219.60345781, 81.61663775])]]]
        >>> delta = 0
        >>> [np.around(arr,8) for arr in CGM.foot_axis_calc(RTOE, LTOE, measurements, ankle_JC, knee_JC, delta)] #doctest: +NORMALIZE_WHITESPACE
        [array([442.81997681, 381.62280273,  42.66047668]), 
        array([ 39.43652725, 382.44522095,  41.78911591]), 
        array([[[442.84624127, 381.6513024 ,  43.65972537],
                [441.87735057, 381.9563035 ,  42.67574106],
                [442.48716163, 380.68048378,  42.69610043]],
            [[ 39.56652626, 382.50901001,  42.77857597],
                [ 38.49313328, 382.14606841,  41.93234851],
                [ 39.74166341, 381.4931502 ,  41.81040459]]])]
        """
        pass

    @staticmethod
    def head_axis_calc():
        pass

    @staticmethod
    def thorax_axis_calc():
        pass

    @staticmethod
    def neck_axis_calc():
        pass

    @staticmethod
    def shoulder_axis_calc():
        pass

    @staticmethod
    def elbow_axis_calc():
        pass

    @staticmethod
    def wrist_axis_calc():
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
    def multi_calc():
        pass

    @staticmethod
    def calc():
        pass
