import numpy as np
import math


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
        c = [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

        return c

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
        try:
            return math.sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2]))
        except:
            return np.nan 

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
        try:
            return np.asarray(math.sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
        except:
            return np.nan
    
    @staticmethod
    def find_joint_c(a, b, c, delta):
        """Calculate the Joint Center function.
        
        This function is based on physical markers, a,b,c and joint center which will be calulcated in this function are all in the same plane.

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

        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> a = [468.14532471, 325.09780884, 673.12591553]
        >>> b = [355.90861996, 365.38260964, 940.6974861]
        >>> c = [452.35180664, 329.0609436, 524.77893066]
        >>> delta = 59.5 
        >>> CGM.find_joint_c(a, b, c, delta)
        array([396.25286248, 347.91367254, 518.63620527])
        """
        # make the two vector using 3 markers, which is on the same plane.
        v1 = (a[0]-c[0],a[1]-c[1],a[2]-c[2])
        v2 = (b[0]-c[0],b[1]-c[1],b[2]-c[2])
    
        # v3 is cross vector of v1, v2
        # and then it normalized.
        # v3 = cross(v1,v2)
        v3 = [v1[1]*v2[2] - v1[2]*v2[1],v1[2]*v2[0] - v1[0]*v2[2],v1[0]*v2[1] - v1[1]*v2[0]]
        v3_div = CGM.norm2d(v3)
        v3 = [v3[0]/v3_div,v3[1]/v3_div,v3[2]/v3_div]
        
        m = [(b[0]+c[0])/2.0,(b[1]+c[1])/2.0,(b[2]+c[2])/2.0]
        length = np.subtract(b,m)
        length = CGM.norm2d(length)

        theta = math.acos(delta/CGM.norm2d(v2))

        cs = math.cos(theta*2)
        sn = math.sin(theta*2)

        ux, uy, uz = v3
        
        # this rotation matrix is called Rodriques' rotation formula.
        # In order to make a plane, at least 3 number of markers is required which means three physical markers on the segment can make a plane. 
        # then the orthogonal vector of the plane will be rotating axis.
        # joint center is determined by rotating the one vector of plane around rotating axis.

        rot = np.matrix([[cs+ux**2.0*(1.0-cs),ux*uy*(1.0-cs)-uz*sn,ux*uz*(1.0-cs)+uy*sn],
                        [uy*ux*(1.0-cs)+uz*sn,cs+uy**2.0*(1.0-cs),uy*uz*(1.0-cs)-ux*sn],
                        [uz*ux*(1.0-cs)-uy*sn,uz*uy*(1.0-cs)+ux*sn,cs+uz**2.0*(1.0-cs)]])  
        r = rot*(np.matrix(v2).transpose())   
        r = r* length/np.linalg.norm(r)

        r = [r[0,0],r[1,0],r[2,0]]
        mr = np.array([r[0]+m[0],r[1]+m[1],r[2]+m[2]])

        return mr

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
        c = [[0 for row in range(len(a))] for col in range(len(b[0]))]
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    c[i][j] += a[i][k]*b[k][j]
        return c

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
        #TODO-G add better doctests for this function
        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)
        Rx = [ [1,0,0],[0,math.cos(x),math.sin(x)*-1],[0,math.sin(x),math.cos(x)] ]
        Ry = [ [math.cos(y),0,math.sin(y)],[0,1,0],[math.sin(y)*-1,0,math.cos(y)] ]
        Rz = [ [math.cos(z),math.sin(z)*-1,0],[math.sin(z),math.cos(z),0],[0,0,1] ]
        Rxy = CGM.matrix_mult(Rx,Ry)
        Rxyz = CGM.matrix_mult(Rxy,Rz)    
        
        Ryx = CGM.matrix_mult(Ry,Rx)
        Ryxz = CGM.matrix_mult(Ryx,Rz)
        
        return Rxyz

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
        >>> from .pycgm import CGM
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
        >>> RASI = np.array([ 395.36532593,  428.09790039, 1036.82763672]) 
        >>> LASI = np.array([ 183.18504333,  422.78927612, 1033.07299805])
        >>> SACR = np.array([ 294.60904694,  242.07158661, 1049.64605713])
        >>> CGM.pelvis_axis_calc(RASI, LASI, SACR=SACR) #doctest: +NORMALIZE_WHITESPACE
        [array([ 289.27518463,  425.44358826, 1034.95031739]), 
        array([[ 289.25166321,  426.44012508, 1034.87056085],
        [ 288.27565385,  425.41858059, 1034.93263018],
        [ 289.25556415,  425.52289134, 1035.94697483]]), 
        array([ 294.60904694,  242.07158661, 1049.64605713])]
        """
        #REQUIRED MARKERS: 
        # RASI
        # LASI 
        # RPSI 
        # LPSI
        
        try:
            #  If no sacrum, mean of posterior markers is used as the sacrum
            sacrum = (RPSI+LPSI)/2.0
        except:
            pass #going to use sacrum marker

        #  If no sacrum, mean of posterior markers is used as the sacrum
        if SACR is not None:
            sacrum = SACR 
            
        # REQUIRED LANDMARKS:
        # origin
        # sacrum 
        
        # Origin is Midpoint between RASI and LASI
        origin = (RASI+LASI)/2.0
            
        # This calculate the each axis
        # beta1,2,3 is arbitrary name to help calculate.
        beta1 = origin-sacrum
        beta2 = LASI-RASI
        
        # Y_axis is normalized beta2
        y_axis = beta2/CGM.norm3d(beta2)

        # X_axis computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990)
        # and then normalized.
        beta3_cal = np.dot(beta1,y_axis)
        beta3_cal2 = beta3_cal*y_axis
        beta3 = beta1-beta3_cal2
        x_axis = beta3/CGM.norm3d(beta3)

        # Z-axis is cross product of x_axis and y_axis.
        z_axis = CGM.cross(x_axis,y_axis)

        # Add the origin back to the vector 
        y_axis = y_axis+origin
        z_axis = z_axis+origin
        x_axis = x_axis+origin

        pelvis_axis = np.asarray([x_axis,y_axis,z_axis])

        pelvis = [origin,pelvis_axis,sacrum] #probably don't need to return sacrum

        return pelvis

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
        >>> from .pycgm import CGM
        >>> measurements = { 'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
        ...                'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031 }
        >>> pel_origin = [ 251.60830688, 391.74131775, 1032.89349365]
        >>> pel_x = [251.74063624, 392.72694721, 1032.78850073]
        >>> pel_y = [250.61711554, 391.87232862, 1032.8741063]
        >>> pel_z = [251.60295336, 391.84795134, 1033.88777762]
        >>> pel = [pel_origin, [pel_x, pel_y, pel_z]]
        >>> CGM.hip_axis_calc(pel, measurements) #doctest: +NORMALIZE_WHITESPACE
        array([[182.57097799, 339.43231799, 935.52900136],
            [308.38050352, 322.80342433, 937.98979092]])

        """
        # TODO, implement pel = [pel_origin, [pel_x, pel_y, pel_z]]
        # before calling this function

        #TODO should most likely remove these np.asarray calls as well
        pel_origin=np.asarray(pel[0])
        pel_x=np.asarray(pel[1][0])
        pel_y=np.asarray(pel[1][1])
        pel_z=np.asarray(pel[1][2])

        # Model's eigen value
        #
        # LegLength
        # MeanLegLength
        # mm (marker radius)
        # interAsisMeasure
        
        #Set the variables needed to calculate the joint angle
        #Half of marker size
        mm = 7.0

        MeanLegLength = measurements['MeanLegLength']
        R_AsisToTrocanterMeasure = measurements['R_AsisToTrocanterMeasure']
        L_AsisToTrocanterMeasure = measurements['L_AsisToTrocanterMeasure']
        interAsisMeasure = measurements['InterAsisDistance']
        C = ( MeanLegLength * 0.115 ) - 15.3
        theta = 0.500000178813934
        beta = 0.314000427722931
        aa = interAsisMeasure/2.0
        S = -1

        # Hip Joint Center Calculation (ref. Davis_1991)
        
        # Left: Calculate the distance to translate along the pelvis axis
        L_Xh = (-L_AsisToTrocanterMeasure - mm) * math.cos(beta) + C * math.cos(theta) * math.sin(beta)
        L_Yh = S*(C*math.sin(theta)- aa)
        L_Zh = (-L_AsisToTrocanterMeasure - mm) * math.sin(beta) - C * math.cos(theta) * math.cos(beta)
        
        # Right:  Calculate the distance to translate along the pelvis axis
        R_Xh = (-R_AsisToTrocanterMeasure - mm) * math.cos(beta) + C * math.cos(theta) * math.sin(beta)
        R_Yh = (C*math.sin(theta)- aa)
        R_Zh = (-R_AsisToTrocanterMeasure - mm) * math.sin(beta) - C * math.cos(theta) * math.cos(beta)
        
        # get the unit pelvis axis
        pelvis_xaxis = pel_x-pel_origin
        pelvis_yaxis = pel_y-pel_origin
        pelvis_zaxis = pel_z-pel_origin
        
        # multiply the distance to the unit pelvis axis
        L_hipJCx = pelvis_xaxis*L_Xh
        L_hipJCy = pelvis_yaxis*L_Yh
        L_hipJCz = pelvis_zaxis*L_Zh
        L_hipJC = np.asarray([  L_hipJCx[0]+L_hipJCy[0]+L_hipJCz[0],
                                L_hipJCx[1]+L_hipJCy[1]+L_hipJCz[1],
                                L_hipJCx[2]+L_hipJCy[2]+L_hipJCz[2]])         

        R_hipJCx = pelvis_xaxis*R_Xh
        R_hipJCy = pelvis_yaxis*R_Yh
        R_hipJCz = pelvis_zaxis*R_Zh
        R_hipJC = np.asarray([  R_hipJCx[0]+R_hipJCy[0]+R_hipJCz[0],
                                R_hipJCx[1]+R_hipJCy[1]+R_hipJCz[1],
                                R_hipJCx[2]+R_hipJCy[2]+R_hipJCz[2]])
                                
        L_hipJC = L_hipJC+pel_origin
        R_hipJC = R_hipJC+pel_origin

        hip_JC = np.asarray([L_hipJC,R_hipJC])
        
        return hip_JC

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
        >>> from .pycgm import CGM
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
        #TODO have to combine hipAxisCenter with this one, which is defined as hipJointCenter
        #Get Global Values
        mm = 7.0
        R_kneeWidth = measurements['RightKneeWidth']
        L_kneeWidth = measurements['LeftKneeWidth']
        R_delta = (R_kneeWidth/2.0)+mm
        L_delta = (L_kneeWidth/2.0)+mm
        
        #REQUIRED MARKERS: 
        # RTHI
        # LTHI
        # RKNE 
        # LKNE 
        # hip_JC

        R_hip_JC = hip_JC[1]
        L_hip_JC = hip_JC[0]
        
        # Determine the position of kneeJointCenter using findJointC function   
        R = CGM.find_joint_c(RTHI,R_hip_JC,RKNE,R_delta)
        L = CGM.find_joint_c(LTHI,L_hip_JC,LKNE,L_delta) 
        
        # Knee Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
        #Right axis calculation
        
        thi_kne_R = RTHI-RKNE #TODO-G, this is unused
        
        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = R_hip_JC-R
        
        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector. 
        # the axis is then normalized.
        # axis_x = cross(axis_z,thi_kne_R)
        axis_x = CGM.cross(axis_z,RKNE-R_hip_JC)
        
        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = CGM.cross(axis_z,axis_x)

        Raxis = np.asarray([axis_x,axis_y,axis_z])
        
        #Left axis calculation
    
        thi_kne_L = LTHI-LKNE #TODO-G, this is unused

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = L_hip_JC-L
        
        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector cross vector. 
        # the axis is then normalized.
        # axis_x = cross(thi_kne_L,axis_z)
        #using hipjc instead of thigh marker
        axis_x = CGM.cross(LKNE-L_hip_JC,axis_z)
        
        # Y axis is determined by cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = CGM.cross(axis_z,axis_x)
    
        Laxis = np.asarray([axis_x,axis_y,axis_z])
        
        # Clear the name of axis and then nomalize it.
        R_knee_x_axis = Raxis[0]
        R_knee_x_axis = R_knee_x_axis/CGM.norm3d(R_knee_x_axis)
        R_knee_y_axis = Raxis[1]
        R_knee_y_axis = R_knee_y_axis/CGM.norm3d(R_knee_y_axis)
        R_knee_z_axis = Raxis[2]
        R_knee_z_axis = R_knee_z_axis/CGM.norm3d(R_knee_z_axis)
        L_knee_x_axis = Laxis[0]
        L_knee_x_axis = L_knee_x_axis/CGM.norm3d(L_knee_x_axis)
        L_knee_y_axis = Laxis[1]
        L_knee_y_axis = L_knee_y_axis/CGM.norm3d(L_knee_y_axis)
        L_knee_z_axis = Laxis[2]
        L_knee_z_axis = L_knee_z_axis/CGM.norm3d(L_knee_z_axis)
        
        #Put both axis in array
        # Add the origin back to the vector 
        y_axis = R_knee_y_axis+R
        z_axis = R_knee_z_axis+R
        x_axis = R_knee_x_axis+R
        Raxis = np.asarray([x_axis,y_axis,z_axis])
    
        # Add the origin back to the vector 
        y_axis = L_knee_y_axis+L
        z_axis = L_knee_z_axis+L
        x_axis = L_knee_x_axis+L
        Laxis = np.asarray([x_axis,y_axis,z_axis])

        axis = np.asarray([Raxis,Laxis])

        return [R,L,axis]

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
        >>> from .pycgm import CGM
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
        # Get subject measurements
        R_ankleWidth = measurements['RightAnkleWidth']
        L_ankleWidth = measurements['LeftAnkleWidth']
        R_torsion = measurements['RightTibialTorsion']
        L_torsion = measurements['LeftTibialTorsion']
        mm = 7.0
        R_delta = ((R_ankleWidth)/2.0)+mm
        L_delta = ((L_ankleWidth)/2.0)+mm
    
        #REQUIRED MARKERS: 
        # tib_R
        # tib_L
        # ank_R 
        # ank_L  
        # knee_JC
                
        tib_R = RTIB           
        tib_L = LTIB           
        ank_R = RANK           
        ank_L = LANK

        knee_JC_R = knee_JC[0]
        knee_JC_L = knee_JC[1]
        
        # This is Torsioned Tibia and this describe the ankle angles
        # Tibial frontal plane being defined by ANK,TIB and KJC
        
        # Determine the position of ankleJointCenter using findJointC function     
        R = CGM.find_joint_c(tib_R, knee_JC_R, ank_R, R_delta)
        L = CGM.find_joint_c(tib_L, knee_JC_L, ank_L, L_delta)
                                
        # Ankle Axis Calculation(ref. Clinical Gait Analysis hand book, Baker2013)
            #Right axis calculation
            
        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = knee_JC_R-R

        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector. 
        # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_R = tib_R-ank_R
        axis_x = CGM.cross(axis_z,tib_ank_R)

        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = CGM.cross(axis_z,axis_x)
        
        Raxis = [axis_x,axis_y,axis_z]
            
            #Left axis calculation
            
        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = knee_JC_L-L
        
        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector cross vector. 
        # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_L = tib_L-ank_L
        axis_x = CGM.cross(tib_ank_L,axis_z)    
        
        # Y axis is determined by cross product of axis_z and axis_x.
        axis_y = CGM.cross(axis_z,axis_x)  
    
        Laxis = [axis_x,axis_y,axis_z]

        # Clear the name of axis and then normalize it.
        R_ankle_x_axis = Raxis[0]
        R_ankle_x_axis_div = CGM.norm2d(R_ankle_x_axis)
        R_ankle_x_axis = [R_ankle_x_axis[0]/R_ankle_x_axis_div,R_ankle_x_axis[1]/R_ankle_x_axis_div,R_ankle_x_axis[2]/R_ankle_x_axis_div]
        
        R_ankle_y_axis = Raxis[1]
        R_ankle_y_axis_div = CGM.norm2d(R_ankle_y_axis)
        R_ankle_y_axis = [R_ankle_y_axis[0]/R_ankle_y_axis_div,R_ankle_y_axis[1]/R_ankle_y_axis_div,R_ankle_y_axis[2]/R_ankle_y_axis_div]
        
        R_ankle_z_axis = Raxis[2]
        R_ankle_z_axis_div = CGM.norm2d(R_ankle_z_axis)
        R_ankle_z_axis = [R_ankle_z_axis[0]/R_ankle_z_axis_div,R_ankle_z_axis[1]/R_ankle_z_axis_div,R_ankle_z_axis[2]/R_ankle_z_axis_div]
        
        L_ankle_x_axis = Laxis[0]
        L_ankle_x_axis_div = CGM.norm2d(L_ankle_x_axis)
        L_ankle_x_axis = [L_ankle_x_axis[0]/L_ankle_x_axis_div,L_ankle_x_axis[1]/L_ankle_x_axis_div,L_ankle_x_axis[2]/L_ankle_x_axis_div]
        
        L_ankle_y_axis = Laxis[1]
        L_ankle_y_axis_div = CGM.norm2d(L_ankle_y_axis)
        L_ankle_y_axis = [L_ankle_y_axis[0]/L_ankle_y_axis_div,L_ankle_y_axis[1]/L_ankle_y_axis_div,L_ankle_y_axis[2]/L_ankle_y_axis_div]
        
        L_ankle_z_axis = Laxis[2]
        L_ankle_z_axis_div = CGM.norm2d(L_ankle_z_axis)
        L_ankle_z_axis = [L_ankle_z_axis[0]/L_ankle_z_axis_div,L_ankle_z_axis[1]/L_ankle_z_axis_div,L_ankle_z_axis[2]/L_ankle_z_axis_div]
        

        #Put both axis in array
        Raxis = [R_ankle_x_axis,R_ankle_y_axis,R_ankle_z_axis]
        Laxis = [L_ankle_x_axis,L_ankle_y_axis,L_ankle_z_axis]
        
        # Rotate the axes about the tibia torsion.
        R_torsion = np.radians(R_torsion)
        L_torsion = np.radians(L_torsion)
        
        Raxis = [[math.cos(R_torsion)*Raxis[0][0]-math.sin(R_torsion)*Raxis[1][0],
                math.cos(R_torsion)*Raxis[0][1]-math.sin(R_torsion)*Raxis[1][1],
                math.cos(R_torsion)*Raxis[0][2]-math.sin(R_torsion)*Raxis[1][2]],
                [math.sin(R_torsion)*Raxis[0][0]+math.cos(R_torsion)*Raxis[1][0],
                math.sin(R_torsion)*Raxis[0][1]+math.cos(R_torsion)*Raxis[1][1],
                math.sin(R_torsion)*Raxis[0][2]+math.cos(R_torsion)*Raxis[1][2]],
                [Raxis[2][0],Raxis[2][1],Raxis[2][2]]]
            
        Laxis = [[math.cos(L_torsion)*Laxis[0][0]-math.sin(L_torsion)*Laxis[1][0],
                math.cos(L_torsion)*Laxis[0][1]-math.sin(L_torsion)*Laxis[1][1],
                math.cos(L_torsion)*Laxis[0][2]-math.sin(L_torsion)*Laxis[1][2]],
                [math.sin(L_torsion)*Laxis[0][0]+math.cos(L_torsion)*Laxis[1][0],
                math.sin(L_torsion)*Laxis[0][1]+math.cos(L_torsion)*Laxis[1][1],
                math.sin(L_torsion)*Laxis[0][2]+math.cos(L_torsion)*Laxis[1][2]],
                [Laxis[2][0],Laxis[2][1],Laxis[2][2]]]
        
        # Add the origin back to the vector 
        x_axis = Raxis[0]+R
        y_axis = Raxis[1]+R
        z_axis = Raxis[2]+R
        Raxis = [x_axis,y_axis,z_axis]
        
        x_axis = Laxis[0]+L
        y_axis = Laxis[1]+L
        z_axis = Laxis[2]+L
        Laxis = [x_axis,y_axis,z_axis]
        
        # Both of axis in array.
        axis = [Raxis,Laxis]

        return [R,L,axis]

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
        >>> from .pycgm import CGM
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
        #REQUIRED MARKERS: 
        # RTOE
        # LTOE
        
        TOE_R = RTOE           
        TOE_L = LTOE          
        
        #REQUIRE JOINT CENTER & AXIS
        #KNEE JOINT CENTER
        #ANKLE JOINT CENTER
        #ANKLE FLEXION AXIS
        
        ankle_JC_R = ankle_JC[0]
        ankle_JC_L = ankle_JC[1]
        ankle_flexion_R = ankle_JC[2][0][1]
        ankle_flexion_L = ankle_JC[2][1][1]
    
        # Toe axis's origin is marker position of TOE
        R = TOE_R
        L = TOE_L
        
        # HERE IS THE INCORRECT AXIS
        
        # the first setting, the foot axis show foot uncorrected anatomical axis and static_info is None
        ankle_JC_R = [ankle_JC_R[0],ankle_JC_R[1],ankle_JC_R[2]]
        ankle_JC_L = [ankle_JC_L[0],ankle_JC_L[1],ankle_JC_L[2]]
        
        # Right
        
        # z axis is from TOE marker to AJC. and normalized it.
        R_axis_z = [ankle_JC_R[0]-TOE_R[0],ankle_JC_R[1]-TOE_R[1],ankle_JC_R[2]-TOE_R[2]]
        R_axis_z_div = CGM.norm2d(R_axis_z)
        R_axis_z = [R_axis_z[0]/R_axis_z_div,R_axis_z[1]/R_axis_z_div,R_axis_z[2]/R_axis_z_div]
        
        # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
        y_flex_R = [ankle_flexion_R[0]-ankle_JC_R[0],ankle_flexion_R[1]-ankle_JC_R[1],ankle_flexion_R[2]-ankle_JC_R[2]]
        y_flex_R_div = CGM.norm2d(y_flex_R)
        y_flex_R = [y_flex_R[0]/y_flex_R_div,y_flex_R[1]/y_flex_R_div,y_flex_R[2]/y_flex_R_div]
        
        # x axis is calculated as a cross product of z axis and ankle flexion axis.
        R_axis_x = CGM.cross(y_flex_R,R_axis_z)
        R_axis_x_div = CGM.norm2d(R_axis_x)
        R_axis_x = [R_axis_x[0]/R_axis_x_div,R_axis_x[1]/R_axis_x_div,R_axis_x[2]/R_axis_x_div]
        
        # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
        R_axis_y = CGM.cross(R_axis_z,R_axis_x)
        R_axis_y_div = CGM.norm2d(R_axis_y)
        R_axis_y = [R_axis_y[0]/R_axis_y_div,R_axis_y[1]/R_axis_y_div,R_axis_y[2]/R_axis_y_div]
            
        R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]
    
        # Left

        # z axis is from TOE marker to AJC. and normalized it.
        L_axis_z = [ankle_JC_L[0]-TOE_L[0],ankle_JC_L[1]-TOE_L[1],ankle_JC_L[2]-TOE_L[2]]
        L_axis_z_div = CGM.norm2d(L_axis_z)
        L_axis_z = [L_axis_z[0]/L_axis_z_div,L_axis_z[1]/L_axis_z_div,L_axis_z[2]/L_axis_z_div]
        
        # bring the flexion axis of ankle axes from AnkleJointCenter function. and normalized it.
        y_flex_L = [ankle_flexion_L[0]-ankle_JC_L[0],ankle_flexion_L[1]-ankle_JC_L[1],ankle_flexion_L[2]-ankle_JC_L[2]]
        y_flex_L_div = CGM.norm2d(y_flex_L)
        y_flex_L = [y_flex_L[0]/y_flex_L_div,y_flex_L[1]/y_flex_L_div,y_flex_L[2]/y_flex_L_div]
        
        # x axis is calculated as a cross product of z axis and ankle flexion axis.
        L_axis_x = CGM.cross(y_flex_L,L_axis_z)
        L_axis_x_div = CGM.norm2d(L_axis_x)
        L_axis_x = [L_axis_x[0]/L_axis_x_div,L_axis_x[1]/L_axis_x_div,L_axis_x[2]/L_axis_x_div]

        # y axis is then perpendicularly calculated from z axis and x axis. and normalized.
        L_axis_y = CGM.cross(L_axis_z,L_axis_x)
        L_axis_y_div = CGM.norm2d(L_axis_y)
        L_axis_y = [L_axis_y[0]/L_axis_y_div,L_axis_y[1]/L_axis_y_div,L_axis_y[2]/L_axis_y_div]
    
        L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]

        foot_axis = [R_foot_axis,L_foot_axis]
        
        # Apply static offset angle to the incorrect foot axes 
        
        # static offset angle are taken from static_info variable in radians.
        R_alpha = measurements['RightStaticRotOff']
        R_beta = measurements['RightStaticPlantFlex']
        #R_gamma = static_info[0][2]
        L_alpha = measurements['LeftStaticRotOff']
        L_beta = measurements['LeftStaticPlantFlex']
        #L_gamma = static_info[1][2]
    
        R_alpha = np.around(math.degrees(R_alpha),decimals=5)
        R_beta = np.around(math.degrees(R_beta),decimals=5)
        #R_gamma = np.around(math.degrees(static_info[0][2]),decimals=5)
        L_alpha = np.around(math.degrees(L_alpha),decimals=5)
        L_beta = np.around(math.degrees(L_beta),decimals=5)
        #L_gamma = np.around(math.degrees(static_info[1][2]),decimals=5)
        
        R_alpha = -math.radians(R_alpha)
        R_beta = math.radians(R_beta)
        #R_gamma = 0
        L_alpha = math.radians(L_alpha)
        L_beta = math.radians(L_beta)
        #L_gamma = 0
        
        R_axis = [[(R_foot_axis[0][0]),(R_foot_axis[0][1]),(R_foot_axis[0][2])],
                [(R_foot_axis[1][0]),(R_foot_axis[1][1]),(R_foot_axis[1][2])],
                [(R_foot_axis[2][0]),(R_foot_axis[2][1]),(R_foot_axis[2][2])]]
                
        L_axis = [[(L_foot_axis[0][0]),(L_foot_axis[0][1]),(L_foot_axis[0][2])],
                [(L_foot_axis[1][0]),(L_foot_axis[1][1]),(L_foot_axis[1][2])],
                [(L_foot_axis[2][0]),(L_foot_axis[2][1]),(L_foot_axis[2][2])]]
        
        # rotate incorrect foot axis around y axis first.
        
        # right
        R_rotmat = [[(math.cos(R_beta)*R_axis[0][0]+math.sin(R_beta)*R_axis[2][0]),
                    (math.cos(R_beta)*R_axis[0][1]+math.sin(R_beta)*R_axis[2][1]),
                    (math.cos(R_beta)*R_axis[0][2]+math.sin(R_beta)*R_axis[2][2])],
                    [R_axis[1][0],R_axis[1][1],R_axis[1][2]],
                    [(-1*math.sin(R_beta)*R_axis[0][0]+math.cos(R_beta)*R_axis[2][0]),
                    (-1*math.sin(R_beta)*R_axis[0][1]+math.cos(R_beta)*R_axis[2][1]),
                    (-1*math.sin(R_beta)*R_axis[0][2]+math.cos(R_beta)*R_axis[2][2])]]
        # left
        L_rotmat = [[(math.cos(L_beta)*L_axis[0][0]+math.sin(L_beta)*L_axis[2][0]),
                    (math.cos(L_beta)*L_axis[0][1]+math.sin(L_beta)*L_axis[2][1]),
                    (math.cos(L_beta)*L_axis[0][2]+math.sin(L_beta)*L_axis[2][2])],
                    [L_axis[1][0],L_axis[1][1],L_axis[1][2]],
                    [(-1*math.sin(L_beta)*L_axis[0][0]+math.cos(L_beta)*L_axis[2][0]),
                    (-1*math.sin(L_beta)*L_axis[0][1]+math.cos(L_beta)*L_axis[2][1]),
                    (-1*math.sin(L_beta)*L_axis[0][2]+math.cos(L_beta)*L_axis[2][2])]]
                    
        # rotate incorrect foot axis around x axis next.
        
        # right
        R_rotmat = [[R_rotmat[0][0],R_rotmat[0][1],R_rotmat[0][2]],
                    [(math.cos(R_alpha)*R_rotmat[1][0]-math.sin(R_alpha)*R_rotmat[2][0]),
                    (math.cos(R_alpha)*R_rotmat[1][1]-math.sin(R_alpha)*R_rotmat[2][1]),
                    (math.cos(R_alpha)*R_rotmat[1][2]-math.sin(R_alpha)*R_rotmat[2][2])],
                    [(math.sin(R_alpha)*R_rotmat[1][0]+math.cos(R_alpha)*R_rotmat[2][0]),
                    (math.sin(R_alpha)*R_rotmat[1][1]+math.cos(R_alpha)*R_rotmat[2][1]),
                    (math.sin(R_alpha)*R_rotmat[1][2]+math.cos(R_alpha)*R_rotmat[2][2])]]
        
        # left          
        L_rotmat = [[L_rotmat[0][0],L_rotmat[0][1],L_rotmat[0][2]],
                    [(math.cos(L_alpha)*L_rotmat[1][0]-math.sin(L_alpha)*L_rotmat[2][0]),
                    (math.cos(L_alpha)*L_rotmat[1][1]-math.sin(L_alpha)*L_rotmat[2][1]),
                    (math.cos(L_alpha)*L_rotmat[1][2]-math.sin(L_alpha)*L_rotmat[2][2])],
                    [(math.sin(L_alpha)*L_rotmat[1][0]+math.cos(L_alpha)*L_rotmat[2][0]),
                    (math.sin(L_alpha)*L_rotmat[1][1]+math.cos(L_alpha)*L_rotmat[2][1]),
                    (math.sin(L_alpha)*L_rotmat[1][2]+math.cos(L_alpha)*L_rotmat[2][2])]]
        
        # Bring each x,y,z axis from rotation axes
        R_axis_x = R_rotmat[0]
        R_axis_y = R_rotmat[1]
        R_axis_z = R_rotmat[2]
        L_axis_x = L_rotmat[0]
        L_axis_y = L_rotmat[1]
        L_axis_z = L_rotmat[2]

        # Attach each axis to the origin
        R_axis_x = [R_axis_x[0]+R[0],R_axis_x[1]+R[1],R_axis_x[2]+R[2]]
        R_axis_y = [R_axis_y[0]+R[0],R_axis_y[1]+R[1],R_axis_y[2]+R[2]]
        R_axis_z = [R_axis_z[0]+R[0],R_axis_z[1]+R[1],R_axis_z[2]+R[2]]
        
        R_foot_axis = [R_axis_x,R_axis_y,R_axis_z]

        L_axis_x = [L_axis_x[0]+L[0],L_axis_x[1]+L[1],L_axis_x[2]+L[2]]
        L_axis_y = [L_axis_y[0]+L[0],L_axis_y[1]+L[1],L_axis_y[2]+L[2]]
        L_axis_z = [L_axis_z[0]+L[0],L_axis_z[1]+L[1],L_axis_z[2]+L[2]]
        
        L_foot_axis = [L_axis_x,L_axis_y,L_axis_z]
        
        foot_axis = [R_foot_axis,L_foot_axis]

        return [R,L,foot_axis]

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
