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
    
        # v3 is CGM.cross vector of v1, v2
        # and then it normalized.
        # v3 = CGM.cross(v1,v2)
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

        Calculates the pelvis joint center and axis and returns both.

        Markers used: RASI,LASI,RPSI,LPSI
        Other landmarks used: origin, sacrum

        Pelvis X_axis: Computed with a Gram-Schmidt orthogonalization procedure(ref. Kadaba 1990) and then normalized.
        Pelvis Y_axis: LASI-RASI x,y,z positions, then normalized.
        Pelvis Z_axis: CGM.cross product of x_axis and y_axis.

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

        # Z-axis is CGM.cross product of x_axis and y_axis.
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
            A dictionary containing the subject measurements given from the file input.  

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
            A dictionary containing the subject measurements given from the file input.  

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
        # and calculated by each point's vector CGM.cross vector. 
        # the axis is then normalized.
        # axis_x = CGM.cross(axis_z,thi_kne_R)
        axis_x = CGM.cross(axis_z,RKNE-R_hip_JC)
        
        # Y axis is determined by CGM.cross product of axis_z and axis_x.
        # the axis is then normalized.
        axis_y = CGM.cross(axis_z,axis_x)

        Raxis = np.asarray([axis_x,axis_y,axis_z])
        
        #Left axis calculation
    
        thi_kne_L = LTHI-LKNE #TODO-G, this is unused

        # Z axis is Thigh bone calculated by the hipJC and  kneeJC
        # the axis is then normalized
        axis_z = L_hip_JC-L
        
        # X axis is perpendicular to the points plane which is determined by KJC, HJC, KNE markers.
        # and calculated by each point's vector CGM.cross vector. 
        # the axis is then normalized.
        # axis_x = CGM.cross(thi_kne_L,axis_z)
        #using hipjc instead of thigh marker
        axis_x = CGM.cross(LKNE-L_hip_JC,axis_z)
        
        # Y axis is determined by CGM.cross product of axis_z and axis_x.
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
            A dictionary containing the subject measurements given from the file input. 

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
        # and calculated by each point's vector CGM.cross vector. 
        # tib_ank_R vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_R = tib_R-ank_R
        axis_x = CGM.cross(axis_z,tib_ank_R)

        # Y axis is determined by CGM.cross product of axis_z and axis_x.
        axis_y = CGM.cross(axis_z,axis_x)
        
        Raxis = [axis_x,axis_y,axis_z]
            
            #Left axis calculation
            
        # Z axis is shank bone calculated by the ankleJC and  kneeJC
        axis_z = knee_JC_L-L
        
        # X axis is perpendicular to the points plane which is determined by ANK,TIB and KJC markers.
        # and calculated by each point's vector CGM.cross vector. 
        # tib_ank_L vector is making a tibia plane to be assumed as rigid segment.
        tib_ank_L = tib_L-ank_L
        axis_x = CGM.cross(tib_ank_L,axis_z)    
        
        # Y axis is determined by CGM.cross product of axis_z and axis_x.
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
            A dictionary containing the subject measurements given from the file input. 
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
        
        # x axis is calculated as a CGM.cross product of z axis and ankle flexion axis.
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
        
        # x axis is calculated as a CGM.cross product of z axis and ankle flexion axis.
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
    def head_axis_calc(LFHD, RFHD, LBHD, RBHD, measurements):
        """The Head Axis Calculation function

        Calculates the head joint center and returns the head joint center and axis.
        
        Markers used: LFHD, RFHD, LBHD, RBHD
        Subject Measurement values used: HeadOffset

        Parameters
        ----------
        LFHD, RFHD, LBHD, RBHD : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        measurements : dict
            A dictionary containing the subject measurements given from the file input. 
        
        Returns
        -------
        head_axis, origin : array
            Returns an array containing a 1x3x3 array containing the x, y, z axis 
            components of the head joint center, and a 1x3 array containing the 
            head origin x, y, z position.
            
        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> measurements = { 'HeadOffset': 0.2571990469310653 }
        >>> RFHD = np.array([325.82983398, 402.55450439, 1722.49816895])
        >>> LFHD = np.array([184.55158997, 409.68713379, 1721.34289551])
        >>> RBHD = np.array([304.39898682, 242.91339111, 1694.97497559])
        >>> LBHD = np.array([197.8621521, 251.28889465, 1696.90197754])
        >>> [np.around(arr,8) for arr in CGM.head_axis_calc(LFHD, RFHD, LBHD, RBHD, measurements)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ 255.21685583,  407.11593888, 1721.82538439],
            [ 254.19105385,  406.14680918, 1721.91767712],
            [ 255.1903437 ,  406.21600904, 1722.91599129]]), 
        array([ 255.19071198,  406.12081909, 1721.92053223])]
        """
        #Get Global Values
        head_off = measurements['HeadOffset']
        head_off = -1*head_off

        #get the midpoints of the head to define the sides
        front = [(LFHD[0]+RFHD[0])/2.0, (LFHD[1]+RFHD[1])/2.0,(LFHD[2]+RFHD[2])/2.0]
        back = [(LBHD[0]+RBHD[0])/2.0, (LBHD[1]+RBHD[1])/2.0,(LBHD[2]+RBHD[2])/2.0]
        left = [(LFHD[0]+LBHD[0])/2.0, (LFHD[1]+LBHD[1])/2.0,(LFHD[2]+LBHD[2])/2.0]
        right = [(RFHD[0]+RBHD[0])/2.0, (RFHD[1]+RBHD[1])/2.0,(RFHD[2]+RBHD[2])/2.0]
        origin = front
        
        #Get the vectors from the sides with primary x axis facing front
        #First get the x direction
        x_vec = [front[0]-back[0],front[1]-back[1],front[2]-back[2]]
        x_vec_div = CGM.norm2d(x_vec)
        x_vec = [x_vec[0]/x_vec_div,x_vec[1]/x_vec_div,x_vec[2]/x_vec_div]
        
        #get the direction of the y axis
        y_vec = [left[0]-right[0],left[1]-right[1],left[2]-right[2]]
        y_vec_div = CGM.norm2d(y_vec)
        y_vec = [y_vec[0]/y_vec_div,y_vec[1]/y_vec_div,y_vec[2]/y_vec_div]
        
        # get z axis by CGM.cross-product of x axis and y axis.
        z_vec = CGM.cross(x_vec,y_vec)
        z_vec_div = CGM.norm2d(z_vec)
        z_vec = [z_vec[0]/z_vec_div,z_vec[1]/z_vec_div,z_vec[2]/z_vec_div]
        
        # make sure all x,y,z axis is orthogonal each other by CGM.cross-product
        y_vec = CGM.cross(z_vec,x_vec)
        y_vec_div = CGM.norm2d(y_vec)
        y_vec = [y_vec[0]/y_vec_div,y_vec[1]/y_vec_div,y_vec[2]/y_vec_div]
        x_vec = CGM.cross(y_vec,z_vec)
        x_vec_div = CGM.norm2d(x_vec)
        x_vec = [x_vec[0]/x_vec_div,x_vec[1]/x_vec_div,x_vec[2]/x_vec_div]
        
        # rotate the head axis around y axis about head offset angle.
        x_vec_rot = [x_vec[0]*math.cos(head_off)+z_vec[0]*math.sin(head_off),
                x_vec[1]*math.cos(head_off)+z_vec[1]*math.sin(head_off),
                x_vec[2]*math.cos(head_off)+z_vec[2]*math.sin(head_off)]
        y_vec_rot = [y_vec[0],y_vec[1],y_vec[2]]
        z_vec_rot = [x_vec[0]*-1*math.sin(head_off)+z_vec[0]*math.cos(head_off),
                x_vec[1]*-1*math.sin(head_off)+z_vec[1]*math.cos(head_off),
                x_vec[2]*-1*math.sin(head_off)+z_vec[2]*math.cos(head_off)]

        #Add the origin back to the vector to get it in the right position
        x_axis = [x_vec_rot[0]+origin[0],x_vec_rot[1]+origin[1],x_vec_rot[2]+origin[2]]
        y_axis = [y_vec_rot[0]+origin[0],y_vec_rot[1]+origin[1],y_vec_rot[2]+origin[2]]
        z_axis = [z_vec_rot[0]+origin[0],z_vec_rot[1]+origin[1],z_vec_rot[2]+origin[2]]
        
        head_axis = [x_axis,y_axis,z_axis]

        #Return the three axis and origin
        return [head_axis,origin]

    @staticmethod
    def thorax_axis_calc(CLAV, C7, STRN, T10):
        """The Thorax Axis Calculation function
        
        Calculates the thorax joint center and returns the thorax joint center and axis.
    
        Markers used: CLAV, C7, STRN, T10

        Parameters
        ----------
        CLAV, C7, STRN, T10 : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        
        Returns
        -------
        thorax_axis, origin : array
            Returns an array which contains a 2x3 array representing the right thorax joint center (1x3)
            and the left thorax joint center (1x3), which is then followed by a 6x3 array representing the
            right thorax x, y, z axis components (3x3) followed by the the left thorax x, y, z axis components (3x3).
        
        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> C7 = np.array([256.78051758, 371.28042603, 1459.70300293])
        >>> T10 = np.array([228.64323425, 192.32041931, 1279.6418457])
        >>> CLAV = np.array([256.78051758, 371.28042603, 1459.70300293])
        >>> STRN = np.array([251.67492676, 414.10391235, 1292.08508301])
        >>> [np.around(arr,8) for arr in CGM.thorax_axis_calc(CLAV, C7, STRN, T10)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ 256.34546332,  365.72239585, 1461.92089119],
            [ 257.26637166,  364.696025  , 1462.23472346],
            [ 256.18427318,  364.43288984, 1461.36304534]]), 
        array([ 256.27295428,  364.79605749, 1462.29053923])]
        """
        
        #Set or get a marker size as mm
        marker_size = (14.0) /2.0
        
        #Temporary origin since the origin will be moved at the end
        origin = CLAV

        #Get the midpoints of the upper and lower sections, as well as the front and back sections
        upper = [(CLAV[0]+C7[0])/2.0,(CLAV[1]+C7[1])/2.0,(CLAV[2]+C7[2])/2.0]
        lower = [(STRN[0]+T10[0])/2.0,(STRN[1]+T10[1])/2.0,(STRN[2]+T10[2])/2.0]
        front = [(CLAV[0]+STRN[0])/2.0,(CLAV[1]+STRN[1])/2.0,(CLAV[2]+STRN[2])/2.0]
        back = [(T10[0]+C7[0])/2.0,(T10[1]+C7[1])/2.0,(T10[2]+C7[2])/2.0]
        
        
        
        C7_CLAV = [C7[0]-CLAV[0],C7[1]-CLAV[1],C7[2]-CLAV[2]]
        C7_CLAV = C7_CLAV/CGM.norm3d(C7_CLAV)
    
        #Get the direction of the primary axis Z (facing down)
        z_direc = [lower[0]-upper[0],lower[1]-upper[1],lower[2]-upper[2]]
        z_vec = z_direc/CGM.norm3d(z_direc)
        
        #The secondary axis X is from back to front
        x_direc = [front[0]-back[0],front[1]-back[1],front[2]-back[2]]
        x_vec = x_direc/CGM.norm3d(x_direc)
        
        # make sure all the axes are orthogonal each othe by CGM.cross-product
        y_direc = CGM.cross(z_vec,x_vec)
        y_vec = y_direc/CGM.norm3d(y_direc)
        x_direc = CGM.cross(y_vec,z_vec)
        x_vec = x_direc/CGM.norm3d(x_direc)
        z_direc = CGM.cross(x_vec,y_vec)
        z_vec = z_direc/CGM.norm3d(z_direc)
        
        # move the axes about offset along the x axis.   
        offset = [x_vec[0]*marker_size,x_vec[1]*marker_size,x_vec[2]*marker_size]
        
        #Add the CLAV back to the vector to get it in the right position before translating it 
        origin = [CLAV[0]-offset[0],CLAV[1]-offset[1],CLAV[2]-offset[2]]

        # Attach all the axes to the origin.
        x_axis = [x_vec[0]+origin[0],x_vec[1]+origin[1],x_vec[2]+origin[2]]
        y_axis = [y_vec[0]+origin[0],y_vec[1]+origin[1],y_vec[2]+origin[2]]
        z_axis = [z_vec[0]+origin[0],z_vec[1]+origin[1],z_vec[2]+origin[2]]

        thorax_axis = [x_axis,y_axis,z_axis]
        
        return [thorax_axis,origin]
    
    #TODO-G - Remove (?) - Should this really be here?
    @staticmethod
    def neck_axis_calc():
        pass

    @staticmethod
    def shoulder_axis_calc(RSHO, LSHO, thorax, wand, measurements):
        """The Shoulder Axis Calculation function
        Calculate each shoulder joint center and returns it.

        Markers used: RSHO, LSHO
        Subject Measurement values used: RightShoulderOffset, LeftShoulderOffset

        Parameters
        ----------
        RSHO, LSHO : dict 
            A 1x3 array of each respective marker containing the XYZ positions.
        thorax : array
            An array containing several x,y,z markers for the thorax.
        wand : array
            An array containing two x,y,z markers for wand.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.

        Returns
        -------
        Sho_JC : array
            Returns a 2x3 array representing the right shoulder joint 
            center x, y, z, marker positions (1x3) followed by the left 
            shoulder joint center x, y, z, marker positions (1x3).
        
        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> measurements = { 'RightShoulderOffset' : 40.0, 'LeftShoulderOffset' : 40.0 }
        >>> RSHO = np.array([428.88496562, 270.552948, 1500.73010254])
        >>> LSHO = np.array([68.24668121, 269.01049805, 1510.1072998])
        >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
        ...        [257.1435863244796, 364.21960599061947, 1459.5889787129829],
        ...        [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
        ...        [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
        >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
        ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
        >>> CGM.shoulder_axis_calc(RSHO, LSHO, thorax, wand, measurements)
        [array([ 429.66971693,  275.06718208, 1453.95397769]), array([  64.51952733,  274.93442161, 1463.63133339])]
        """
        thorax_origin = thorax[1]

    
        #Get Subject Measurement Values
        R_shoulderoffset = measurements['RightShoulderOffset']
        L_shoulderoffset = measurements['LeftShoulderOffset']
        mm = 7.0
        R_delta =( R_shoulderoffset + mm ) 
        L_delta =( L_shoulderoffset + mm ) 

        
        #REQUIRED MARKERS: 
        # RSHO
        # LSHO 
        
        # Calculate the shoulder joint center first.
        R_wand = wand[0]
        L_wand = wand[1]

        R_Sho_JC = CGM.find_joint_c(R_wand,thorax_origin,RSHO,R_delta)
        L_Sho_JC = CGM.find_joint_c(L_wand,thorax_origin,LSHO,L_delta)
        Sho_JC = [R_Sho_JC,L_Sho_JC]
    
        return Sho_JC


    @staticmethod
    def elbow_axis_calc(RSHO, LSHO, RELB, LELB, RWRA , RWRB, LWRA, LWRB, thorax, shoulderJC, wand, measurements):
        """The Elbow Axis Calculation function

        Calculate each elbow joint axis and returns it.

        Markers used: RSHO, LSHO, RELB, LELB, RWRA , RWRB, LWRA, LWRB
        Subject Measurement values used: RightElbowWidth, LeftElbowWidth

        Parameters
        ---------- 
        RSHO, LSHO, RELB, LELB, RWRA , RWRB, LWRA, LWRB : array 
            A 1x3 array of each respective marker containing the XYZ positions.
        thorax : array
            An array containing the thorax joint center origin and axis. 
        shoulderJC : array
            A 2x3 array containing the x,y,z position of the right and left shoulder joint center.
        wand : array
            A 2x3 array containing the x,y,z position of the right and left wand marker.
        measurements : dict
            A dictionary containing the subject measurements given from the file input.
        
        Returns
        -------
        origin, axis, wrist_O : array 
            Returns an array containing a 2x3 array containing the right 
            elbow x, y, z marker positions (1x3), and the left elbow x, y, 
            z marker positions (1x3), which is followed by a 2x3x3 array containing 
            right elbow x, y, z axis components (1x3x3) followed by the left x, y, z axis 
            components (1x3x3) which is then followed by the right wrist joint center 
            x, y, z marker positions (1x3), and the left wrist joint center x, y, z marker positions (1x3).

        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> RSHO = np.array([428.88496562, 270.552948, 1500.73010254])
        >>> LSHO = np.array([68.24668121, 269.01049805, 1510.1072998])
        >>> RELB = np.array([658.90338135, 326.07580566, 1285.28515625])
        >>> LELB = np.array([-156.32162476, 335.2593313, 1287.39916992])
        >>> RWRA = np.array([776.51898193,495.68103027, 1108.38464355])
        >>> RWRB = np.array([830.9072876, 436.75341797, 1119.11901855])
        >>> LWRA = np.array([-249.28146362, 525.32977295, 1117.09057617])
        >>> LWRB = np.array([-311.77532959, 477.22512817, 1125.1619873])
        >>> thorax = [[[256.23991128535846, 365.30496976939753, 1459.662169500559],
        ...        [257.1435863244796, 364.21960599061947, 1459.5889787129829],
        ...        [256.08430536580352, 354.32180498523223, 1458.6575930699294]],
        ...        [256.14981023656401, 364.30906039339868, 1459.6553639290375]]
        >>> shoulderJC = [np.array([429.66951995, 275.06718615, 1453.953978131]),
        ...            np.array([64.51952734, 274.93442161, 1463.6313334])]
        >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
        ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
        >>> measurements = { 'RightElbowWidth': 74.0, 'LeftElbowWidth': 74.0,
        ...                'RightWristWidth': 55.0, 'LeftWristWidth': 55.0}
        >>> parameters = [RSHO, LSHO, RELB, LELB, RWRA , RWRB, LWRA, LWRB, thorax, shoulderJC, wand, measurements]
        >>> [np.around(arr,8) for arr in CGM.elbow_axis_calc(*parameters)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ 633.66707588,  304.95542115, 1256.07799541],
            [-129.16966701,  316.86794653, 1258.06440971]]), 
        array([[[ 633.81070139,  303.96579005, 1256.07658507],
                [ 634.35247992,  305.05386589, 1256.79947301],
                [ 632.95321804,  304.8508319 , 1256.77043175]],
            [[-129.32406616,  315.88151182, 1258.00866516],
                [-128.45131692,  316.79460332, 1257.37260488],
                [-128.4913352 ,  316.72108835, 1258.78433931]]]), 
        array([[ 793.32814303,  451.29134788, 1084.4325513 ],
            [-272.45939135,  485.80149026, 1091.36664789]])]
        """
        R_elbowwidth = measurements['RightElbowWidth']
        L_elbowwidth = measurements['LeftElbowWidth']
        R_elbowwidth = R_elbowwidth * -1
        L_elbowwidth = L_elbowwidth 
        mm = 7.0
        R_delta =( (R_elbowwidth/2.0)-mm )  
        L_delta =( (L_elbowwidth/2.0)+mm )  
        

        RWRI = [(RWRA[0]+RWRB[0])/2.0,(RWRA[1]+RWRB[1])/2.0,(RWRA[2]+RWRB[2])/2.0]
        LWRI = [(LWRA[0]+LWRB[0])/2.0,(LWRA[1]+LWRB[1])/2.0,(LWRA[2]+LWRB[2])/2.0]
        
        # make humerus axis
        tho_y_axis = np.subtract(thorax[0][1],thorax[1])
        
        R_sho_mod = [(RSHO[0]-R_delta*tho_y_axis[0]-RELB[0]),
                    (RSHO[1]-R_delta*tho_y_axis[1]-RELB[1]),
                    (RSHO[2]-R_delta*tho_y_axis[2]-RELB[2])]
        L_sho_mod = [(LSHO[0]+L_delta*tho_y_axis[0]-LELB[0]),
                    (LSHO[1]+L_delta*tho_y_axis[1]-LELB[1]),
                    (LSHO[2]+L_delta*tho_y_axis[2]-LELB[2])]
        
        # right axis
        z_axis = R_sho_mod
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
            # this is reference axis
        x_axis = np.subtract(RWRI,RELB)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        y_axis = CGM.cross(z_axis,x_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        R_axis = [x_axis,y_axis,z_axis]
        
        # left axis
        z_axis = np.subtract(L_sho_mod,LELB)
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
            # this is reference axis
        x_axis = L_sho_mod
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        y_axis = CGM.cross(z_axis,x_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        L_axis = [x_axis,y_axis,z_axis]
        
        RSJC = shoulderJC[0]
        LSJC = shoulderJC[1]
        
        # make the construction vector for finding Elbow joint center
        R_con_1 = np.subtract(RSJC,RELB)
        R_con_1_div = CGM.norm2d(R_con_1)
        R_con_1 = [R_con_1[0]/R_con_1_div,R_con_1[1]/R_con_1_div,R_con_1[2]/R_con_1_div]
        
        R_con_2 = np.subtract(RWRI,RELB)
        R_con_2_div = CGM.norm2d(R_con_2)
        R_con_2 = [R_con_2[0]/R_con_2_div,R_con_2[1]/R_con_2_div,R_con_2[2]/R_con_2_div]
        
        R_cons_vec = CGM.cross(R_con_1,R_con_2)
        R_cons_vec_div = CGM.norm2d(R_cons_vec)
        R_cons_vec = [R_cons_vec[0]/R_cons_vec_div,R_cons_vec[1]/R_cons_vec_div,R_cons_vec[2]/R_cons_vec_div]
        
        R_cons_vec = [R_cons_vec[0]*500+RELB[0],R_cons_vec[1]*500+RELB[1],R_cons_vec[2]*500+RELB[2]]
        
        L_con_1 = np.subtract(LSJC,LELB)
        L_con_1_div = CGM.norm2d(L_con_1)
        L_con_1 = [L_con_1[0]/L_con_1_div,L_con_1[1]/L_con_1_div,L_con_1[2]/L_con_1_div]
        
        L_con_2 = np.subtract(LWRI,LELB)
        L_con_2_div = CGM.norm2d(L_con_2)
        L_con_2 = [L_con_2[0]/L_con_2_div,L_con_2[1]/L_con_2_div,L_con_2[2]/L_con_2_div]

        L_cons_vec = CGM.cross(L_con_1,L_con_2)
        L_cons_vec_div = CGM.norm2d(L_cons_vec)

        L_cons_vec = [L_cons_vec[0]/L_cons_vec_div,L_cons_vec[1]/L_cons_vec_div,L_cons_vec[2]/L_cons_vec_div]

        L_cons_vec = [L_cons_vec[0]*500+LELB[0],L_cons_vec[1]*500+LELB[1],L_cons_vec[2]*500+LELB[2]]

        REJC = CGM.find_joint_c(R_cons_vec,RSJC,RELB,R_delta)
        LEJC = CGM.find_joint_c(L_cons_vec,LSJC,LELB,L_delta)

        
        # this is radius axis for humerus
        
            # right
        x_axis = np.subtract(RWRA,RWRB)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        z_axis = np.subtract(REJC,RWRI)
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
        y_axis = CGM.cross(z_axis,x_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        R_radius = [x_axis,y_axis,z_axis]

            # left
        x_axis = np.subtract(LWRA,LWRB)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        z_axis = np.subtract(LEJC,LWRI)
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
        y_axis = CGM.cross(z_axis,x_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        L_radius = [x_axis,y_axis,z_axis]
        
        # calculate wrist joint center for humerus
        R_wristThickness = measurements['RightWristWidth']
        L_wristThickness = measurements['LeftWristWidth']
        R_wristThickness = (R_wristThickness / 2.0 + mm )
        L_wristThickness = (L_wristThickness / 2.0 + mm )

        RWJC = [RWRI[0]+R_wristThickness*R_radius[1][0],RWRI[1]+R_wristThickness*R_radius[1][1],RWRI[2]+R_wristThickness*R_radius[1][2]]
        LWJC = [LWRI[0]-L_wristThickness*L_radius[1][0],LWRI[1]-L_wristThickness*L_radius[1][1],LWRI[2]-L_wristThickness*L_radius[1][2]]

        # recombine the humerus axis 

            #right
        
        z_axis = np.subtract(RSJC,REJC)
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
        x_axis = np.subtract(RWJC,REJC)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        y_axis = CGM.cross(x_axis,z_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        # attach each calulcated elbow axis to elbow joint center.
        x_axis = [x_axis[0]+REJC[0],x_axis[1]+REJC[1],x_axis[2]+REJC[2]]
        y_axis = [y_axis[0]+REJC[0],y_axis[1]+REJC[1],y_axis[2]+REJC[2]]
        z_axis = [z_axis[0]+REJC[0],z_axis[1]+REJC[1],z_axis[2]+REJC[2]]
        
        R_axis = [x_axis,y_axis,z_axis]
        
            # left
        
        z_axis = np.subtract(LSJC,LEJC)
        z_axis_div = CGM.norm2d(z_axis)
        z_axis = [z_axis[0]/z_axis_div,z_axis[1]/z_axis_div,z_axis[2]/z_axis_div]
        
        x_axis = np.subtract(LWJC,LEJC)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        y_axis = CGM.cross(x_axis,z_axis)
        y_axis_div = CGM.norm2d(y_axis)
        y_axis = [y_axis[0]/y_axis_div,y_axis[1]/y_axis_div,y_axis[2]/y_axis_div]
        
        x_axis = CGM.cross(y_axis,z_axis)
        x_axis_div = CGM.norm2d(x_axis)
        x_axis = [x_axis[0]/x_axis_div,x_axis[1]/x_axis_div,x_axis[2]/x_axis_div]
        
        # attach each calulcated elbow axis to elbow joint center.
        x_axis = [x_axis[0]+LEJC[0],x_axis[1]+LEJC[1],x_axis[2]+LEJC[2]]
        y_axis = [y_axis[0]+LEJC[0],y_axis[1]+LEJC[1],y_axis[2]+LEJC[2]]
        z_axis = [z_axis[0]+LEJC[0],z_axis[1]+LEJC[1],z_axis[2]+LEJC[2]]
        
        L_axis = [x_axis,y_axis,z_axis]
        
        axis = [R_axis,L_axis]
        
        origin = [REJC,LEJC]
        wrist_O = [RWJC,LWJC]
        
        return [origin,axis,wrist_O]

    @staticmethod
    def wrist_axis_calc(RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB, shoulderJC, elbowJC, wand):
        """The Wrist Axis Calculation function

        Calculate each wrist joint axis and returns it.

        Markers used: RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB

        Parameters
        ----------
        RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB : array
            A 1x3 array of each respective marker containing the XYZ positions.
        shoulderJC : array
            A 2x3 array containing the x,y,z position of the right and left shoulder joint center.
        elbowJC : array
            An array containing position of the left and right elbow joint centers. 
        wand : array
            A 2x3 array containing the x,y,z position of the right and left wand marker.

        Returns
        --------
        origin, axis : array 
            Returns the Shoulder joint center and axis in three array
                return = [[R_wrist_JC_x, R_wrist_JC_y, R_wrist_JC_z],
                            [L_wrist_JC_x,L_wrist_JC_y,L_wrist_JC_z],
                            [[[R_wrist x axis, x,y,z position],
                            [R_wrist y axis, x,y,z position],
                            [R_wrist z axis, x,y,z position]],
                            [[L_wrist x axis, x,y,z position],
                            [L_wrist y axis, x,y,z position],
                            [L_wrist z axis, x,y,z position]]]]
            
        Examples
        --------
        >>> import numpy as np 
        >>> from .pycgm import CGM
        >>> RSHO = np.array([428.88496562, 270.552948, 1500.73010254])
        >>> LSHO = np.array([68.24668121, 269.01049805, 1510.1072998])
        >>> RELB = np.array([658.90338135, 326.07580566, 1285.28515625])
        >>> LELB = np.array([-156.32162476, 335.2593313, 1287.39916992])
        >>> RWRA = np.array([776.51898193,495.68103027, 1108.38464355])
        >>> RWRB = np.array([830.9072876, 436.75341797, 1119.11901855])
        >>> LWRA = np.array([-249.28146362, 525.32977295, 1117.09057617])
        >>> LWRB = np.array([-311.77532959, 477.22512817, 1125.1619873])
        >>> wand = [[255.92550222678443, 364.32269504976051, 1460.6297868417887],
        ...        [256.42380097331767, 364.27770361353487, 1460.6165849382387]]
        >>> shoulderJC = [np.array([429.66951995, 275.06718615, 1453.953978131]),
        ...               np.array([64.51952734, 274.93442161, 1463.6313334])]
        >>> elbowJC = [[np.array([633.66707587, 304.95542115, 1256.07799541]),
        ...           np.array([-129.1695218, 316.8671644, 1258.06440717])],
        ...           [[[633.81070138699954, 303.96579004975194, 1256.07658506845],
        ...           [634.35247991784638, 305.05386589332528, 1256.7994730142241],
        ...           [632.95321803901493, 304.85083190737765, 1256.7704317504911]],
        ...           [[-129.32391792749493, 315.88072913249465, 1258.0086629318362],
        ...           [-128.45117135279025, 316.79382333592832, 1257.37260287807],
        ...           [-128.49119037560905, 316.7203088419364, 1258.783373067024]]],
        ...           [[793.32814303250677, 451.29134788252043, 1084.4325513020426],
        ...           [-272.4594189740742, 485.80152210947699, 1091.3666238350822]]]
        >>> parameters = [RSHO, LSHO, RELB, LELB, RWRA, RWRB, LWRA, LWRB, shoulderJC, elbowJC, wand]
        >>> [np.around(arr,8) for arr in CGM.wrist_axis_calc(*parameters)] #doctest: +NORMALIZE_WHITESPACE
        [array([[ 793.32814303,  451.29134788, 1084.4325513 ],
                [-272.45941897,  485.80152211, 1091.36662384]]), 
        array([[[ 793.77133728,  450.44879187, 1084.12648231],
                [ 794.01354708,  451.38979263, 1085.1540289 ],
                [ 792.75038863,  450.76181223, 1085.05367274]],
                [[-272.92507295,  485.01202419, 1090.9667996 ],
                [-271.74106833,  485.72818103, 1090.67481935],
                [-271.94256432,  485.19216661, 1091.96791174]]])]
        """
        # Bring Elbow joint center, axes and Wrist Joint Center for calculating Radius Axes

        REJC = elbowJC[0][0]
        LEJC = elbowJC[0][1]

        R_elbow_axis = elbowJC[1][0]
        L_elbow_axis = elbowJC[1][1]

        R_elbow_flex = [R_elbow_axis[1][0]-REJC[0],R_elbow_axis[1][1]-REJC[1],R_elbow_axis[1][2]-REJC[2]]
        L_elbow_flex = [L_elbow_axis[1][0]-LEJC[0],L_elbow_axis[1][1]-LEJC[1],L_elbow_axis[1][2]-LEJC[2]]

        RWJC = elbowJC[2][0]
        LWJC = elbowJC[2][1]

        # this is the axis of radius

            # right
        y_axis = R_elbow_flex
        y_axis = y_axis/ CGM.norm3d(y_axis)

        z_axis = np.subtract(REJC,RWJC)
        z_axis = z_axis/ CGM.norm3d(z_axis)

        x_axis = CGM.cross(y_axis,z_axis)
        x_axis = x_axis/ CGM.norm3d(x_axis)

        z_axis = CGM.cross(x_axis,y_axis)
        z_axis = z_axis/ CGM.norm3d(z_axis)

        # Attach all the axes to wrist joint center.
        x_axis = [x_axis[0]+RWJC[0],x_axis[1]+RWJC[1],x_axis[2]+RWJC[2]]
        y_axis = [y_axis[0]+RWJC[0],y_axis[1]+RWJC[1],y_axis[2]+RWJC[2]]
        z_axis = [z_axis[0]+RWJC[0],z_axis[1]+RWJC[1],z_axis[2]+RWJC[2]]

        R_axis = [x_axis,y_axis,z_axis]

            # left

        y_axis = L_elbow_flex
        y_axis = y_axis/ CGM.norm3d(y_axis)

        z_axis = np.subtract(LEJC,LWJC)
        z_axis = z_axis/ CGM.norm3d(z_axis)

        x_axis = CGM.cross(y_axis,z_axis)
        x_axis = x_axis/ CGM.norm3d(x_axis)

        z_axis = CGM.cross(x_axis,y_axis)
        z_axis = z_axis/ CGM.norm3d(z_axis)

        # Attach all the axes to wrist joint center.
        x_axis = [x_axis[0]+LWJC[0],x_axis[1]+LWJC[1],x_axis[2]+LWJC[2]]
        y_axis = [y_axis[0]+LWJC[0],y_axis[1]+LWJC[1],y_axis[2]+LWJC[2]]
        z_axis = [z_axis[0]+LWJC[0],z_axis[1]+LWJC[1],z_axis[2]+LWJC[2]]

        L_axis = [x_axis,y_axis,z_axis]

        origin = [RWJC,LWJC]

        axis = [R_axis,L_axis]

        return [origin,axis]

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
