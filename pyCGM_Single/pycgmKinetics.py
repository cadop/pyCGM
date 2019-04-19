#pyCGM

# This module was contributed by Neil M. Thomas
# the CoM calculation is not an exact clone of PiG, 
# but the differences are not clinically significant. 
# We will add updates accordingly.
#

from __future__ import division
import os
import numpy as np
import sys

if sys.version_info[0]==2:
    pyver = 2
else:
    pyver = 3
    
#helper functions useful for dealing with frames of data, i.e. 1d arrays of (x,y,z)
#coordinate. Also in Utilities but need to clean everything up somewhat!         
def f(p, x):
    return (p[0] * x) + p[1]

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z
  
def length(v):
    x,y,z = v
    return np.sqrt(x*x + y*y + z*z)
  
def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)
  
def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)
  
def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

def pnt2line(pnt, start, end):
    lineVec = vector(start, end)

    pntVec = vector(start, pnt)
    
    lineLength = length(lineVec)
    lineUnitVec = unit(lineVec)
    pntVecScaled = scale(pntVec, 1.0/lineLength)
    
    t = dot(lineUnitVec, pntVecScaled)
        
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
        
    nearest = scale(lineVec, t)
    dist = distance(nearest, pntVec)
    
    nearest = add(nearest, start)
    
    return dist, nearest, pnt 
 
 
#def norm3d(v): 
#    try:
#        return np.asarray(sqrt((v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))
#    except:
#        return np.nan


def findL5_Pelvis(frame):
    #The L5 position is estimated as (LHJC + RHJC)/2 + 
    #(0.0, 0.0, 0.828) * Length(LHJC - RHJC), where the value 0.828 
    #is a ratio of the distance from the hip joint centre level to the 
    #top of the lumbar 5: this is calculated as in teh vertical (z) axis
    LHJC = frame['LHip']
    RHJC = frame['RHip']
    midHip = (LHJC+RHJC)/2
    #zOffset = ([0.0,0.0,distance(RHJC, LHJC)*0.925])
    #L5 = midHip + zOffset
    
    offset = distance(RHJC,LHJC) * .925
    z_axis = frame['Pelvis_axis'][1][2] 
    norm_dir = np.array(unit(z_axis))
    L5 = midHip + offset * norm_dir

    return midHip, L5#midHip + ([0.0, 0.0, zOffset])   

def findL5_Thorax(frame):
    C7_ = frame['C7']
    x_axis,y_axis,z_axis = frame['Thorax_axis'][0] 
    norm_dir_y = np.array(unit(y_axis))
    if C7_[1] >= 0:
        C7 = C7_ + 7 * -norm_dir_y
    else:
        C7 = C7_ + 7 * norm_dir_y
        
    norm_dir = np.array(unit(z_axis))
    LHJC = frame['LHip']
    RHJC = frame['RHip']
    midHip = (LHJC+RHJC)/2
    offset = distance(RHJC,LHJC) * .925
    
    L5 = midHip + offset * norm_dir
    return L5
   
def getKinetics(data, Bodymass):
    '''
    Estimates whole body CoM in global coordinate system using PiG scaling 
    factors for determining individual segment CoM. 
    
    
    Parameters
    -----------
    data: list of dicts
        Joint centres in the global coordinate system. List indices correspond 
        to each frame of trial. Dict keys correspond to name of each joint centre,
        dict values are arrays ([],[],[]) of x,y,z coordinates for each joint 
        centre
    
    Bodymass: float
        Total bodymass (kg) of subject
    
    
    Notes
    -----
    The PiG scaling factors are taken from Dempster -- they are available at:
    http://www.c-motion.com/download/IORGaitFiles/pigmanualver1.pdf
    
    
    Returns
    -------
    CoM: 3D numpy array
        CoM trajectory in the global coordinate system 
    
        
    Todo 
    ----
    Tidy up and optimise code
    
    Joint moments etc. 
    
    Figure out weird offset 
    
    '''
    
    #get PiG scaling table
    #PiG_xls =  pd.read_excel(os.path.dirname(os.path.abspath(__file__)) +
    #                    '/segments.xls', skiprows = 0)
    
    segScale = {}
    with open(os.path.dirname(os.path.abspath(__file__)) +'/segments.csv','r') as f:
        header = False
        for line in f:
            if header == False:
                header = line.rstrip('\n').split(',')
                header = True
            else:
                row = line.rstrip('\n').split(',')
                segScale[row[0]] = {'com':float(row[1]),'mass':float(row[2]),'x':row[3],'y':row[4],'z':row[5]}
    
    #names of segments
    sides = ['L', 'R']
    segments = ['Foot','Tibia','Femur','Pelvis','Radius','Hand','Humerus','Head','Thorax']
    
    #empty array for CoM coords
    CoM_coords = np.empty([len(data), 3])

    #iterate through each frame of JC
    for ind, frame in enumerate(data): #enumeration used to populate CoM_coords
        
        #find distal and proximal joint centres
        segTemp = {}
        
        for s in sides:
            for seg in segments:
                if seg!='Pelvis' and seg!='Thorax' and seg!='Head':
                    segTemp[s+seg] = {}
                else:
                    segTemp[seg] = {}
                    
                #populate dict with appropriate joint centres
                if seg == 'Foot':
                    #segTemp[s+seg]['Prox'] = frame[s+'Ankle']
                    #segTemp[s+seg]['Dist'] = frame[s+'Foot']
                    segTemp[s+seg]['Prox'] = frame[s+'Foot'] #should be heel to toe?
                    segTemp[s+seg]['Dist'] = frame[s+'HEE']
                    
                if seg == 'Tibia':
                    segTemp[s+seg]['Prox'] = frame[s+'Knee']
                    segTemp[s+seg]['Dist'] = frame[s+'Ankle']
               
                if seg == 'Femur':
                    segTemp[s+seg]['Prox'] = frame[s+'Hip']
                    segTemp[s+seg]['Dist'] = frame[s+'Knee']
                
                if seg == 'Pelvis':
                    
                    midHip,L5 = findL5_Pelvis(frame) #see function above
                    segTemp[seg]['Prox'] = midHip 
                    segTemp[seg]['Dist'] = L5
                
                if seg == 'Thorax':
                    #The thorax length is taken as the distance between an 
                    #approximation to the C7 vertebra and the L5 vertebra in the 
                    #Thorax reference frame. C7 is estimated from the C7 marker, 
                    #and offset by half a marker diameter in the direction of 
                    #the X axis. L5 is estimated from the L5 provided from the 
                    #pelvis segment, but localised to the thorax.
                    
                    
                    L5 = findL5_Thorax(frame)
                    #_,L5 = findL5_Pelvis(frame)
                    C7 = frame['C7']
                    
                    #y_axis = frame['Thorax_axis'][0][0] 
                    #norm_dir_y = np.array(unit(y_axis))
                    #if C7_[1] >= 0:
                    #    C7 = C7_ + 100000 * norm_dir_y
                    #else:
                    #    C7 = C7_ + 100000 * norm_dir_y.flip()
                        
                    #C7 = C7_ + 100 * -norm_dir_y
                    
                    CLAV = frame['CLAV']
                    STRN = frame['STRN']
                    T10 = frame['T10']
                    
                    upper = np.array([(CLAV[0]+C7[0])/2.0,(CLAV[1]+C7[1])/2.0,(CLAV[2]+C7[2])/2.0])
                    lower = np.array([(STRN[0]+T10[0])/2.0,(STRN[1]+T10[1])/2.0,(STRN[2]+T10[2])/2.0])
                    
                    #Get the direction of the primary axis Z (facing down)
                    z_vec = upper - lower
                    z_dir = np.array(unit(z_vec))
                    newStart = upper + (z_dir * 300)
                    newEnd = lower - (z_dir * 300)
        
                    _,newL5,_ = pnt2line(L5, newStart, newEnd)
                    _,newC7,_ = pnt2line(C7, newStart, newEnd)
                    
                    segTemp[seg]['Prox'] = np.array(newC7)
                    segTemp[seg]['Dist'] = np.array(newL5)
                
                if seg == 'Humerus':
                    segTemp[s+seg]['Prox'] = frame[s+'Shoulder'] 
                    segTemp[s+seg]['Dist'] = frame[s+'Humerus'] 
                    
                if seg == 'Radius':
                    segTemp[s+seg]['Prox'] = frame[s+'Humerus'] 
                    segTemp[s+seg]['Dist'] = frame[s+'Radius'] 
                    
                if seg == 'Hand':
                    segTemp[s+seg]['Prox'] = frame[s+'Radius']  
                    segTemp[s+seg]['Dist'] = frame[s+'Hand'] 

                if seg == 'Head':
                    segTemp[seg]['Prox'] = frame['Back_Head']
                    segTemp[seg]['Dist'] = frame['Front_Head']
                    
                    
                #iterate through csv scaling values 
                for row in list(segScale.keys()):
                    #if row[0] == seg:
                    #scale = row[1] #row[1] contains segment names
                    # print(seg,row,segScale[row]['mass'])
                    scale = segScale[row]['com']
                    mass = segScale[row]['mass']
                    if seg == row:
                        #s = sides, which are added to limbs (not Pelvis etc.)
                        if seg!='Pelvis' and seg!='Thorax' and seg!='Head':
                            
                            prox = segTemp[s+seg]['Prox']
                            dist = segTemp[s+seg]['Dist']
                            
                            #segment length
                            length = prox - dist
                            
                            #segment CoM
                            CoM = dist + length * scale 
                            
                            #CoM = prox + length * scale
                            segTemp[s+seg]['CoM'] = CoM
                            
                            #segment mass (kg)
                            mass = Bodymass*mass #row[2] contains mass corrections
                            segTemp[s+seg]['Mass'] = mass
                            
                            #segment torque
                            torque = CoM * mass
                            segTemp[s+seg]['Torque'] = torque
                            
                            #vector
                            Vector = np.array(vector(([0,0,0]), CoM))
                            val = Vector*mass
                            segTemp[s+seg]['val'] = val
                            
                        
                        #this time no side allocation
                        else:
                            prox = segTemp[seg]['Prox']
                            dist = segTemp[seg]['Dist']
                            
                            #segment length
                            length = prox - dist
                            
                            #segment CoM
                            CoM = dist + length * scale
                            #CoM = prox + length * scale
                            
                            segTemp[seg]['CoM'] = CoM
                            
                            #segment mass (kg)
                            mass = Bodymass*mass #row[2] is mass correction factor
                            segTemp[seg]['Mass'] = mass
                            
                            #segment torque
                            torque = CoM * mass
                            segTemp[seg]['Torque'] = torque
                            
                            #vector
                            Vector = np.array(vector(([0,0,0]), CoM))
                            val = Vector*mass
                            segTemp[seg]['val'] = val
                                
                            
        keylabels  = ['LHand', 'RTibia', 'Head', 'LRadius', 'RFoot', 'RRadius', 'LFoot', 'RHumerus', 'LTibia', 'LHumerus', 'Pelvis', 'RHand', 'RFemur', 'Thorax', 'LFemur']
        # print(segTemp['RFoot'])
        
        # for key in keylabels:
            # print(key,segTemp[key]['val'])

        vals = []
        
        # for seg in list(segTemp.keys()):
            # vals.append(segTemp[seg]['val'])
        if pyver == 2:
            forIter = segTemp.iteritems()
        if pyver == 3:
            forIter = segTemp.items()
            
        for attr, value in forIter:
           vals.append(value['val'])
           #print(value['val'])
        
        CoM_coords[ind,:] = sum(vals) / Bodymass
        
        #add all torques and masses
        #torques = []
        #masses = []
        #for attr, value in segTemp.iteritems():
        #    torques.append(value['Torque'])
        #    masses.append(value['Mass'])

        #calculate whole body centre of mass coordinates and add to CoM_coords array
        #CoM_coords[ind,:] = sum(torques) / sum(masses) 
            
    return CoM_coords



            
            

 
             
    

  
    
