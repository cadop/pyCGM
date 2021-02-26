"""
This file is used to calculate frames for gap filling.
"""

# flake8: noqa # Ignore this file for now

def calcFramesClusters(data,vsk):
    """ Calculates frames using cluster information for gap filling.

    Parameters
    ----------
    data : list
        First element is a list of marker lables.
        Second element is a list of data.
    vsk : list
        First element is a list of marker lables.
        Second element is a list of data.

    Returns
    -------
    angles, joints : tuple(list, list)
        Returns the joint angles for the right and left pelvis, hip, knee, and ankle
    """
    angles=[]
    joints = [] #temp solution
    if type(data[0])!=type({}):
        data=createMotionDataDict(data[0],data[1])
    if type(vsk)!=type({}):
        vsk=createVskDataDict(vsk[0],vsk[1])

    from .clusterCalc import targetName, getMarkerLocation, targetDict, groupInClustDict,getStaticTransform
    missingMarkerName = targetName()
    targets = targetDict()
    clusters = groupInClustDict()
    
    import numpy as np
    for i in range(len(data)):
        frame = data[i]
        # markers = [ frame[x] for x in missingMarkerName ]            
        
        #I thought there are two cases, but now can only think of one: 
        # the marker is missing in the dynamic trial and should be calculated with the offset
        # or ???
        #Since it is missing in the dynamic trial, there is no key for it,
        # usually when loading if the data is missing, the key exists but data is gone
        # so we enter a nan key, and then in the loop it will calculate the transform
        
        #list_check = [name for name in missingMarkerName if name in frame ]
        #assign the missing marker names to only the ones that are in the trial
        #missingMarkerName = list_check
        
        #get list of markers that are not in the dynamic
        removedMarkers = [name for name in missingMarkerName if name not in frame ]

        for item in removedMarkers:
            frame[item] = (float('nan'),float('nan'),float('nan'))
        
        # for each marker that is missing, find the nearest previous frame that the  
        # cluster and marker exists and calculate the transform
        markers = {k:frame[k] for k in missingMarkerName}
        if np.any(np.isnan(list(markers.values()))):
            for key in markers:
                if np.isnan(markers[key][0]) == False: continue
                
                cluster_ver = targets[key] #this is the ['Trunk_A_C7'] name
                
                #this target marker is missing, move back in time to find its position
                # if there is no position available (maybe this is first frame)
                # , just use the stored one from static trial
                # if this is one of the removed markers, no point in searching for it
                
                #we can either calculate the missing marker based on the last real frame
                # or we can use the calculated position in the previous frame to 
                #  estimate this one (although it is slower)
                
                last_time = None
                if key not in removedMarkers:
                    j = i
                    while j >=0:
                        if np.isnan(data[j][key][0]):
                            j -= 1
                            continue
                        #
                        # check if the markers needed for the cluster are also available
                        clustname = clusters[cluster_ver[0]]
                        cm1 = (np.isnan(data[j][clustname[0]][0]))
                        cm2 = (np.isnan(data[j][clustname[1]][0]))
                        cm3 = (np.isnan(data[j][clustname[2]][0]))
                        if cm1 or cm2 or cm3:
                            j -= 1
                            continue
                        #if everything is there, we can use this frame to calculate transform
                        last_time = j
                        break
                
                #print('found the last time the target marker',key,' was visible',last_time)
                
                
                clust_bool = True
                for clust in cluster_ver: #incase there are multiple options
                    #Pm is marker that is missing, C is cluster
                    Cname = clusters[clust]
                    
                    
                    #if last_time is None, it means we couldn't find a recent useable frame,
                    # so we will use the static trial
                    if last_time == None:
                        Pm = vsk['clusterLJMU'][clust]
                    #otherwise we can calculate Pm from the data of that previous frame
                    else:
                        tmp_frame = data[last_time]
                        p = tmp_frame[key]
                        C = [tmp_frame[Cname[0]],tmp_frame[Cname[1]],tmp_frame[Cname[2]]]
                        Pm = getStaticTransform(p,C) #p is target, C is cluster
                        
                    
                    C = [frame[Cname[0]],frame[Cname[1]],frame[Cname[2]]]
                    #if we are missing any of the cluser markers 
                    if np.any(np.isnan(C)): 
                        clust_bool = False
                        continue 
                if clust_bool == False: continue
                
                est_pos = getMarkerLocation(Pm,C) 
                #set the marker in this frame to the estimated position
                frame[key] = est_pos

        angle,jcs = JointAngleCalc(frame,vsk)
        angles.append(angle)
        joints.append(jcs)
    return angles,joints
