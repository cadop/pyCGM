# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:16:44 2015

@author: cadop
"""
import pyCGM

#Each rank should run this function and return the array of data
# motiondata should be passed individually with scatter
# static and vsk should use bcast
def calcFramesMPI(motiondata,vsk):
    angles=[]
    
    for frame in motiondata:
        angle = pyCGM.JointAngleCalc(frame,vsk)
        angles.append(angle)
    #should be send to master rank
    return angles
 ###############################################################################
