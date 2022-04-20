def axes():
    """
    map function names to the axes they return
    """
    return {'calc_axis_pelvis': ['Pelvis'],
            'calc_joint_center_hip': ['RHipJC', 'LHipJC'],
            'calc_axis_hip': ['Hip'],
            'calc_axis_knee': ['RKnee', 'LKnee'],
            'calc_axis_ankle': ['RAnkle', 'LAnkle'],
            'calc_axis_foot': ['RFoot', 'LFoot'],
            'calc_axis_head': ['Head'],
            'calc_axis_thorax': ['Thorax'],
            'calc_marker_wand': ['RWand', 'LWand'],
            'calc_joint_center_shoulder': ['RClavJC', 'LClavJC'],
            'calc_axis_shoulder': ['RClav', 'LClav'],
            'calc_axis_elbow': ['RHum', 'LHum', 'RWristJC', 'LWristJC'],
            'calc_axis_wrist': ['RRad', 'LRad'],
            'calc_axis_hand': ['RHand', 'LHand']}

def angles():
    """
    map function names to the angles they return
    """
    return {'calc_angle_pelvis': ['Pelvis'],
            'calc_angle_hip': ['RHip', 'LHip'],
            'calc_angle_knee': ['RKnee', 'LKnee'],
            'calc_angle_ankle': ['RAnkle', 'LAnkle'],
            'calc_angle_foot': ['RFoot', 'LFoot'],
            'calc_angle_head': ['Head'],
            'calc_angle_thorax': ['Thorax'],
            'calc_angle_neck': ['Neck'],
            'calc_angle_spine': ['Spine'],
            'calc_angle_shoulder': ['RShoulder', 'LShoulder'],
            'calc_angle_elbow': ['RElbow', 'LElbow'],
            'calc_angle_wrist': ['RWrist', 'LWrist']}
