def parameters():
    return {
        "calc_axis_pelvis" :[
            'RASI', 'LASI',
            'RPSI', 'LPSI',
            'SACR'
        ],

        "calc_joint_center_hip" :[
            'Pelvis', 'MeanLegLength',
            'R_AsisToTrocanterMeasure',
            'L_AsisToTrocanterMeasure',
            'InterAsisDistance'
        ],

        "calc_axis_hip" :[
            'RHipJC',
            'LHipJC',
            'Pelvis'
        ],

        "calc_axis_knee" :[
            'RTHI', 'LTHI',
            'RKNE', 'LKNE',
            'RHipJC', 'LHipJC',
            'RightKneeWidth',
            'LeftKneeWidth'
        ],

        "calc_axis_ankle" : [
            'RTIB', 'LTIB',
            'RANK', 'LANK',
            'RKnee',  'LKnee',
            'RightAnkleWidth',
            'LeftAnkleWidth',
            'RightTibialTorsion',
            'LeftTibialTorsion'
        ],

        "calc_axis_foot" : [
            'RTOE', 'LTOE',
            'RAnkle', 'LAnkle',
            'RightStaticRotOff',
            'LeftStaticRotOff',
            'RightStaticPlantFlex',
            'LeftStaticPlantFlex',
        ],

        "calc_axis_head" : [
            'LFHD',
            'RFHD',
            'LBHD',
            'RBHD',
            'HeadOffset'
        ],

        "calc_axis_thorax" : [
                'CLAV', 'C7',
                'STRN', 'T10'
        ],

        "calc_marker_wand" : [
                'RSHO', 'LSHO',
                'Thorax'

        ],

        "calc_joint_center_shoulder" : [
                'RSHO', 'LSHO',
                'Thorax', 'RWand', 'LWand',
                'RightShoulderOffset',
                'LeftShoulderOffset'
        ],

        "calc_axis_shoulder" : [
                'Thorax',
                'RClavJC',
                'LClavJC',
                'RWand',
                'LWand'
        ],

        "calc_axis_elbow" : [
                'RELB', 'LELB',
                'RWRA', 'RWRB',
                'LWRA', 'LWRB',
                'RClav',  'LClav',
                'RightElbowWidth',
                'LeftElbowWidth',
                'RightWristWidth',
                'LeftWristWidth',
                7.0  # marker mm
        ],

        "calc_axis_wrist" : [
                'RHum',     'LHum',
                'RWristJC', 'LWristJC'
        ],

        "calc_axis_hand" : [
                'RWRA',   'RWRB',
                'LWRA',   'LWRB',
                'RFIN',   'LFIN',
                'RWristJC', 'LWristJC',
                'RightHandThickness',
                'LeftHandThickness'
        ],

        "calc_angle_pelvis" : [
            'GCS',
            'Pelvis'
        ],

        "calc_angle_hip" : [
            'Hip', 'RKnee',
            'Hip', 'LKnee'
        ],

        "calc_angle_knee" : [
            'RKnee', 'RAnkle',
            'LKnee', 'LAnkle'
        ],

        "calc_angle_ankle" : [
            'RAnkle', 'RFoot',
            'LAnkle', 'LFoot'
        ],

        "calc_angle_foot" : [
            'GCS', 'RFoot',
            'GCS', 'LFoot'
        ],

        "calc_angle_head" : [
            'GCS',
            'Head'
        ],

        "calc_angle_thorax" : [
            'GCS',
            'Thorax'
        ],

        "calc_angle_neck" : [
            'Head',
            'Thorax'
        ],

        "calc_angle_spine" : [
            'Pelvis',
            'Thorax'
        ],

        "calc_angle_shoulder" : [
            'Thorax',
            'RHum', 'LHum'
        ],

    "calc_angle_elbow" : [
            'RHum', 'RRad',
            'LHum', 'LRad'
        ],

    "calc_angle_wrist" : [
            'RRad', 'RHand',
            'LRad', 'LHand'
        ]
    }


def returns():
    return { 'calc_axis_pelvis': ['Pelvis'],
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
             'calc_axis_hand': ['RHand', 'LHand'],
             'calc_angle_pelvis': ['Pelvis'],
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
             'calc_angle_wrist': ['RWrist', 'LWrist']
        }

