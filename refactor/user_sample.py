#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os

if __name__ == "__main__":
    from refactor.pycgm import CGM

    sample_dir = "SampleData/59993_Frame/"
    static_trial = sample_dir + "59993_Frame_Static.c3d"
    dynamic_trial = sample_dir + "59993_Frame_Dynamic.c3d"
    measurements = sample_dir + "59993_Frame_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements)
    print(subject1.marker_data[0][0])
    print(subject1.measurements['Bodymass'])

    measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
                    'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
    pelvis_axis = np.array([[251.60830688, 391.74131775, 1032.89349365],
                            [251.74063624, 392.72694721, 1032.78850073],
                            [250.61711554, 391.87232862, 1032.8741063],
                            [251.60295336, 391.84795134, 1033.88777762]])
    print(CGM.hip_axis_calc(pelvis_axis, measurements))
