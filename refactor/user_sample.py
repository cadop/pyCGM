#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import timeit

if __name__ == "__main__":
    from refactor.pycgm import CGM

    sample_dir = "../SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements)
    subject1.run()
    setup = '''from refactor.pycgm import CGM
sample_dir = "../SampleData/ROM/"
static_trial = sample_dir + "Sample_Static.c3d"
dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
measurements = sample_dir + "Sample_SM.vsk"
subject1 = CGM(static_trial, dynamic_trial, measurements)'''
    code = '''subject1.run()'''
    print(timeit.timeit(code, setup, number=1), "sec")
    # print(subject1.axis_results[0])

    # measurements = {'MeanLegLength': 940.0, 'R_AsisToTrocanterMeasure': 72.512,
    #                 'L_AsisToTrocanterMeasure': 72.512, 'InterAsisDistance': 215.908996582031}
    # pelvis_axis = np.array([[251.60830688, 391.74131775, 1032.89349365],
    #                         [251.74063624, 392.72694721, 1032.78850073],
    #                         [250.61711554, 391.87232862, 1032.8741063],
    #                         [251.60295336, 391.84795134, 1033.88777762]])
    # print(CGM.hip_axis_calc(pelvis_axis, measurements))
