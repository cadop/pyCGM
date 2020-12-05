#!/usr/bin/python
# -*- coding: utf-8 -*-

import timeit

if __name__ == "__main__":
    from refactor.pycgm import CGM

    sample_dir = "../SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements)
    subject1.run()
    subject1.write_results(sample_dir + "Sample_Refactor_Results.csv")
    print(subject1.pelvis_axes[0])
    print(subject1.elbow_angles[0])

# timeit sample
#     setup = '''from refactor.pycgm import CGM
# sample_dir = "../SampleData/ROM/"
# static_trial = sample_dir + "Sample_Static.c3d"
# dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
# measurements = sample_dir + "Sample_SM.vsk"
# subject1 = CGM(static_trial, dynamic_trial, measurements)'''
#     code = '''subject1.run()'''
#     print(timeit.timeit(code, setup, number=10), "sec")
