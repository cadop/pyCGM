#!/usr/bin/python
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from pycgm.pycgm import CGM

    sample_dir = "SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
    subject1.write_results(sample_dir + "Sample_Refactor_Results.csv")

    print("pelvis jc")
    print(subject1.pelvis_joint_centers)
    print("pelvis axes")
    print(subject1.pelvis_axes)