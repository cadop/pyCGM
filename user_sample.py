#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM
from pycgm.customizations import CustomCGM1, CustomCGM2

if __name__ == "__main__":
    sample_dir = "SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
    # subject1.write_results(sample_dir + "Sample_Refactor_Results.csv")
    print("pelvis jc standard")
    print(subject1.pelvis_joint_centers)

    subject2 = CustomCGM1(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject2.run()
    print("pelvis jc custom")
    print(subject2.pelvis_joint_centers)

    subject3 = CustomCGM2(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject3.run()
    print("hip axis standard")
    print(subject1.hip_axes[0])

    print("hip axis custom")
    print(subject3.hip_axes[0])
