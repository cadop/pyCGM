#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM
from pycgm.customizations import CustomCGM1, CustomCGM2
import pycgm

if __name__ == "__main__":
    static_trial, dynamic_trial, measurements = pycgm.get_rom_data()
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
    # subject1.write_results(sample_dir + "Sample_Refactor_Results.csv")
    print("head jc standard")
    print(subject1.head_joint_centers[:, 0])
    # print(subject1.head_flexions)
    # 
    # subject2 = CustomCGM1(static_trial, dynamic_trial, measurements, start=0, end=5)
    # subject2.run()
    # print("pelvis jc custom")
    # print(subject2.pelvis_joint_centers)
    # 
    # subject3 = CustomCGM2(static_trial, dynamic_trial, measurements, start=0, end=5)
    # subject3.run()
    # print("hip axis standard")
    # print(subject1.hip_axes[0])
    # 
    # print("hip axis custom")
    # print(subject3.hip_axes[0])

    static_trial_renamed = static_trial[:-4] + "_Renamed.csv"
    dynamic_trial_renamed = dynamic_trial[:-4] + "_Renamed.c3d"
    # Static and Dynamic trial contain the four xxHD markers renamed to xxHead format
    subject4 = CGM(static_trial_renamed, dynamic_trial_renamed, measurements, start=0, end=5)
    subject4.remap('RFHD', 'RFHead')
    subject4.remap('LFHD', 'LFHead')
    subject4.remap('RBHD', 'RBHead')
    subject4.remap('LBHD', 'LBHead')
    subject4.run()
    print("head jc with renamed head markers")
    print(subject4.head_joint_centers[:, 0])
