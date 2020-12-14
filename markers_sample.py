#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycgm.pycgm import CGM
from pycgm.customizations import CustomCGM3
import pycgm

if __name__ == "__main__":
    static_trial, dynamic_trial, measurements = pycgm.get_rom_data()
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
    print("head jc standard")
    print(subject1.head_joint_centers[0])

    static_trial_renamed = static_trial[:-4] + "_Renamed.csv"
    dynamic_trial_renamed = dynamic_trial[:-4] + "_Renamed.c3d"
    # Static and Dynamic trial contain the four xxHD markers renamed to xxHead format
    subject2 = CGM(static_trial_renamed, dynamic_trial_renamed, measurements, start=0, end=5)
    subject2.remap('RFHD', 'RFHead')  # Fix one marker with remap(), or multiple with bulk_remap()
    subject2.bulk_remap({'LFHD': 'LFHead', 'RBHD': 'RBHead', 'LBHD': 'LBHead'})
    subject2.run()
    print("head jc with renamed head markers")
    print(subject2.head_joint_centers[0])

    dynamic_trial_extra = dynamic_trial[:-4] + "_Extra.c3d"
    static_trial_extra = static_trial[:-4] + "_Extra.c3d"
    # Dynamic trial now contains fake markers RWRI and LWRI which replace their respective calculations
    subject3 = CustomCGM3(static_trial_extra, dynamic_trial_extra, measurements, start=0, end=5)
    subject3.bulk_remap({"RWRI": "RWRI", "LWRI": "LWRI"})  # Must define new markers
    subject3.run()
    print("wrist jc standard")
    print(subject1.wrist_joint_centers[0])
    print("wrist jc extra markers")
    print(subject3.wrist_joint_centers[0])
