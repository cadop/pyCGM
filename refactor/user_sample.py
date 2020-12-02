if __name__ == "__main__":

    from refactor.pycgm import CGM
    dir = "../SampleData/59993_Frame/"
    static_trial = dir + "59993_Frame_Static.c3d"
    dynamic_trial = dir + "59993_Frame_Dynamic.c3d"
    measurements = dir + "59993_Frame_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements)
    print(subject1.marker_data[0][0])
