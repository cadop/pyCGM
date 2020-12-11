"""
An example of the most basic use case of pycgm to perform joint angle calculations.

    >>> from pycgm.pycgm import CGM

    After importing, locate the input files (Static trial, Dynamic trial and Subject Measurements).

    >>> sample_dir = "SampleData/ROM/"
    >>> static_trial = sample_dir + "Sample_Static.c3d"
    >>> dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    >>> measurements = sample_dir + "Sample_SM.vsk"

    Create the `CGM` object by passing in the input file paths.

    >>> subject1 = CGM(static_trial, dynamic_trial, measurements)

    Explicitly tell the object when to perform the calculations with `run()`.

    >>> subject1.run()

    # TODO: Show accessing output

"""

if __name__ == "__main__":
    from pycgm.pycgm import CGM

    sample_dir = "SampleData/ROM/"
    static_trial = sample_dir + "Sample_Static.c3d"
    dynamic_trial = sample_dir + "Sample_Dynamic.c3d"
    measurements = sample_dir + "Sample_SM.vsk"
    subject1 = CGM(static_trial, dynamic_trial, measurements, start=0, end=5)
    subject1.run()
