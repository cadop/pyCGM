import pytest
import numpy as np
import os
import sys
import pyCGM_Single.pyCGM_Helpers as pyCGM_Helpers
import pyCGM_Single.pycgmStatic as pycgmStatic
import pyCGM_Single.pycgmIO as pycgmIO
import pyCGM_Single.pycgmCalc as pycgmCalc

class TestPycgmCalc:
    @classmethod
    def setup_class(cls):
        """
        Called once for all tests for pycgmCalc.
        Sets rounding_precision, and loads from SampleData/ROM/
        to be used for testing the calculation functions.
        """
        cls.rounding_precision = 8
        cwd = os.getcwd()
        if(cwd.split(os.sep)[-1]=="pyCGM_Single"):
            parent = os.path.dirname(cwd)
            os.chdir(parent)
        cls.cwd = os.getcwd()

        #Load data from SampleData/ROM/ for testing
        dynamic_trial,static_trial,vsk_file,_,_ = pyCGM_Helpers.getfilenames(x=2)
        cls.motion_data = pycgmIO.loadData(os.path.join(cls.cwd, dynamic_trial))
        cls.static_data = pycgmIO.loadData(os.path.join(cls.cwd, static_trial))
        cls.vsk_data = pycgmIO.loadVSK(os.path.join(cls.cwd, vsk_file), dict=False)
        cls.cal_SM = pycgmStatic.getStatic(cls.static_data,cls.vsk_data,flat_foot=False)

    @pytest.mark.parametrize("kargs, expected_len_result, expected_first_angle, expected_first_axis", [
        #All of the following tests return angles and axis
        #Calculate for only one frame
        ({'frame' : 25}, 1, 
        np.array([-0.28586517, -6.16733918,  7.43502326]),
        np.array([[ 252.09220886,  393.2449646 , 1032.81604004],
                  [ 252.22214051,  394.23081787, 1032.71015762],
                  [ 251.10069832,  393.37361742, 1032.79719092],
                  [ 252.08724847,  393.35239723, 1033.81024003]])),
        #Calculate for a range of frames
        ({'start': 2, 'end': 4}, 2, 
        np.array([-0.30389522, -6.12213066,  7.55373785]),
        np.array([[ 251.62969971,  391.89085388, 1032.89422607],
                  [ 251.76171462,  392.8765222 , 1032.78920216],
                  [ 250.63846601,  392.02156018, 1032.87494881],
                  [ 251.62442601,  391.99750201, 1033.88850891]]))
    ])
    def test_calcAngles_angles_and_axis(self, kargs, expected_len_result, expected_first_angle, expected_first_axis):
        """
        This function tests pycgmCalc.calcAngles(data, **kargs),
        where data is the motion capture data to calculate angles for,
        and **kargs contains many options for how to return the calculated
        data.

        The 'vsk' argument in **kargs is required and used in all tests.

        The previously loaded motion_data in setup_class() is used
        for testing.
        
        This function only tests the usage of key word arguments to 
        customize the layout format, not the accuracy of the results.

        We test the usage of calcAngles() to return both angles and 
        axis.
        """
        kargs['vsk'] = self.cal_SM
        angles, axis = pycgmCalc.calcAngles(self.motion_data, **kargs)
        np.testing.assert_equal(len(angles), expected_len_result)
        np.testing.assert_equal(len(axis), expected_len_result)
        np.testing.assert_almost_equal(angles[0][0], expected_first_angle, self.rounding_precision)
        np.testing.assert_almost_equal(axis[0][0], expected_first_axis, self.rounding_precision)
    
    @pytest.mark.parametrize("kargs, expected_len_result, expected_truncated_results", [
        #Return angles only
        ({'frame': 0, 'formatData': False, 'axis': False}, 57, 
         np.array([[-0.30849491],
                   [-6.12129279],
                   [ 7.5714311 ],
                   [ 2.91422293],
                   [-6.86706898]])),
        #Return axis only
        ({'frame': 0, 'formatData': False, 'angles': False}, 216, 
         np.array([[ 251.60830688],
                   [ 391.74131775],
                   [1032.89349365],
                   [ 251.74063624],
                   [ 392.72694721]]))
    ])
    def test_calcAngles_angles_or_axis(self, kargs, expected_len_result, expected_truncated_results):
        """
        Test returning angles only or axis only through the
        keyword arguments in calcAngles(). Test the results are 
        accurate by testing for the first 5 values in the returned
        arrays.
        """
        kargs['vsk'] = self.cal_SM
        result = pycgmCalc.calcAngles(self.motion_data, **kargs)
        np.testing.assert_equal(len(result), expected_len_result)
        np.testing.assert_almost_equal(result[0:5], expected_truncated_results)

    def test_calcAngles_joint_centers(self):
        """
        Test returning joint_centers through calcAngles().
        """
        _,joint_centers = pycgmCalc.calcAngles(self.motion_data, vsk=self.cal_SM, frame=0,\
                                               formatData=False, splitAnglesAxis=False, returnjoints=True)
        #Verify that several the expected joint_centers are returned.
        expected_Front_Head = np.array([ 255.19071198,  406.12081909, 1721.92053223])
        expected_LHip = np.array([182.57097863, 339.43231855, 935.52900126])
        expected_RHand = np.array([ 859.80614366,  517.28239823, 1051.97278944])
        expected_Thorax = np.array([256.149810236564, 364.3090603933987, 1459.6553639290375])
        expected_LKnee = np.array([143.55478579, 279.90370346, 524.78408753])
        expected_result = [expected_Front_Head, expected_LHip, expected_RHand, expected_Thorax, expected_LKnee]
        result = [
            joint_centers[0]['Front_Head'],
            joint_centers[0]['LHip'],
            joint_centers[0]['RHand'],
            joint_centers[0]['Thorax'],
            joint_centers[0]['LKnee']
        ]
        np.testing.assert_almost_equal(result, expected_result, self.rounding_precision)

    @pytest.mark.parametrize("kargs", [
        ({'start': 10, 'end':0}),
        ({'start': -1}),
        ({'frame': -1}),
        ({'frame': 4000}),
        ({'end': 4000}),
    ])
    def test_calcAngles_exceptions(self, kargs):
        """
        Test exceptions raised by pycgmCalc in the following cases:
        - 'start' value > 'end' value
        - negative 'start' value
        - negative 'frame' value
        - 'frame' value out of range
        - 'end' value out of range
        """
        kargs['vsk'] = self.cal_SM
        with pytest.raises(Exception):
            pycgmCalc.calcAngles(self.motion_data, **kargs)
    
    @pytest.mark.parametrize("start, end, expected_len_result, expected_first_angles, expected_LHumerus", [
        (0, 10, 10, 
        np.array([-0.30849491, -6.12129279,  7.5714311 ,  2.91422293, -6.86706898]), 
        np.array([-129.16952218,  316.8671644 , 1258.06440717])),
        (1, 2, 1,
        np.array([-0.30611243, -6.12165307,  7.5624917 ,  2.91095144, -6.86847416]),
        np.array([-129.1399663 ,  316.89254513, 1258.06483031])),
    ])
    def test_Calc_accuracy(self, start, end, expected_len_result, expected_first_angles, expected_LHumerus):
        """
        This function tests pycgmCalc.Calc(start, end, data, vsk), where
        start is the start frame index to begin the calculation, end is 
        the end frame index to end the calculation, data is the motion
        capture data to calculate angles for and vsk is dictionary containing
        subject measurement values.

        The previously loaded motion_data and vsk_data in setup_class() is used
        for testing.

        This function only tests the usage of start and end to change the range
        of the calcuation. We test that Calc is accurate by testing for the 
        first 5 angles and a joint center.
        """
        angles, jcs = pycgmCalc.Calc(start, end, self.motion_data, self.cal_SM)
        np.testing.assert_equal(len(angles), expected_len_result)
        np.testing.assert_equal(len(jcs), expected_len_result)
        np.testing.assert_almost_equal(angles[0][:5], expected_first_angles, self.rounding_precision)
        np.testing.assert_almost_equal(jcs[0]['LHumerus'], expected_LHumerus, self.rounding_precision)
    
    @pytest.mark.parametrize("start, end", [
        (0, 0),
        (1, 0),
        (-1, 100),
        (4000, 4001)
    ])
    def test_Calc_exceptions(self, start, end):
        """
        Test exceptions caused by invalid start and end values.
        """
        with pytest.raises(Exception):
            pycgmCalc.Calc(start, end, self.motion_data, self.cal_SM)
    
    def test_calcFrames(self):
        """
        This function tests pycgmCalc.calcFrames(data, vsk), where
        data is an array of dictionaries containing motion capture data, and vsk
        is a dictionary containing subject measurement values.

        We test the accuracy of calcFrames by testing for the first
        five returned angles and several returned subject measurement
        values.
        """
        angles, joint_centers = pycgmCalc.calcFrames(self.motion_data[0:1], self.cal_SM)

        expected_angles = np.array([-0.30849491,-6.12129279,7.5714311,2.91422293,-6.86706898])
        result_angles = angles[0][:5]

        expected_Front_Head = np.array([ 255.19071198,  406.12081909, 1721.92053223])
        expected_LHip = np.array([182.57097863, 339.43231855, 935.52900126])
        expected_RHand = np.array([ 859.80614366,  517.28239823, 1051.97278944])
        expected_Thorax = np.array([256.149810236564, 364.3090603933987, 1459.6553639290375])
        expected_LKnee = np.array([143.55478579, 279.90370346, 524.78408753])
        expected_joints = [expected_Front_Head, expected_LHip, expected_RHand, expected_Thorax, expected_LKnee]

        result_joints = [
            joint_centers[0]['Front_Head'],
            joint_centers[0]['LHip'],
            joint_centers[0]['RHand'],
            joint_centers[0]['Thorax'],
            joint_centers[0]['LKnee']
        ]

        np.testing.assert_almost_equal(result_angles, expected_angles, self.rounding_precision)
        np.testing.assert_almost_equal(result_joints, expected_joints, self.rounding_precision)