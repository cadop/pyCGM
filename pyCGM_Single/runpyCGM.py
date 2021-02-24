#pyCGM

# Copyright (c) 2015 Mathew Schwartz <umcadop@gmail.com>
# Core Developers: Seungeun Yeon, Mathew Schwartz
# Contributors Filipe Alves Caixeta, Robert Van-wesep
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

###########
#To Run, Copy the sample files into the same directory as the code, then in a terminal type:
# python runpyCGM.py -i ReplicateRobotGait01.c3d -o test -v Jisub.vsk --staticinput "Jisub Cal 01.c3d"
#This will output a test.csv file with the results
##########

import sys
import getopt
# Relative imports 
from . import pycgmStatic
from . import pycgmIO
from . import pycgmCalc

def main(argv):
    """
    Take in motion data file, time span
    returns output file of angles
    pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end>
    """
    flat_foot = False
    global vskdata
    start,end = None,None
    try:
        opts, args = getopt.getopt(argv,"h:i:o:s:e:v:x",["ifile=","ofile=","start=","end=","vskfile=","staticinput="])
    except getopt.GetoptError:
        print('pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
                print('pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end>')
                sys.exit()
        elif opt in ("-i", "--ifile"):
                inputfile = arg
        elif opt in ("-o", "--ofile"):
                outputfile = arg
        elif opt in ("-s", "--start"):
                start = int(arg)
        elif opt in ("-e", "--end"):
                end = int(arg)
        elif opt in ("-v","--vskfile"):
                inputvsk = arg
        elif opt in ("-x","--staticinput"):
                staticfile = arg

# TODO -x is not working for input

    filename = './'+inputfile
    motionData  = pycgmIO.loadData(filename)
    if len(motionData) == 0 or motionData == None:
        print("No Data Loaded")
        sys.exit()

    if inputvsk != None:
        vskdata = pycgmIO.loadVSK(inputvsk)
        if vskdata!=None:
                vsk = pycgmIO.createVskDataDict(vskdata[0],vskdata[1])

    if staticfile != None:
        staticData = pycgmIO.loadData(staticfile)
        calibratedMeasurements = pycgmStatic.getStatic(staticData,vsk,flat_foot)

    result=pycgmCalc.calcAngles(motionData,start=start,end=end,vsk=calibratedMeasurements,splitAnglesAxis=False,formatData=False)

    pycgmIO.writeResult(result,outputfile)

    sys.exit()

if __name__ == '__main__':
    main(sys.argv[1:])
