import time
import c3d
import multiprocessing
import time
from math import *
import numpy as np
import xml.etree.ElementTree as ET
import sys
import getopt
from pyCGM import *
from pycgmIO import *
import pycgmStatic
import pycgmCalc

def main(argv):
    """
    Take in motion data file, time span
    returns output file of angles
    pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end> -c <calctype> -sf <singleframe>
    """
    start,end,singleframe,procs,inputvsk,staticfile,static,vsk = None,None,None,None,None,None,None,None

    flat_foot = False
    global vskdata
    
    inputfile="Jong 10 Min_Working_bck.c3d"
    outputfile="10Min_pycgm"
    inputvsk="Jong.vsk"
    staticfile="Jong Cal 03.c3d"
    calctype='multi'
    procs = 7

    totalTime=time.time()
    loadStaticTime=0
    loadVskTime=0
    loadDataTime=0
    calculateStaticTime=0
    calculateAnglesTime=0
    savaDataTime=0

    try:
        opts, args = getopt.getopt(argv,"h:i:o:s:e:c:f:p:v:x",["ifile=","ofile=","start=","end=","calctype=","singleframe=","nprocs=","vskfile=","staticinput="])
    except getopt.GetoptError:
        print 'pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end> -c <calctype> -sf <singleframe>'
        sys.exit(2)
            
    for opt, arg in opts:
        if opt == '-h':
                print 'pyCGM.py -i <motionFile> -o <outputfile> -s <start> -e <end> -c <calctype> -sf <singleframe> -p <procs>'
                sys.exit()
        elif opt in ("-i", "--ifile"):
                inputfile = arg
        elif opt in ("-o", "--ofile"):
                outputfile = arg+'.csv'
        elif opt in ("-s", "--start"):
                start = int(arg)
        elif opt in ("-e", "--end"):
                end = int(arg)
        elif opt in ("-c", "--calctype"):
                calctype = arg
        elif opt in ("-f","--singleframe"):
                singleframe = arg
        elif opt in ("-p","--nprocs"):
                procs = int(arg)
        elif opt in ("-v","--vskfile"):
                inputvsk = arg
        elif opt in ("-x","--staticinput"):
                staticfile = arg

    filename = './'+inputfile
    
    loadDataTime=time.time()
    motionData  = loadData(filename) 
    loadDataTime=time.time()-loadDataTime
    
    if len(motionData) == 0:
        print "No Data Loaded"
        sys.exit()
    
    if inputvsk != None:
        loadVskTime=time.time()
        vskdata = loadVSK(inputvsk)
        if vskdata!=None:
                vsk = createVskDataDict(vskdata[0],vskdata[1])
        loadVskTime=time.time()-loadVskTime
        
    if staticfile != None:
        loadStaticTime=time.time()
        staticData = loadData(staticfile)
        loadStaticTime=time.time()-loadStaticTime
        calculateStaticTime=time.time()
        calibratedMeasurements = pycgmStatic.getStatic(staticData,vsk,flat_foot)
        calculateStaticTime=time.time()-calculateStaticTime
        print calibratedMeasurements
            
    if calctype == 'single':
        multi=True
    else:
        multi=False

    calculateAnglesTime=time.time()
    result=pycgmCalc.calcAngles(motionData,start=start,end=end,cores=procs,vsk=calibratedMeasurements,multiprocessing=True,nprocs=8,splitAnglesAxis=False,formatData=False)
    calculateAnglesTime=time.time()-calculateAnglesTime
    print "CALCULATION TIME: ",calculateAnglesTime
    savaDataTime=time.time()
    writeResult(result,outputfile)
    savaDataTime=time.time()-savaDataTime
    
    totalTime=time.time()-totalTime

    printTimes=True
    if printTimes==True:
        print "Total time = %.10fs"%(totalTime,)
        totalTime=totalTime/100
        print "Total time to load data     \t= %.10fs\t%0.4f%%"%(loadStaticTime+loadVskTime+loadDataTime,(loadStaticTime+loadVskTime+loadDataTime)/totalTime)
        print "\tTime to load VSK            = %.10fs\t%0.4f%%"%(loadVskTime,(loadVskTime)/totalTime)
        print "\tTime to load static data    = %.10fs\t%0.4f%%"%(loadStaticTime,(loadStaticTime)/totalTime)
        print "\tTime to load data           = %.10fs\t%0.4f%%"%(loadDataTime,(loadDataTime)/totalTime)
        print "Total time for calculations \t= %.10fs\t%0.4f%%"%(calculateStaticTime+calculateAnglesTime,(calculateStaticTime+calculateAnglesTime)/totalTime)
        print "\tTime to calculate static    = %.10fs\t%0.4f%%"%(calculateStaticTime,(calculateStaticTime)/totalTime)
        print "\tTime to calculate angles    = %.10fs\t%0.4f%%"%(calculateAnglesTime,(calculateAnglesTime)/totalTime)
        print "Total time to save csv      \t= %.10fs\t%0.4f%%"%(savaDataTime,(savaDataTime)/totalTime)
    sys.exit()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(sys.argv[1:])
