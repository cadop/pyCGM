#TO Use:
# calculates a file based on the number of cores in total, across multiple nodes

#Example With Input Args:
# mpirun -c 12 python runpyCGM_MPI_Frames.py -o MPIFramesOutput -i Sample_ROM.c3d -v Sample_SM.vsk -x Sample_Static.c3d

import time
import numpy as np
import sys
import getopt
import pycgmIO 
import pycgmCalc
import pycgmStatic

from mpi4py import MPI

def mainMPI(argv):
    testtime =  time.strftime("%d_%H_%M_%S",time.gmtime())
    runfolder = str(testtime)+'/'
    pycgmIO.make_sure_path_exists(runfolder)
    
    try:
	opts, args = getopt.getopt(argv,"h:i:o:s:e:c:f:p:v:x:r",["ifile=","ofile=","start=","end=","calctype=","singleframe=","nprocs=","vskfile=","staticinput=","rank="])
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
		outputfile = arg
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
	elif opt in ("-r","--rank"):
		rank = arg

    #Get this processes rank and the size of the mpi rank call
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_size = comm.Get_size()
    
    #Set the variable motiondata to none at start for all ranks
    motiondata = None
    vsk = None
    static = None
    motiondata_lab = None
    if rank == 0:
        #File to use in calculation
        start = 0
        flat_foot = False
        filename = './'+inputfile

        #Time setup
        totalTime=time.time()
        loadDataTime=time.time()

        #Load motion data from file
        motionData  = pycgmIO.loadData(filename) 

        loadDataTime=time.time()-loadDataTime
        
        if len(motionData) == 0:
            print "No Data Loaded"
            sys.exit()
            
        #Load VSK File and parse into dict
        if inputvsk != None:
            loadVskTime=time.time()
            vskdata = pycgmIO.loadVSK(inputvsk)
            if vskdata==None:
                print "VSK Not Loaded"
            vsk = pycgmIO.createVskDataDict(vskdata[0],vskdata[1])
            loadVskTime=time.time()-loadVskTime
		
        if staticfile != None:
            #Load static file and time it
            loadStaticTime=time.time()
            staticData = pycgmIO.loadData(staticfile)
            loadStaticTime=time.time()-loadStaticTime
            #Calculate Static and time it
            calculateStaticTime=time.time()
            vsk = pycgmStatic.getStatic(staticData,vsk,flat_foot)
            calculateStaticTime=time.time()-calculateStaticTime

        #Split the motion data to labels and values to send faster with MPI
        # then combine again in each process
        motiondata_val,motiondata_lab = pycgmIO.splitDataDict(motionData)
		
    #Start timing the calculation time
    calculateAnglesTime=time.time()

    if rank == 0:
        #Split the data evenly for the processes  
        motionData_val_scatter = []
        motionData_val_scatter.append([None])
        l=len(motiondata_val)/(rank_size-1)
        for i in range(rank_size-1):
            start=i*l
            end=(i+1)*l
            if i == rank_size-2:
                end = len(motiondata_val)
            #Append each dataset as an array of the data, vsk info, and static
            motionData_val_scatter.append(motiondata_val[start:end])
            
        #Set the master rank single_result value to None so there is a variable
        # for it to read when gathering
        single_result = None
    else:
        motionData_val_scatter = []
        single_result = None
        
    motionData = comm.scatter(motionData_val_scatter, root=0)
    vsk = comm.bcast(vsk, root=0)
    static = comm.bcast(static, root=0)
    motiondata_lab = comm.bcast(motiondata_lab,root=0)
 
    if rank !=0:     
        #combine the values and labels back into a dict
        motionData = pycgmIO.combineDataDict(motionData,motiondata_lab)
        single_result=pycgmCalc.calcFramesMPI(motionData,vsk)

    result_total = comm.gather(single_result,root=0)
        
    if rank == 0:
        results = np.asarray(result_total)[1:]
        
        #Put all angles in a list instead of separate arrays
        angles = []
        for i in results:
            angles=angles+i
        result = angles
        
        #Check the time
        calculateAnglesTime=time.time()-calculateAnglesTime
        savaDataTime=time.time()
        #Write the result to file
        angles=['R Hip','L Hip','R Knee','L Knee','R Ankle','L Ankle']
        axis =["PELO","PELX","PELY","PELZ","HIPO","HIPX","HIPY","HIPZ","R KNEO","R KNEX","R KNEY","R KNEZ","L KNEO","L KNEX","L KNEY","L KNEZ","R ANKO","R ANKX","R ANKY","R ANKZ","L ANKO","L ANKX","L ANKY","L ANKZ","R FOOO","R FOOX","R FOOY","R FOOZ","L FOOO","L FOOX","L FOOY","L FOOZ"]

        pycgmIO.writeResult(result,runfolder+outputfile,axis=axis,angles=angles)
        savaDataTime=time.time()-savaDataTime

        totalTimes=time.time()-totalTime
        totalTime=totalTimes/100
        with open(runfolder+str(time.time())+'_Root_Node_Stats.txt','wb') as f:
            f.write("Total time = %.10fs"%(totalTimes,)+'\n')
            f.write( "Cores = "+str(rank_size)+'\n')
            f.write( "Total time to load data     \t= %.10fs\t%0.4f%%"%(loadStaticTime+loadVskTime+loadDataTime,(loadStaticTime+loadVskTime+loadDataTime)/totalTime)+'\n')
            f.write( "\tTime to load VSK            = %.10fs\t%0.4f%%"%(loadVskTime,(loadVskTime)/totalTime)+'\n')
            f.write( "\tTime to load static data    = %.10fs\t%0.4f%%"%(loadStaticTime,(loadStaticTime)/totalTime)+'\n')
            f.write( "\tTime to load data           = %.10fs\t%0.4f%%"%(loadDataTime,(loadDataTime)/totalTime)+'\n')
            f.write( "Total time for calculations \t= %.10fs\t%0.4f%%"%(calculateStaticTime+calculateAnglesTime,(calculateStaticTime+calculateAnglesTime)/totalTime)+'\n')
            f.write( "\tTime to calculate static    = %.10fs\t%0.4f%%"%(calculateStaticTime,(calculateStaticTime)/totalTime)+'\n')
            f.write( "\tTime to calculate angles    = %.10fs\t%0.4f%%"%(calculateAnglesTime,(calculateAnglesTime)/totalTime)+'\n')
            f.write( "Total time to save output      \t= %.10fs\t%0.4f%%"%(savaDataTime,(savaDataTime)/totalTime)+'\n')
            
        sys.exit()

if __name__ == '__main__':
	mainMPI(sys.argv[1:])
