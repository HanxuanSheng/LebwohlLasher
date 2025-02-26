import sys
from OpenMP_Cython_LebwohlLasher import main

if int(len(sys.argv)) == 6:
    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    NUMTHREADS = int(sys.argv[5])
    main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NUMTHREADS)
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG><THREADS>".format(sys.argv[0]))