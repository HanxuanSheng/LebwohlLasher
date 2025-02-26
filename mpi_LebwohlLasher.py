"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpi4py import MPI
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

MAXWORKER  = 17          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
LTAG       = 2          # message tag
RTAG       = 3          # message tag
NONE       = 0          # indicates no neighbour
DONE       = 4          # message tag
MASTER     = 0          # taskid of first process
#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
   #按照行分的，但是列没有
    
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    central_value = arr[ix, iy]
   #注意两边的取值，运行的时候
    
    ang = central_value-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = central_value-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = central_value-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = central_value-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    
    return en

#=======================================================================
def all_energy(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    en = 0.0
    
    shifted_xp = np.roll(arr, -1, axis=0)
    shifted_xm = np.roll(arr, 1, axis=0)
    shifted_yp = np.roll(arr, -1, axis=1)
    shifted_ym = np.roll(arr, 1, axis=1)
    
    
    ang_xp = arr - shifted_xp
    ang_xm = arr - shifted_xm
    ang_yp = arr - shifted_yp
    ang_ym = arr - shifted_ym
    
    
    en = 0.5 * (1.0 - 3.0 * np.cos(ang_xp) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(ang_xm) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(ang_yp) ** 2)
    en += 0.5 * (1.0 - 3.0 * np.cos(ang_ym) ** 2)
    
    return np.sum(en)
#=======================================================================
def get_order(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    delta = np.eye(3)
    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)))  # 3 x nmax x nmax
    
    
    Qab = np.tensordot(lab, lab, axes=([1, 2], [1, 2]))  
    Qab = (3 * Qab - delta * (nmax * nmax)) / (2 * nmax * nmax)
    
    eigenvalues, _ = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
def MC_step(lattice,Ts,nmax,taskid,numtasks,comm):
    numworkers = numtasks-1
    lattice_old = lattice.copy()
   
#************************* master code *******************************/
    
    if taskid == 0:
        averow = nmax//numworkers
        extra = nmax%numworkers
        offset = 0

        for i in range(1,numworkers+1):
            rows = averow
            if i <= extra:
                rows+=1

            if i == 1:
                above = NONE
            else:
                above = i - 1
            if i == numworkers:
                below = NONE
            else:
                below = i + 1

        # Now send startup information to each worker
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)
            comm.Send(lattice[offset:offset+rows,:], dest=i, tag=BEGIN)
            offset += rows
            
         # Now wait for results from all worker tasks
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv([lattice[offset,:],rows*nmax,MPI.DOUBLE], source=i, tag=DONE)
            
        accept =1- np.sum(lattice_old == lattice)/(nmax*nmax)
        return accept

#************************* workers code **********************************/
    elif taskid != 0:
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)
        comm.Recv([lattice[offset:offset+rows, :],rows*nmax,MPI.DOUBLE], source=MASTER, tag=BEGIN)
        
        start=offset
        end=offset+rows-1
 
        scale=0.1+Ts

        if above != NONE:
            req=comm.Isend([lattice[offset,:],nmax,MPI.DOUBLE], dest=above, tag=RTAG)
            comm.Recv([lattice[offset-1,:],nmax,MPI.DOUBLE], source=above, tag=LTAG)
        if below != NONE:
            req=comm.Isend([lattice[offset+rows-1,:],nmax,MPI.DOUBLE], dest=below, tag=LTAG)
            comm.Recv([lattice[offset+rows,:],nmax,MPI.DOUBLE], source=below, tag=RTAG)


        for ix in range(start,end+1):
            for iy in range(nmax):
                ang = np.random.normal(loc=0.0, scale=1.0)
                en0 = one_energy(lattice,ix,iy,nmax)
                lattice[ix,iy] += ang
                en1 = one_energy(lattice,ix,iy,nmax)

                if en1>en0:
                    boltz = np.exp( -(en1 - en0) / Ts )

                    if boltz < np.random.uniform(0.0,1.0):
                        lattice[ix,iy] -= ang 
                        
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
        comm.Send([lattice[offset,:],rows*nmax,MPI.DOUBLE], dest=MASTER, tag=DONE)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1
    
    lattice = initdat(nmax)
    plotdat(lattice,pflag,nmax)

    energy = np.zeros(nsteps+1,dtype=float)
    ratio = np.zeros(nsteps+1,dtype=float)
    order = np.zeros(nsteps+1,dtype=float)
    
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)
 
    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax,taskid,numtasks,comm)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    

    global_order = comm.reduce(order[nsteps-1], op=MPI.SUM, root=0)

    # 只让主进程
    if taskid == 0:
        global_order /= numtasks 
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax, nsteps, temp, global_order, runtime))
        savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
        plotdat(lattice, pflag, nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
