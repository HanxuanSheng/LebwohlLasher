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
from time import time
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

from libc.math cimport cos, sin, pi, exp, exp
from scipy.linalg import eig
from libc.stdlib cimport rand, RAND_MAX
import numpy as np
cimport numpy as cnp
cdef extern from "stdlib.h":
    double drand48()
    double sqrt(double)
    double log(double)

from cython.parallel import prange
cimport openmp

#=======================================================================
#def initdat(nmax):
cdef initdat(int nmax):
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
    # arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    # return arr
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr = np.empty((nmax, nmax), dtype=np.float64)
    cdef int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            arr[i, j] = drand48() * 2.0 * pi
    
    return arr

#=======================================================================
def plotdat(arr,pflag,nmax,threads):
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
                cols[i,j] = one_energy(arr,i,j,nmax,threads)
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
#def one_energy(arr,ix,iy,nmax):#耗时最长
cdef double one_energy(cnp.ndarray[cnp.float_t, ndim=2] arr, int ix, int iy, int nmax, int num_threads):
    cdef double en = 0.0
    cdef int ixp, ixm, iyp, iym
    cdef double ang, central_value
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
    #en = 0.0
   
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    central_value = arr[ix, iy]
    #只有四个数，numpy无用
    
    # with nogil:
    #     # 设置线程数，num_threads=4
    #     for ix in prange(nmax, schedule='static', num_threads=num_threads):
    #         en += 0.5 * (1.0 - 3.0 * cos(central_value - arr[ixp, iy])**2)
    #         en += 0.5 * (1.0 - 3.0 * cos(central_value - arr[ixm, iy])**2)
    #         en += 0.5 * (1.0 - 3.0 * cos(central_value - arr[ix, iyp])**2)
    #         en += 0.5 * (1.0 - 3.0 * cos(central_value - arr[ix, iym])**2)
    ang = central_value-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0 * cos(ang) * cos(ang))#把公式简化有用吗？仅有四个值，向量化的作用不大，反而会计算空间，np.sum也是大材小用
    ang = central_value-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0 * cos(ang) * cos(ang))
    ang = central_value-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0 * cos(ang) * cos(ang))
    ang = central_value-arr[ix,iym]
    en += 0.5*(1.0 - 3.0 * cos(ang) * cos(ang))
    
    return en
#=======================================================================
# def all_energy(arr,nmax):#可以向量化吗啊？
#     """
#     Arguments:
# 	  arr (float(nmax,nmax)) = array that contains lattice data;
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to compute the energy of the entire lattice. Output
#       is in reduced units (U/epsilon).
# 	Returns:
# 	  enall (float) = reduced energy of lattice.
#     """
#     en = 0.0
#     # 使用 periodic boundary conditions
#     shifted_xp = np.roll(arr, -1, axis=0)
#     shifted_xm = np.roll(arr, 1, axis=0)
#     shifted_yp = np.roll(arr, -1, axis=1)
#     shifted_ym = np.roll(arr, 1, axis=1)
    
#     # 计算角度差
#     ang_xp = arr - shifted_xp
#     ang_xm = arr - shifted_xm
#     ang_yp = arr - shifted_yp
#     ang_ym = arr - shifted_ym
    
#     # 计算能量贡献
#     en = 0.5 * (1.0 - 3.0 * np.cos(ang_xp) ** 2)
#     en += 0.5 * (1.0 - 3.0 * np.cos(ang_xm) ** 2)
#     en += 0.5 * (1.0 - 3.0 * np.cos(ang_yp) ** 2)
#     en += 0.5 * (1.0 - 3.0 * np.cos(ang_ym) ** 2)
    
#     return np.sum(en)

cdef double all_energy(cnp.ndarray[cnp.float_t, ndim=2] arr, int nmax,int num_threads):
    cdef int ix, iy
    cdef double total_energy = 0.0
    
    for ix in range(nmax):
        for iy in range(nmax):
            total_energy += one_energy(arr, ix, iy, nmax,num_threads)
    
    return total_energy 
# cdef double all_energy(cnp.ndarray[cnp.float_t, ndim=2] arr, int nmax, int num_threads):
#     cdef int ix, iy
#     cdef double total_energy = 0.0

#     # 使用 OpenMP 并行化外部循环
#     with gil:#(num_threads=num_threads):
#         for ix in prange(nmax):  # prange 自动处理线程分配
#             for iy in range(nmax):
#                 # 如果 one_energy 函数没有并行化，仍然可以并行调用
#                 total_energy += one_energy(arr, ix, iy, nmax, num_threads)

#     return total_energy
#=======================================================================
# def get_order(arr,nmax):
#     """
#     Arguments:
# 	  arr (float(nmax,nmax)) = array that contains lattice data;
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to calculate the order parameter of a lattice
#       using the Q tensor approach, as in equation (3) of the
#       project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
# 	Returns:
# 	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
#     """

cdef double get_order(cnp.ndarray[cnp.float64_t, ndim=2] arr, int nmax,int num_threads):
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] Qab = np.zeros((3, 3), dtype=np.complex128)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] delta = np.eye(3, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] lab
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] eigenvalues
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] eigenvectors
    cdef int a, b, i, j
    
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    # for a in range(3):
    #     for b in range(3):
    #         for i in range(nmax):
    #             for j in range(nmax):
    #                 Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    with nogil:  # 释放 GIL
        for a in range(3):
            for b in range(3):
                for i in prange(nmax, num_threads=num_threads):  # prange 自动处理线程分配
                    for j in range(nmax):
                        # 更新 Qab
                        Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.real.max()
#=======================================================================
#def MC_step(arr,Ts,nmax):
cdef double box_muller():
    cdef double u1, u2
    u1 = drand48()
    u2 = drand48()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


cdef double MC_step(cnp.ndarray[cnp.float_t, ndim=2] arr, double Ts, int nmax, int num_threads):
    cdef int ix, iy
    cdef double en0, en1, ang, accept = 0
    cdef double scale = 0.1 + Ts
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    for ix in range(nmax):
  
        for iy in range(nmax):
            ang = box_muller() * scale
            en0 = one_energy(arr, ix, iy, nmax,num_threads)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax,num_threads)

            if en1<=en0:#在实际物理系统中，如果某个格点的扰动导致其能量下降，它确实会释放能量，并趋于更稳定的状态。
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= drand48():#np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang 
            

          
    return accept/(nmax*nmax)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag,threads):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax,threads)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=float)
    ratio = np.zeros(nsteps+1,dtype=float)
    order = np.zeros(nsteps+1,dtype=float)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax,threads)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax,threads)

    # Begin doing and timing some MC steps.
    initial = time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax,threads)
        energy[it] = all_energy(lattice,nmax,threads)
        order[it] = get_order(lattice,nmax,threads)
    final = time()
    runtime = final-initial
    #order 会不降反增吗？ 真的是相关性吗？
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax,threads)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
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
#=======================================================================
