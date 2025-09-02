#################################################################################################
# FUNCTIONS USED TO SIMULATE CONTINUOUS TIME MARKOV CHAINS
#################################################################################################

#------------------------------------------------------------------------------------------------

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt 
import matplotlib.pyplot as plt
import multiprocess
from scipy.integrate import odeint

#------------------------------------------------------------------------------------------------

## Function to sample a state given a probability vector (multinomial distribution) 
#### Arguments:
## pvec : vector of probabilities (values in [0,1] summing to 1)

def sampleState(pvec):
    
    iSortpvec=np.argsort(pvec)  # Indices of sorted probability vector (greatest to smallest)
    U=np.random.uniform(low=0,high=1,size=1)[0] # Sample unifor number in [0,1]

    pTest=0
    for i in range(len(pvec)):
        itest=np.where(iSortpvec==i)[0][0]
        pTest=pTest+pvec[itest]
        if U<pTest:
            inext=itest
            break
        
    return inext


#------------------------------------------------------------------------------------------------


## Function to compute next jump of the markov chain
#### Arguments:
## Tcur : Time of the last jump recorded
## icur : Index of the current state
## N : Total number of states
## tauMat : Function returing the transition rate between i and j at time t (i.e. tau_{ij}(t))

def computeNextJump(Tcur,icur,N,tauMat):
    
    E=np.random.exponential(size=1)[0] # Sample from expoenential distribution

    # Function returning the i-th row sum of the tau matrix, at time t
    def tauMatRowsum(t,i,N):
        return np.array([tauMat(t,i,j) for j in range(N)]).sum()

    # Compute the next jump time
    def f0(t):
        return integrate.quad(tauMatRowsum, Tcur, Tcur+t, args=(icur,N))[0]-E
    res=opt.fsolve(f0, 0)   # Find the zero of the function f0
    Tnext=Tcur+res[0]    

    # Compute the probability vector of the states at the time of the jump
    lval=tauMatRowsum(Tnext,icur,N)    
    pvec=np.array([tauMat(Tnext,icur,j)/lval for j in range(N)])

    # Sample new state
    inext=sampleState(pvec)
    
    return [Tnext,inext]


#------------------------------------------------------------------------------------------------


## Function to simulate a markov chain on a time interval [0,Tmax]
#### Arguments:
## Tmax : Time horizon
## N : Total number of states
## i0 : Index of the initial state
## tauMat : Function returing the transition rate between i and j at time t (i.e. tau_{ij}(t))
#### Return: An array containing the times of the consectutive jumps and the correponding states

def simCTMC(Tmax,N,i0,tauMat):
    Tcur=0
    icur=i0
    mc=np.array([Tcur,icur])
    while Tcur < Tmax:
        Tcur, icur=computeNextJump(Tcur,icur,N,tauMat)
        mc=np.vstack([mc,[Tcur,icur]])

    if Tcur > Tmax:
        return mc[:(len(mc)-1)]
    else:
        return mc


#------------------------------------------------------------------------------------------------


## Function that evaluates the piecewise constant function corresponding to a markov chain defined from its jumps
#### Arguments:
## Tseq : Time steps on which the function is evaluated
## simMC : Jumps of a Markov chain (output of the function simCTMP)
## stateVal : Values associated to each state (if None, the indices of the states are used)

def computePCFunc(Tseq,simMC,stateVal=None):

    simTseq=np.zeros(len(Tseq))
    icur=0
    while icur <simMC.shape[0]:
        Tcur=simMC[icur,0]
        if stateVal is None:
            simTseq[Tseq>=Tcur]=int(simMC[icur,1])
        else:
            simTseq[Tseq>=Tcur]=stateVal[int(simMC[icur,1])]
        icur=icur+1

    return simTseq


#------------------------------------------------------------------------------------------------


## Function to plot the transition rates
#### Arguments:
## tauMat : Function returing the transition rate between i and j at time t (i.e. tau_{ij}(t))
## N : Total number of states
## Tseq : Time steps on which the function is evaluated
## ylim : Span of values on y-axis for the plots

def pltTauMatrix(tauMat,N,Tseq,ylim,plotLegend=True):
    fig, axs = plt.subplots(N, N)
    for i in range(N):
        for j in range(N):
            axs[i,j].plot(Tseq,[tauMat(t,i,j) for t in Tseq],label="i="+str(i)+", j="+str(j))
            axs[i,j].set_ylim(ylim)
            if plotLegend:
                axs[i,j].legend()
    plt.show()


#------------------------------------------------------------------------------------------------


## Function to compute the transition rate matrix (used to solve the Kolomogorv equations)
#### Arguments:
## t : time at which the matrix is evaluated
## tauMat : Function returing the transition rate between i and j at time t (i.e. tau_{ij}(t))
## N : Total number of states
## Tseq : Time steps on which the function is evaluated

def RFuncMat(t,N,tauMat):
    R=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                R[i,j]=tauMat(t,i,j)
    RsumVec=R.sum(1)
    for i in range(N):
        R[i,i]=-RsumVec[i]
    return R


#------------------------------------------------------------------------------------------------


## Function to compute the probability that the chain is at a given state, as a function of time, (using the Kolomogorv equations)
#### Arguments:
## Tseq : Time steps on which the function is evaluated
## tauMat : Function returing the transition rate between i and j at time t (i.e. tau_{ij}(t))
## N : Total number of states
## p0 : Probabilities for the initial state
#### Return: An array where each i-th row contains the probability that the chain is at state i at each time step in Tseq

def computeProbStates(Tseq,tauMat, N,p0):
    # function that returns dp/dt
    def model(p,t,N,tauMat):
        pMat=p.reshape((N,N))
        pdotMat=np.matmul(pMat,RFuncMat(t,N,tauMat))
        return pdotMat.flatten()

    # initial condition
    pInit = np.identity(N).flatten()
    # solve ODE
    p = odeint(model,pInit,Tseq,args=(N,tauMat))
    # Compute state probabilities
    pStates=np.array([np.matmul(p[t].reshape((N,N)).transpose(),p0) for t in range(p.shape[0])])

    return pStates.transpose()


#------------------------------------------------------------------------------------------------


## Function to compute the probability that the chain is at a given state, as a function of time, from simulations
#### Arguments:
## Tseq : Time steps on which the function is evaluated
## simFunc : Function returing a simulation and taking an initial state as sole argument
## N : Total number of states
## p0 : Probabilities for the initial state
## nbsim : number of simulations averaged to compute the probabilities
## nProcesses : Number of processes in parallel used to carry out the computation (None = Use all available CPUs)
#### Return: An array where each i-th row contains the probability that the chain is at state i at each time step in Tseq

def computeProbStatesFromSim(Tseq,simFunc,N,p0,nbsim,nProcesses=None):
    pMat=np.zeros((N,len(Tseq)))
    # Sample the initial states of each simulation to come
    i0Vec=[sampleState(p0) for i in range(nbsim)]

    with multiprocess.Pool(processes=nProcesses) as pool:
        for simTseq in pool.map(simFunc,i0Vec):
            for k in range(N):
                ik=np.where(simTseq==k)[0]
                pMat[k,ik]+=1

    pMat=pMat/nbsim

    return pMat


#------------------------------------------------------------------------------------------------