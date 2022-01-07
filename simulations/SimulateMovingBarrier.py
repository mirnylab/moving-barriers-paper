import pickle
import os
import time
import numpy as np
import sys
import shutil
from openmmlib import openmmlib
from openmmlib import polymerutils
from openmmlib.polymerutils import scanBlocks
from openmmlib.openmmlib import Simulation
from openmmlib.polymerutils import grow_rw
import pyximport; pyximport.install()
from smcTranslocator_MovingBarrier import smcTranslocatorDirectional
import tools
import random

# -------defining parameters----------
# -- basic loop extrusion parameters--

logname="log.txt"
GPU = 0 
LIFETIME = 100   # Processivity of cohesin, Default: 100 for WT, 1000 for Wapl KO
SEPARATION = 200 # Separation LEFs in number of monomers, Default: 200 for WT, 100 for Wapl KO, assuming a  monomer size of 1kb
N = 10000                     # System size in number of monomers
smcStepsPerBlock = 1          # Number of LEF steps between blocks of polymer simulations, Default: 1
stiff = 1                     # Polymer siffness in unit of bead size, default: 1
dens = 0.2                    # density in beads / volume. The density can roughly be estimated by looking at the amount of DNA in a nuclear volume, Default: 0.2
box = (N / dens) ** 0.33      # Define size of the bounding box for Periodic Boundary Conditions, Default: 0.33
data = polymerutils.grow_rw(N, int(box) - 2)  # creates a compact conformation 
block = 0  # starting block 
stg = 0.8# same as Banigan, van den Berg, Brandao eLife 2020 #0.1  #stall probability at ctcf sites 
unstallLEFRate = 0.005 # about once per two typical LEF lifetimes.
ctcf_interval = 300 # 300 kb, e.g., see Busslinger et al. Nature 2017

# -- LEF and transcription dynamics -- 

# Note that the each rate can have a maximum value of 1. 
lef_speed=1.0
lefperm = 0. # controls amount of LEF-LEF bypassing we have
pauseArray = np.ones(N,dtype=np.double) # The speed of LEFS at each position. Default is an array of ones, which means that LEFS go on maximum speed everywhere. To simulate the presence of CTCF Aafke reduced the speed at CTCF sites to 0.005. However, this doesn't account for directionality of CTCF!
shrinkPauseArray=np.zeros(N,dtype=np.double)# speed at which loops shrink
shrink_speed=0.0
kinPol= 0.001     # PolII initiation rate, Default range: 0.00025-0.002
kterPol=0.002                 # Termination rate of PolII, Default range: 0.002-1.0. Normally transcription is initiation limited, so make sure that kinPol < kterPol
kterPolArray=np.zeros(N,dtype=np.double) # fix this later based on gene stucture
PolSpeed=0.1                  # The speed of PolII as a fraction of the speed of cohesin. Default value: 0.1
poldissoc=0.
#pauseArrayPol=np.zeros(N,dtype=np.double) + PolSpeed # The speed of PolII at each lattice site #initialize later
polloading=np.array([950,1950,2950,3950,4950,5950,6950,7950,8950]) # PolII initiation sites. The direction of the gene is set by the relative position of the polloading and poltermination sites.
#np.array([700,903,1100,1303,1700,1903,2100,2303,2700,2903,3100,3303,3700,3903,4100,4303,4700,4903,5100,5303,5700,5903,6100,6303])
poltermination=np.array([1850,2850,3850,4850,5850,6850,7850,8850,9850]) #PolII termination sites. At a termination site, PolII stalls with probability 1 and then unloads with rate kterPol. To simulate a broad PolII unloading area, I would set the poltermination site far beyond the gene length and then define wide region where PolII can stall, as shown on the lines below
#np.array([800,803,1200,1203,1800,1803,2200,2203,2800,2803,3200,3203,3800,3803,4200,4203,4800,4803,5200,5203,5800,5803,6200,6203])
stalProbPol=np.zeros(N,dtype=np.double) # The rate of PolII stalling. Once stalled, PolII unloads with a rate 'kterPol'. One can choose a single stall site, or a range of stall sites. If this array is set to zero everywhere, PolII will stall at the defined termination sites.
stalProb=0.001                
stall_in_gene=0.
unstall_in_gene=1.
unstallArray=np.zeros(N,dtype=np.double) #array for unstalling in of Pol II in gene

STALL_FROM_FILE = False # option to take stall probabilities from file for spatially varying patterns 
stallfile=""
gene_stall=[]
ctcf_left_list=[]
ctcf_right_list=[]
strongCTCFstall = 0

TSSloadbias = 1.0
TSSloadstart=0 #use this and TSSloadend to control offset/width of loading near TSS,  note: positive TSSloadstart denotes num sites before TSS, positive TSSloadend denotes sites after TSS
TSSloadend=0

base_genelen=200
#genelength=[base_genelen]*len(polloading)
variable_genelength = 0 
fixed_variable_genelength=0
variable_permeability = 0 #This varaible toggles on/off variable permeability of cohesin through RNAp depending on position in gene, 0 or 1 for constant permeability or variable perm, respectively
variable_type = 0 #type of variable permeability: 0- linearly decreasing, 1- step function
variable_offset = 0 #param for var perm function: for linear this is a constant offset from 0, for step this is offset for the bottom step
variable_pos = base_genelen // 2 #param: irrelevant for linear, position of step for step function.
variable_return = 0 #return to max permeability after passing end of gene but before reaching termination site
variable_TSSfactor = 1.0  # factor by which permeability at TSS is lower (or higher)

permLeftArray=np.zeros(N,dtype=np.double)
permRightArray=np.zeros(N,dtype=np.double)

collisionLifeFactor=1.0 # factor by which head-on collision with RNAP changes lifetime of cohesin
collisionLife=LIFETIME*collisionLifeFactor
TTSunload=1.0 # factor by which cohesin life time is changed near TTS
unloadZone=10 # width of zone in which cohesin life is changed by TTS

PolPause=0.002   # Step rate PolII at TSS, Default: 0.002
#later:
#for i in polloading:
#    pauseArrayPol[i]=PolPause

L=0.          # Set permeability of PolII to cohesin. L=0 means PolII is impermeable. 
R=0.          # Set permeability of PolII to cohesin coming from the right

run_id=1


# -- polymer simulation settings --

steps = int(200*(smcStepsPerBlock)) # nr of 3D simulation blocks btw advancing LEFs. For deterministic stepping choose 200-250 steps per block, otherwise, rescale with stepping probability. When genes are sparse, smcStepsPerBLock is approximately the number of smc steps per smc block.

saveRNAP=True # whether or not to print RNAP positions w/ each printed block

saveEveryBlocks = int(200/(smcStepsPerBlock))  # number of blocks until polymer configuration is saved
skipSavedBlocksBeginning = int(20/(smcStepsPerBlock))  # how many blocks (saved) to skip after you restart LEF positions
#totalSavedBlocks = 5000  # how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
totalSavedBlocks = 4000  # how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
#restartMilkerEveryBlocks = int(200/(smcStepsPerBlock))   
restartMilkerEveryBlocks = int(400/(smcStepsPerBlock))   
#Only one Hamiltonian can be loaded at a time to the simkt, but we want to update the bonds every time a LEF steps. Loading a new Hamiltonian costs a lot of time. Instead we precalculate bonds and load all positions at once as one big Hamiltonian and just change the prefactors. 

# parameters for smc bonds 

smcBondWiggleDist = 0.2
smcBondDist = 0.5


#if len(sys.argv)!=8:
#    print("Warning: Number of input arguments != 8")
#    sys.exit('Number of input arguments is not correct')

FLAG="" #extra label for directory name

#######use custom class to parse inputs with keywords#######################

params= tools.argsList()
for p in params.arg_dict:
    print(p, params.arg_dict[p])

if "gpu" in params.arg_dict:
    GPU = int(params.arg_dict["gpu"])
if "lifetime" in params.arg_dict:
    LIFETIME = float(params.arg_dict["lifetime"])
if "separation" in params.arg_dict:
    SEPARATION = float(params.arg_dict["separation"])
if "initiation" in params.arg_dict:
    kinPol = float(params.arg_dict["initiation"])
if "termination" in params.arg_dict:
    kterPol = float(params.arg_dict["termination"])
if "dissociation" in params.arg_dict:
    poldissoc= float(params.arg_dict["dissociation"])
if "stall" in params.arg_dict: #pol stall
    stalProb = float(params.arg_dict["stall"])
if "stallgene" in params.arg_dict:
    stall_in_gene = float(params.arg_dict["stallgene"])
if "unstall" in params.arg_dict: # pol unstall
    unstall_in_gene=float(params.arg_dict["unstall"])
if "lefspeed" in params.arg_dict:
    lef_speed=float(params.arg_dict["lefspeed"])
    pauseArray= lef_speed*pauseArray
if "lefperm" in params.arg_dict:
    lefperm=float(params.arg_dict["lefperm"])
if "shrink" in params.arg_dict:
    shrink_speed=float(params.arg_dict["shrink"])
    shrinkPauseArray= shrinkPauseArray + shrink_speed
    if shrink_speed+lef_speed>1.:
        print("WARNING: step+shrink > 1!! extrusion probabilities will not be computed correctly.\n")
if "lefstall" in params.arg_dict:
    stg = float(params.arg_dict["lefstall"])
if "lefunstall" in params.arg_dict:
    unstallLEFRate = float(params.arg_dict["lefunstall"])
if "polspeed" in params.arg_dict:
    PolSpeed = float(params.arg_dict["polspeed"])
if "polpause" in params.arg_dict:
    PolPause = float(params.arg_dict["polpause"])
if "permL" in params.arg_dict:
    L=float(params.arg_dict["permL"])
if "permR" in params.arg_dict:
    R=float(params.arg_dict["permR"])
if "collisionlife" in params.arg_dict:
    collisionLifeFactor=float(params.arg_dict["collisionlife"])
if "tssload" in params.arg_dict:
    TSSloadbias=float(params.arg_dict["tssload"])
if "tssloadstart" in params.arg_dict:
    TSSloadstart=int(params.arg_dict["tssloadstart"])
if "tssloadend" in params.arg_dict:
    TSSloadend=int(params.arg_dict["tssloadend"])
if "ttsunload" in params.arg_dict:
    TTSunload=float(params.arg_dict["ttsunload"])
if "ttszone" in params.arg_dict:
    unloadZone=int(params.arg_dict["ttszone"])
if "genelen" in params.arg_dict:
    base_genelen=int(params.arg_dict["genelen"])
if "vperm" in params.arg_dict:
    variable_permeability = int(params.arg_dict["vperm"])
if "vpermtype" in params.arg_dict:
    variable_type = int(params.arg_dict["vpermtype"])
if "vpermoffset" in params.arg_dict:
    variable_offset = float(params.arg_dict["vpermoffset"])
if "vpermpos" in params.arg_dict:
    variable_pos = int(params.arg_dict["vpermpos"])
if "vreturn" in params.arg_dict:
    variable_return = int(params.arg_dict["vreturn"])
if "vtss" in params.arg_dict:
    variable_TSSfactor = float(params.arg_dict["vtss"])
if "vgene" in params.arg_dict:
    variable_genelength = int(params.arg_dict["vgene"])
if "fixed_vgene" in params.arg_dict:
    fixed_variable_genelength=int(params.arg_dict["fixed_vgene"])
if "convergent" in params.arg_dict:
    if int(params.arg_dict["convergent"]):
        #print("warning only correct for genelength = 110 right now.\n")
        polloading=     np.array([840,  1160, 1640, 1960, 2440, 2760, 3240, 3560, 4040, 4360, 4840, 5160, 5640, 5960, 6440, 6760, 7240, 7560, 8040, 8360, 8840, 9160]) 
        poltermination= np.array([1000, 1001, 1800, 1801, 2600, 2601, 3400, 3401, 4200, 4201, 5000, 5001, 5800, 5801, 6600, 6601, 7400, 7401, 8200, 8201, 9000, 9001])
if "sparse" in params.arg_dict:
    if int(params.arg_dict["sparse"]):
        #print("warning only correct for genelength = 110 right now.\n")
        polloading= np.array([1950,3950,5950,7950])
        poltermination=np.array([2850,4850,6850,8850])
if "ctcfint" in params.arg_dict:
    ctcf_interval=int(params.arg_dict["ctcfint"])
if "ctcf" in params.arg_dict:
    if params.arg_dict["ctcf"] == "tss":
        #puts ctcf just before each tss, alternating left/right 
        ctcf_left_list = polloading[0::2] - 1
        ctcf_right_list = polloading[1::2] - 1
    elif params.arg_dict["ctcf"] == "body":
        #puts ctcf halfway through the gene, assuming fixed gene length and genes pointing downstream
        ctcf_left_list = polloading[0::2] + base_genelen // 2
        ctcf_right_list = polloading[1::2] + base_genelen // 2
    elif params.arg_dict["ctcf"] == "distributed":
        #put in sites every ~300 kb
        Nctcf = int(N/ctcf_interval/2)
        ctcf_left_list = np.arange(1,Nctcf+2)*2*ctcf_interval-ctcf_interval
        ctcf_right_list = np.arange(1,Nctcf+1)*2*ctcf_interval
        ctcf_left_list=np.delete(ctcf_left_list, np.where(ctcf_left_list>=N)[0])
        ctcf_right_list=np.delete(ctcf_right_list, np.where(ctcf_right_list>=N)[0])
    elif params.arg_dict["ctcf"] == "distributed2":
        Nctcf = int(N/ctcf_interval/2)
        ctcf_left_list = np.arange(1,Nctcf+2)*2*ctcf_interval-int(3*ctcf_interval/2)
        ctcf_right_list = np.arange(1,Nctcf+1)*2*ctcf_interval-int(ctcf_interval/2)
        ctcf_left_list=np.delete(ctcf_left_list, np.where(ctcf_left_list>=N)[0])
        ctcf_right_list=np.delete(ctcf_right_list, np.where(ctcf_right_list>=N)[0])
    else:
        with open(params.arg_dict["ctcf"], "r") as ctcffile:
            ctcfdata=ctcffile.readlines()
            #should only be 2 lines
            entries=ctcfdata[0].split()
            ctcf_left_list = np.array([int(x) for x in entries])
            entries=ctcfdata[1].split()
            ctcf_right_list = np.array([int(x) for x in entries])
if "strongctcf" in params.arg_dict:
    strongCTCFstall= int(params.arg_dict["strongctcf"])
#granting command line control of these variables for flexibility:
if "save" in params.arg_dict:
    saveEveryBlocks = int(int(params.arg_dict["save"])/(smcStepsPerBlock))
if "skip" in params.arg_dict:
    skipSavedBlocksBeginning = int(int(params.arg_dict["skip"])/(smcStepsPerBlock))
if "total" in params.arg_dict:
    totalSavedBlocks = int(params.arg_dict["total"])
if "stallfile" in params.arg_dict:
    stallfile=params.arg_dict["stallfile"]
    STALL_FROM_FILE=True
if "restart" in params.arg_dict:
    restartMilkerEveryBlocks = int(int(params.arg_dict["restart"])/(smcStepsPerBlock))
if "log" in params.arg_dict:
    logname=params.arg_dict["log"]
if "flag" in params.arg_dict:
    FLAG=params.arg_dict["flag"]


###### a few variables to be initialized last b/c they depend on others ###
collisionLife= LIFETIME*collisionLifeFactor

pauseArrayPol=np.zeros(N,dtype=np.double) + PolSpeed ## The speed of PolII at each lattice site
for i in polloading:
    pauseArrayPol[i]=PolPause

genelength=[base_genelen]*len(polloading)
if variable_genelength:
    if fixed_variable_genelength:
        genelength=random.shuffle([80,81,82,84,86,90,95,100,110])
    else:
        for ii in range(len(genelength)):
            genelength[ii] = genelength[ii] + int(np.random.normal(0, np.sqrt(genelength[ii])))
    print("genelengths:", genelength)

#stalling, unstalling and termination in gene and after gene ends
for i,j, gl in zip(polloading,poltermination, genelength):
    if j>i: 
        stalProbPol[i:i+gl]=stall_in_gene 
        unstallArray[i:i+gl]=unstall_in_gene
        stalProbPol[i+gl:j]=stalProb # This line adds stall probability to all sites between end of gene (loading site + gene len) and the "termination site" (where pol stops with prob=1) 
               # int(poltermination[0]-polloading[0])
        kterPolArray[i+gl:j]=kterPol
        kterPolArray[j]=kterPol
    else: # flipped genes
        stalProbPol[i-gl+1:i+1]=stall_in_gene
        unstallArray[i-gl:i+1]=unstall_in_gene
        stalProbPol[j+1:i-gl+1]=stalProb
        kterPolArray[j+1:i-gl+1]=kterPol
        kterPolArray[j]=kterPol


#permeabilities
if not variable_permeability:
        permLeftArray += L
        permRightArray += R
else:
    for ii in range(len(genelength)):
        direc = 2*int(poltermination[ii]>polloading[ii])-1
        gene_end=genelength[ii]*direc+polloading[ii]
        RL_factor=1.
        if not L==0.:
            RL_factor=R/L
        if variable_type == 0:
            for j in range(polloading[ii], gene_end):
                permLeftArray[j] = L*(gene_end-j)/(gene_end-polloading[ii])*direc + variable_offset
                permRightArray[j] = R*(gene_end-j)/(gene_end-polloading[ii])*direc + variable_offset*RL_factor
            permLeftArray[polloading[ii]] *= variable_TSSfactor
            permRightArray[polloading[ii]] *= variable_TSSfactor
            if gene_end<poltermination[ii]:
                s1=gene_end
                s2=poltermination[ii]
            else:
                s1=poltermination[ii]+1
                s2=gene_end+1
            for j in range(s1,s2):
                if not variable_return:
                    permLeftArray[j] = variable_offset
                    permRightArray[j] = variable_offset*RL_factor  # note that RL_factor adjusts right-left permeability of RNAP, not head-on/co-directional collision permeability. should change that. 
                else:
                    permLeftArray[j] = L + variable_offset
                    permRightArray[j] = R + variable_offset*RL_factor
        elif variable_type == 1:
            perm_break=polloading[ii]+direc*variable_pos
            if polloading[ii] < perm_break:
                s1=polloading[ii]
                s2=perm_break
            else:
                s1=perm_break+1
                s2=polloading[ii]+1
            for j in range(s1,s2):
                permLeftArray[j] = L
                permRightArray[j] = R
            permLeftArray[polloading[ii]] *= variable_TSSfactor
            permRightArray[polloading[ii]] *= variable_TSSfactor
            if perm_break<gene_end:
                s1=perm_break
                s2=gene_end
            else:
                s1=gene_end+1
                s2=perm_break+1
            for j in range(s1,s2):
                permLeftArray[j] = variable_offset
                permRightArray[j] = variable_offset*RL_factor
            if gene_end<poltermination[ii]:
                s1=gene_end
                s2=poltermination[ii]
            else:
                s1=poltermination[ii]+1
                s2=gene_end+1
            for j in range(s1,s2):
                if not variable_return:
                    permLeftArray[j] = variable_offset
                    permRightArray[j] = variable_offset*RL_factor
                else:
                    permLeftArray[j] = L
                    permRightArray[j] = R

# -- data folder -
folder="data/"
#folder = "/net/levsha/scratch/ebanigan/txn/"


FullFileName=folder
while True:
    fname="extr_life{0}_sep{1}_density{2}_N{3}_kinpol{4}_kterPol{5}_TSSPause{6}_save{7}_total{8}_PSpeed{9}_stf{10}_stg{11}_th0.001_genelen{12}_stall{13}_L{14}_R{15}".format(LIFETIME, SEPARATION, dens,N,kinPol,kterPol,PolPause,saveEveryBlocks,totalSavedBlocks,PolSpeed,stiff,stg,base_genelen,stalProb,L,R)
    if variable_permeability:
        fname=fname+"_vperm"+str(variable_type)+"_offset"+str(variable_offset)
        if variable_type == 1:
            fname=fname+"_pos"+str(variable_pos)
        if variable_return == 1:
            fname=fname+"_vret"
        if not (variable_TSSfactor == 1.):
            fname=fname+"_vtss"+str(variable_TSSfactor)
    if not (TSSloadbias == 1.):
        fname=fname+"_tssload"+str(TSSloadbias)
        if not ((TSSloadstart==0) and (TSSloadend==0)):
            fname=fname+"_st{0}end{1}".format(TSSloadstart,TSSloadend)
    if variable_genelength:
        fname=fname+"_vgenelen"
    if not (collisionLifeFactor == 1.0):
        fname=fname+"_collife"+str(collisionLife)
    if not (TTSunload == 1.):
        fname=fname+"_ttslife"+str(TTSunload*LIFETIME)+"width"+str(unloadZone)
    if poldissoc > 0.:
        fname=fname+"_dissoc"+str(poldissoc)
    if lef_speed < 1.0:
        fname=fname+"_lefspeed"+str(lef_speed)
    if shrink_speed > 0.0:
        fname=fname+"_shrink"+str(shrink_speed)
    if stall_in_gene > 0.:
        fname=fname+"_stallgene{0}_unstall{1}".format(stall_in_gene, unstall_in_gene)
    if lefperm > 0.:
        fname=fname+"_lefperm{0}".format(lefperm)
    fname=fname+FLAG
    fname=fname+"_"+str(run_id)
    FullFileName=os.path.join(folder, fname) 
    sleeptime=np.random.uniform(0,10)
    os.system("sleep {0}".format(sleeptime))
    if not os.path.exists(FullFileName):
        os.system("mkdir {0}".format(FullFileName))
        print("directory is {0}".format(FullFileName))
        break
    else:
        run_id = run_id+1


# -- Assertions to make sure parameters have been chosen correctly --

assert restartMilkerEveryBlocks % saveEveryBlocks == 0 
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert (totalSavedBlocks * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert smcStepsPerBlock<6 # max number of steps per smc block should not be too large to prevent 'jerky' polymer motion

savesPerMilker = restartMilkerEveryBlocks // saveEveryBlocks
milkerInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning  // restartMilkerEveryBlocks
milkerInitsTotal  = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartMilkerEveryBlocks
print("Milker will be initialized {0} times, first {1} will be skipped".format(milkerInitsTotal, milkerInitsSkip))

# create filenames for Ekin, Epot and time

Ekin_fname = os.path.join(FullFileName,'Ekin.txt')
Epot_fname = os.path.join(FullFileName,'Epot.txt')
time_fname = os.path.join(FullFileName,'time.txt')
Par_fname  = os.path.join(FullFileName,'Pars.txt')


def save_Es_ts_Rg():
    with open(time_fname, "a+") as time_file:
        time_file.write('%f\n'%(a.state.getTime()/openmmlib.ps))
    with open(Ekin_fname, "a+") as Ekin_file:
        Ekin_file.write('%f\n'%((a.state.getKineticEnergy())/a.N/a.kT))
    with open(Epot_fname, "a+") as Epot_file:
        Epot_file.write('%f\n'%((a.state.getPotentialEnergy()) /a.N /a.kT))

class smcTranslocatorMilker(object):

    def __init__(self, smcTransObject):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.smcObject = smcTransObject
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks = 100, smcStepsPerBlock = 1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """


        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        #precalculating all bonds
        allBonds = []
        for dummy in range(blocks):
            self.smcObject.steps(smcStepsPerBlock)
            left, right = self.smcObject.getSMCs()
            bonds = [(int(i), int(j)) for i,j in zip(left, right)]
            allBonds.append(bonds)

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, []))) # 'sum' preserves order and makes one long list with bonds, 'set' creates a set with left bonds from different time points ordered from small to large and eliminates two equal bonds (also if they were created by different LEFs at different times). List turns set into a list with unique bonds at different time points.

        # adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0) # pop(0) removes and returns first list of bonds

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curBonds, pastBonds

    def getRNAP(self):
        return self.smcObject.getPolOccupied()

def initModel():    
    birthArray = np.zeros(N,dtype=np.double) + 0.1
    #birthArray[polloading] *= TSSloadbias
    for nn, tt  in zip(polloading,poltermination):
        #the following accounts for different gene directions:
        if tt > nn:
            loadbegin=min(max(nn-TSSloadstart,0), N) #max/min construction here ensures range will not exceed ends of polymer, noting that TSSloadstart can be + or -
            loadend=max(min(nn+TSSloadend+1,N),0)
        else:
            loadbegin= min(max(nn-TSSloadend,0),N)
            loadend= max(min(nn+TSSloadstart+1,N),0)
        for jj in range(loadbegin,loadend):
            birthArray[jj] *= TSSloadbias
    deathArray = np.zeros(N, dtype=np.double) + 1. / (LIFETIME)
    collisionDeathArray= np.zeros(N,dtype=np.double) + 1./collisionLife
    if not (TTSunload==1.):
        for nn,tt,gg in zip(polloading, poltermination, genelength):
            if tt>nn:
                zonebegin=nn+gg
                zoneend=min(nn+gg+unloadZone,N)
            else:
                zonebegin= max(nn-gg+1-unloadZone,0)
                zoneend= nn-gg+1
            for jj in range(zonebegin, zoneend):
                deathArray[jj] = 1./(TTSunload*LIFETIME)
                collisionDeathArray[jj] =  1./(TTSunload*LIFETIME)

    stallLeftArray = np.zeros(N, dtype=np.double) #stall prob for left LEF legs, i.e. CTCF sites that point rightward
    stallRightArray = np.zeros(N, dtype=np.double)
    if STALL_FROM_FILE: #read in stall rates from file; file should provide list of LEF stalling rates starting from the TSS, extending as long as the configuration of repeats permits
        with open("stallfiles/"+stallfile,"r") as infile:
            stall_list = infile.readlines()
        for entry in stall_list:
            gene_stall.append(float(entry.split()[0]))
        for nn,tt in zip(polloading,poltermination):
            ii=0
            if nn<tt:
                gene_direc=1
            else:
                gene_direc=-1
            for ii in range(len(gene_stall)):
                stallLeftArray[nn+ii*gene_direc]=gene_stall[ii]
                stallRightArray[nn+ii*gene_direc]=gene_stall[ii]
    else: #stall rates set by stall params & CTCF position options
        stallLeftArray[ctcf_left_list] = stg  
        stallRightArray[ctcf_right_list] = stg

    unstallLEFArray = np.ones(N, dtype=np.double) * unstallLEFRate # generally have a single unstall rate.  doesn't matter if non-stall sites have unstall>0, since there won't be stalls there anyway
    stallDeathArray = np.zeros(N, dtype=np.double) + 1. / (LIFETIME) # unbinding rate during stall can be different, but we usually leave it the same 
    smcNum = int(N / SEPARATION)
    curPos = 0
        
    SMCTran = smcTranslocatorDirectional(birthArray, deathArray, stallLeftArray, stallRightArray, unstallLEFArray, pauseArray, stallDeathArray, smcNum, 
                                         kinPol,kterPolArray,pauseArrayPol,shrinkPauseArray, polloading,poltermination, stalProbPol, unstallArray, 
                                         PolPermL=permLeftArray, PolPermR=permRightArray, collisionFalloffProb=collisionDeathArray, 
                                         poldissoc= poldissoc, 
                                         LefPerm=lefperm,
                                         strongCTCF=strongCTCFstall
                                        )  
    return SMCTran


SMCTran = initModel()  # defining actual smc translocator object 



# now polymer simulation code starts

# ------------feed smcTran to the milker---
SMCTran.steps(1000000)  # first steps to "equilibrate" SMC dynamics. If desired of course. 
milker = smcTranslocatorMilker(SMCTran)   # now feed this thing to milker (do it once!)
#--------- end new code ------------

for milkerCount in range(milkerInitsTotal):
    doSave = milkerCount >= milkerInitsSkip
    
    # simulation parameters are defined below 
    a = Simulation(timestep=80, thermostat=0.001)#Collision rate in inverse picoseconds, low collistion rate means ballistic like motion, default in openmmpolymer is 0.001. Motion polymer is not diffusive, this is ok for statistical average,
    #but not for dynamics of the polymer
    a.setup(platform="CUDA", PBC=True, PBCbox=[box, box, box], GPU=GPU, precision="mixed")  # set up GPU here, PBC=Periodic Boundary Conditions. Default integrator is langevin with 300 K, friction coefficient of 1/ps, step size 0.002ps
    a.saveFolder(FullFileName)
    a.load(data)
    a.addHarmonicPolymerBonds(wiggleDist=0.1) # WiggleDist controls distance at which energy of bond equals kT
    if stiff > 0:
        a.addGrosbergStiffness(stiff) # Chain stiffness is introduced by an angular potential U(theta)=stiff(1-cos(theta-Pi))
    a.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05) #Polynomial repulsive potential between particles. Has value trunc=3.0 at zero, stays flat until 0.6-0.7 and then drops to zero. For attraction between a selective set of particles, use LeonardJones or addSelectiveSSWForce (see blocks.py or ask Johannes)
    a.step = block

    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0} 
    milker.setParams(activeParams, inactiveParams)
     
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.forceDict["HarmonicBondForce"],
                 blocks=restartMilkerEveryBlocks,   # default value; milk for 100 blocks
                 smcStepsPerBlock=smcStepsPerBlock)  # 
    print("Restarting milker")

    a.doBlock(steps=steps, increment=False)  # do block for the first time with first set of bonds in
    #print('done 1')
    for i in range(restartMilkerEveryBlocks - 1):
        #print(i)
        curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
        if i % saveEveryBlocks == (saveEveryBlocks - 2):  
            a.doBlock(steps=steps, increment = doSave)
            if doSave: 
                a.save()
                pickle.dump(curBonds, open(os.path.join(a.folder, "SMC{0}.dat".format(a.step)),'wb'))
                save_Es_ts_Rg() # save energies and time
                if saveRNAP:
                    pickle.dump(milker.getRNAP(), open(os.path.join(a.folder, "RNAP{0}.dat".format(a.step)), 'wb'))
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)

    data = a.getData()  # save data and step, and delete the simulation
    block = a.step
    del a
    
    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

with open(Par_fname,"a+") as Parfile:
    Parfile.write(" tau="+str(LIFETIME)+"\n Separation="+str(SEPARATION)+"\n N="+str(N)+"\n smcStepsPerBlock="+str(smcStepsPerBlock)+"\n stiff="+str(stiff)+"\n dens="+str(dens)+"\n block="+str(block)+"\n  SaveEveryBlocks="+str(saveEveryBlocks)+"\n skipSavedBlocksBeginning="+str(skipSavedBlocksBeginning)+"\n totalSavedBlocks="+str(totalSavedBlocks)+"\n restartMilkerEveryBlocks="+str(restartMilkerEveryBlocks)+"\n smcBondWiggleDist="+str(smcBondWiggleDist)+"\n smcBondDist="+str(smcBondDist)+"\n SmcTimestep=1\n NumMonos"+str(N))


os.system("mv {0} {1}".format(logname, FullFileName))


