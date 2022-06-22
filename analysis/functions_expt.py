# Python functions used for analysis of experimental data

import pandas as pd
import os, sys
import numpy as np
import joblib
import warnings
import re

import cooler
#from mirnylib.genome import Genome
#from mirnylib.numutils import  zoomArray, completeIC, observedOverExpected, coarsegrain, ultracorrect
#from mirnylib.h5dict import h5dict
#from mirnylib import genome
import matplotlib
import matplotlib.pyplot as plt
import bioframe
import math
from scipy.linalg import toeplitz

import pyBigWig

def pileupGenes(GenePositions,filename,pad=500000,doBalance=False,
                TPM=0,CTCFWapldKO=False,TPMlargerthan=True,
                minlength=0,maxlength=5000000,OE=None, useTTS=False):
    """
    This function piles up Hi-C contact maps around genes, centered on TSSs or TTSs.
    Inputs
    ------
    GenePositions - pandas dataframe - with genes and their transcription intensity
    filename - str - is path to cooler file
    pad - int - half of the window size in bp
    OE - str or None - path to scaling data to use as "expected" to compute observed over expected
    useTTS - bool - False to pile on TSS, True to pile on TTS

    other parameters do some optional filtering of the data frame
    """

    sortString="start"
    if useTTS:
        sortString="end"
    OrderedPositions=GenePositions.sort_values(sortString)
    c = cooler.Cooler(filename)
    res = c.info['bin-size']
    chromsizes = bioframe.fetch_chromsizes('mm9')
    chrmList = list(chromsizes.index)
    runningCount = 0
    pile = []

    for mychr in chrmList: #Iterate over chromosomes
        mychrstrCooler=mychr
        mychrstrDataFrame=mychr#Chromosomes in the dataframe GenePositions 
        #are labeled 1 to 19, X Y and M, while in the cooler file they are labeld 0 to 21

        current = OrderedPositions[OrderedPositions["chrom"] ==  mychrstrDataFrame]

        if len(current) <= 0:
            continue

        #identify + and - so we can reorient genes
        #genes for which strand is +, and current gene is not too long and not too short
        currentPlusStrand=current[(current['strand']=='+')&(current['gene_length']<maxlength)
                                  &(current['gene_length']>minlength)]
        #genes for which strand is -, and current gene is not too long and not too short
        currentMinusStrand=current[(current['strand']=='-')&(current['gene_length']<maxlength)
                                   &(current['gene_length']>minlength)]

        if TPMlargerthan: #filter by TPM > threshold
            if CTCFWapldKO:
                currentPlusStrand=currentPlusStrand[(currentPlusStrand['TPM_dKO+Adeno-Cre_30251-30253']>=TPM)] 
                currentMinusStrand=currentMinusStrand[(currentMinusStrand['TPM_dKO+Adeno-Cre_30251-30253']>=TPM)] 
            else:
                currentPlusStrand=currentPlusStrand[(currentPlusStrand['TPM_wildtype']>=TPM)]
                currentMinusStrand=currentMinusStrand[(currentMinusStrand['TPM_wildtype']>=TPM)]
        else: #filter by TPM < thresh
            if CTCFWapldKO:
                currentPlusStrand=currentPlusStrand[(currentPlusStrand['TPM_dKO+Adeno-Cre_30251-30253']<=TPM)&(currentPlusStrand['next_TPM_dKO']>0)]
                currentMinusStrand=currentMinusStrand[(currentMinusStrand['TPM_dKO+Adeno-Cre_30251-30253']<=TPM)
                                                      &(currentPlusStrand['TPM_dKO+Adeno-Cre_30251-30253']>0)]
            else:
                currentPlusStrand=currentPlusStrand[(currentPlusStrand['TPM_wildtype']<=TPM)
                                                    &(currentPlusStrand['next_TPM_wildtype']>0)]
                currentMinusStrand=currentMinusStrand[(currentMinusStrand['TPM_wildtype']<=TPM)
                                                      &(currentMinusStrand['TPM_wildtype']>0)]

        centerString="start"
        if useTTS:
            centerString="end"
        for st, end in zip(currentPlusStrand[centerString].values, currentPlusStrand[centerString].values):

            reg1 = '{}:{}-{}'.format(mychrstrCooler, int(np.floor((st - pad) / res) * res),
                                     int(np.floor((st + pad) / res) * res),)
            reg2 = '{}:{}-{}'.format(mychrstrCooler,int(np.floor((end - pad) / res) * res),
                                     int(np.floor((end + pad) / res) * res))

            #from balanced matrix, fetch regions
            try:
                mat = c.matrix(balance=doBalance).fetch(reg1, reg2)
                if OE!=None:#Divide by expected
                    mat=mat/OE[mychr]   
                pile.append(mat)
            except Exception as e:
                print(e)
                #mat = np.nan * np.ones((pad * 2 //res, pad * 2 //res))
                print('Cannot retrieve a window:', reg1, reg2)

        centerString="end"
        if useTTS:
            centerString="start"
        for st, end in zip(currentMinusStrand[centerString].values, currentMinusStrand[centerString].values):
            reg1 = '{}:{}-{}'.format(mychrstrCooler, int(np.floor((st - pad) / res) * res),
                                     int(np.floor((st + pad) / res) * res),)
            reg2 = '{}:{}-{}'.format(mychrstrCooler,int(np.floor((end - pad) / res) * res),
                                     int(np.floor((end + pad) / res) * res))

            try:
                temp=c.matrix(balance=doBalance).fetch(reg1, reg2)
                if OE!=None:#Divide by expected
                    temp=temp/OE[mychr]
                mat = temp[::-1].T[::-1].T #Rotate matrix 180 degrees to align genes
                pile.append(mat)
            except Exception as e:
                print(e)
                #mat = np.nan * np.ones((pad * 2 //res, pad * 2 //res))
                print('Cannot retrieve a window:', reg1, reg2)

    return pile





def pileupPairs(IslandList,filename,pad=700000,OE=None):
    """
    Computes Hi-C contact maps around contact point (pixel) between pairs of genomic loci, such as cohesin islands.
    Inputs
    ------
    IslandList - dataframe with pairs of genomic loci
    filename - path to cooler with Hi-C data
    pad - half window size
    OE - path to "expected" / scaling data.  
    """
    #temp is a list of positions, chromosome,halfway position of centers between convergent genes
    print('new pile')
    c = cooler.Cooler(filename)
    res = c.info['bin-size'] 
    pile=[]
    IslandOrdered=IslandList#.sort_values('start')
    chromsizes = bioframe.fetch_chromsizes('mm9')
    chrmList = list(chromsizes.index)
    #chrmList = {'chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chrX','chrY','chrM'}
    runningCount = 0
    num=len(chrmList)
   
        
    for mychr in chrmList: #Iterate over chromosomes
        mychrstrDataFrame=mychr#Chromosomes in the dataframe GenePositions 
        #are labeled 1 to 19, X Y and M, while in the cooler file they are labeld 0 to 21
        
        currentIslands=IslandOrdered[IslandOrdered["chrom"] ==  mychrstrDataFrame]
        if OE!=None:
            gvec=OE[mychr]
            expvector=np.concatenate([gvec[-1:0:-1],gvec])
        # assert len(current) > 0  
        for st, end in zip(currentIslands["halfway"].values, currentIslands["halfway2"].values):
            reg1 = '{}:{}-{}'.format(mychrstrDataFrame, int(np.floor((st - pad) / res) * res),
                                 int(np.floor((st + pad) / res) * res),)
            reg2 = '{}:{}-{}'.format(mychrstrDataFrame,int(np.floor((end - pad) / res) * res),
            int(np.floor((end + pad) / res) * res))
            
            try:
                mat = c.matrix(balance=True).fetch(reg1, reg2)
                if OE!=None:#Divide by expected
                    expected=toeplitz(expvector[int(len(gvec)-1+np.floor((end-st)/res)):int(len(gvec)-1+np.floor((end-st-2*pad)/res)):-1],expvector[int(len(gvec)-1+np.floor((end-st)/res)):int(len(gvec)-1+np.floor((end-st+2*pad)/res))]) 
                    mat=mat/expected
                    #mat=mat/OE[mychr]
                pile.append(mat)
            except:
                #mat = np.nan * np.ones((pad * 2 //res, pad * 2 //res))
                print('Cannot retrieve a window:chr:'+str(mychrstrDataFrame))
            
    return pile



def computeDotStrength(mat, dotsize=5, shift=4):
    """
    Compute numerical value for strength of avg dot over background
    Inputs
    ------
    mat - Hi-C map / symmetric matrix containing dot
    dotsize - half window size around dot & of background regions
    shift - distance from dot window to set windows for background 
    """
    #assumes square matrix
    middle=np.shape(mat)[0]//2
    halfshift = shift//2

    background1= np.nansum(mat[middle-shift*dotsize:middle-halfshift*dotsize,middle-shift*dotsize:middle-halfshift*dotsize])
    background2= np.nansum(mat[middle+halfshift*dotsize:middle+shift*dotsize,middle+halfshift*dotsize:middle+shift*dotsize])

    dot= np.nansum(mat[middle-dotsize:middle+dotsize,middle-dotsize:middle+dotsize])
    #dot_mean=np.nanmean(mat[middle-dotsize:middle+dotsize,middle-dotsize:middle+dotsize])
 
   if (background1+background2)>0:
        dotstr= dot/(0.5*(background1+background2))

    return np.nanmean(dotstr)




def stackupChIP(chipseq,genepos,flank,pos='start'):
    """
    create stackup, every line of the matrix is chipseq around a gene 
    Inputs
    ------
    chipseq - BigWig file w/ chip-seq 
    genepos - DataFrame with gene positions
    flank - region on either side of pos to consider
    pos - str - name of column in dataframe containing positions on which to center regions, 
        e.g. 'start' for TSS, 'end' for TTS, or 'halfway' if stacking up on particular sites halfway between 2 sites
    """
    chromsizes = bioframe.fetch_chromsizes('mm9')
    chrmList = list(chromsizes.index)
    chrmList.sort()
    StackMatrix=np.zeros((len(genepos),2*flank))
    counter=0
    for mychr in chrmList:
        genepostemp=genepos[genepos['chrom']==mychr]
        
        #'pos' tells which column to use for positions of stackup
        for halfway in genepostemp[pos].values:
            try:
                #this is a matrix where each row is chip-seq in a gene
                StackMatrix[counter,:]=chipseq.values(mychr, 
                                                      int(np.floor(halfway))-flank, 
                                                      int(np.floor(halfway))+flank)
            except Exception as e:
                print(e)
                StackMatrix[counter,:]=np.nan*np.ones(2*flank)[:]
                print('cannot retrieve window')
            counter+=1
    return StackMatrix





def pileUpPoint(Positions,filename,pad=200000,doBalance=False,OE=None):
    """
    pile up on a particular site, e.g., to compute insulation 
    Positions is dataframe with positions to pile up, such as CTCF locations or Island centers
    filename is path to cooler file
    pad = half of the window size in bp
    """
    print('Next pileup')
    OrderedPositions=Positions.sort_values('start')
    c = cooler.Cooler(filename)
    res = c.info['bin-size'] 
    chromsizes = bioframe.fetch_chromsizes('mm9')
    chrmList = list(chromsizes.index)
    pile = []
    runningCount = 0
    
        
    for mychr in chrmList: #Iterate over chromosomes
        mychrstrCooler=mychr
        mychrstrDataFrame=mychr#Chromosomes in the dataframe GenePositions 
        #are labeled 1 to 19, X Y and M, while in the cooler file they are labeld 0 to 21
       
        current = OrderedPositions[OrderedPositions["chrom"] ==  mychrstrDataFrame]#
        
        if len(current) <= 0:
            continue
        
        for st, end in zip(current["start"].values, current["end"].values):
            halfway=int(st+(end-st)/2)
            reg1 = '{}:{}-{}'.format(mychrstrCooler, int(np.floor((halfway - pad) / res) * res),
                                 int(np.floor((halfway + pad) / res) * res),)
            reg2 = '{}:{}-{}'.format(mychrstrCooler,int(np.floor((halfway - pad) / res) * res),
            int(np.floor((halfway + pad) / res) * res))
        
            try:
                mat = c.matrix(balance=doBalance).fetch(reg1, reg2)
                if OE!=None:#Divide by expected
                    mat=mat/OE[mychr] 
                pile.append(mat)
            except Exception as e:
                print(e)
                
    return pile



def insul_diamond(mat, window=10, ignore_diags=2, normalize=True, useMean=True):
    """
    Adapted from cooltools

    Calculates the insulation score of a Hi-C interaction matrix using a diamond.
    Should include upstream and downstream contacts within window
    and be similar to what's in cooltools (need to check)
    
    
    Parameters
    ----------
    mat : numpy.array
        A dense square matrix of Hi-C interaction frequencies. 
        May contain nans, e.g. in rows/columns excluded from the analysis.
    
    window : int
        The width of the window to calculate the insulation score.
    ignore_diags : int
        If > 0, the interactions at separations <= `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals 
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.
    normalize : bool
        normalize ins. score by the median or not. 
        note nansum/median(nansum) for normalization vs. nanmean for unnormalized
    useMean: bool
        use nanmean to compute score instead of nansum
    
    """
    if (ignore_diags):
        mat = mat.copy()
        for i in range(-ignore_diags, ignore_diags+1):
            cooltools.lib.numutils.set_diag(mat, np.nan, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N = mat.shape[0]
        score = np.nan * np.ones(N)
        
        if normalize:
            for ii in range(window, N-window):
                if useMean:
                    score[ii]=np.nanmean(mat[ii-window:ii+1,ii:ii+window+1])
                else:
                    score[ii]=np.nansum(mat[ii-window:ii+1,ii:ii+window+1])
            score /= np.nanmedian(score)
        else:
            for ii in range(window, N-window):
                score[ii]=np.nanmean(mat[ii-window:ii+1,ii:ii+window+1])
                
    return score





