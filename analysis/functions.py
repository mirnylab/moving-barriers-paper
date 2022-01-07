#More to come...

import pandas as pd
import os, sys
import numpy as np
import joblib
import warnings
import re

import cooler
from mirnylib.genome import Genome
from mirnylib.numutils import  zoomArray, completeIC, observedOverExpected, coarsegrain, ultracorrect
from mirnylib.h5dict import h5dict
from mirnylib import genome
import matplotlib
import matplotlib.pyplot as plt
try:
    from bioframe.tools import bedtools, tsv
    print("using old version of bioframe")
except ModuleNotFoundError:
    from bioframe.util import bedtools, tsv
import bioframe
import math
from scipy.linalg import toeplitz



def pile_up_genes(GenePositions,filename,pad=500000,doBalance=False,
                  TPM=0,CTCFWapldKO=False,TPMlargerthan=True,
                  minlength=0,maxlength=5000000,OE=None, useTTS=False):
    """
    This function piles up genes.
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
