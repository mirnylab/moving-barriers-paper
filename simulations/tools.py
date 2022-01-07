import os, sys, numpy
import numpy as np


"""
Library contains;
argList- class to parse argument list for compartment sim. uses speciifed keywords, but this kind of code
        could in principle be used elsewhere for arbitrary command line options
"""


###########################################################################
class argsList:
    #A simple class to parse an arg list of the form param1=value1, etc., with 
    #spaces delineating arguments. e.g., execute:
    #python run_sim.py param1=0 param2=1000 param3=True
    #Help option specific to my compartment sims, but basic structure can be reused.

    def __init__(self):
        self.arg_dict= {}
        self.parseArgs()
    
    def parseArgs(self):
        if "?" in sys.argv:
            print("""
                Format:
                    param1=value1 param2=value2 ...etc...
                Options:
                    Extrusion Params
                        lifetime - float - cohesin lifetime - default 100
                        separation - float - mean distance between cohesins - default 200
                        lef_speed - float - cohesin extrusion speed - default 1.0
                        lefperm - float - cohesin-cohesin bypassing rate - default 0.0
                        shrink - float - cohesin loop shrink speed - default 0.0
                        lefstall - float - cohesin stall prob upon encountering CTCF - default 0.8
                        lefunstall - float - cohesin unstall from CTCF - default 0.005
                    Transcription Params
                        initiation - float - polymerase initiation rate - default 0.001
                        termination - float - polymerase termination rate upon reaching termination region - default 0.002
                        stall - float - rate of polymerase stalling after transcribing gene - default 0.001
                        stallgene - float - rate of polymerase stalling while in the gene - default 0
                        unstall - float - rate of polymerase unstalling when stalled in gene - default 1.
                        polspeed - float - polymerase speed - default 0.1
                        polpause - float - step rate of polymerase at the TSS - default 0.002
                    Extrusion-transcription interactions
                        collisionlife - float - factor by which lifetime is altered upon collision of an SMC with an RNAP that pushes it - default 1.0
                        tssload - float - fold enhancement to loading at TSS relative to other sites - default 1.
                        tssloadstart - int - num sites before TSS to include in targeted loading, can be negative - default 0
                        tssloadend - int - num sites after TSS to include in targeted loading, can be negative - default 0
                        ttsunload - float - factor by which cohesin lifetime is altered near TTS - default 1.
                        ttszone - int - width of unloading zone near TTS - default 10
                        permL - float - permeability of RNApol to cohesin coming from left - default 0 
                        permR - float - permeability of RNApol to cohesin coming from right - default 0 
                        vperm - int - whether or not to use a variable permeability for extrusion through RNApol - default 0
                        vpermtype - int - 0 for linear dependence on position in gene, 1 for step function - default 0
                        vpermoffset - float - offset from 0 for permeability - default 0.
                        vpermpos - int - position in gene of step function step - default 100
                        vreturn - int - whether permeability returns to its max (TSS) value after passing end of gene - default 0
                        vtss - float - factor by which permeability at TSS differs from baseline - default 1.
                    System setup
                        genelen - int - length of genes - default 200
                        vgene - int - whether or not to use variable length for genes - default 0
                        fixed_vgene - int - use variable gene lengths (prespecified) - default 0
                        convergent - int - if true, sets up pairs of convergent genes, only works for genelength = 110 currently - default 0
                        sparse - int - if true, sets up genes in a sparse layout, only works for genelength = 110 currently - default 0 
                        stallfile - str - if present, gives a list of stalling probs within gene - default ""
                        ctcf - str - if present, gives option for CTCF configs: tss, body, or distributed - default ""
                        ctcfint - int - how far apart CTCFs in 'distributed' configuration will be - default 300
                        strongctcf - int - if true, RNAP does not push cohesin over CTCF site - default 0
                    Sim stuff
                        save - how frequently to save a block, in blocks - default 200
                        skip - how many blocks to skip at the beginning before beginning saves - default 20
                        total - how many total blocks to save - default 4000
                    flag - string - gets added to end of data directory - default ""
                    gpu - int - choice of GPU - default 0
                """)
            exit()
        
        if len(sys.argv) > 1:
            for element in sys.argv[1:]:
                var_name=""
                var_value=""

                if "=" not in element: #assume end of commands
                    break
                
                for k, char in enumerate(element):
                    if k < element.index("="):
                        var_name= var_name+char
                    elif k > element.index("="):
                        var_value= var_value+char
            
                self.arg_dict[var_name]=var_value



