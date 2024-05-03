# *********************************************************************
# Name: m5gpGlobals.py
# Description: Modulo que implementa variables y metodos globales
# para su uso comun en todos los modulos el sistema
# *********************************************************************


import math
import os
from pickle import TRUE
import csv
import sys
import warnings
import numpy as np
import atexit
from   pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
import pycuda.tools as tools
#import skcuda.cublas as cublas
from   datetime import datetime
import pandas as pd
from   random import randint
import time
import gc
import csv
from queue import LifoQueue


import pycuda.driver as pycuda
from pycuda.tools import make_default_context, DeviceMemoryPool, clear_context_caches


NOOP = -10013
PI = 3.14159265

MAX_R2_NEG  = -5000
MAX_RMSE = 9999999
MAX_CONSTANT = 999
MIN_CONSTANT = MAX_CONSTANT * (-1)

global device_id
global gpu_memory
global free_mem 

global sizeMemPopulation
global sizeMemIndividuals 
global sizeTournament


def pycudasetup(gpu_device_number=0):
    try:
        pycuda.init()
    except pycuda.LogicError:
        raise RuntimeError("Cannot initialize GPU device")
        
    #device_id = int(device_id) if device_id is not None else 0
    device = pycuda.Device(gpu_device_number)

    # check compute capability
    compute_capability = device.compute_capability()
    if compute_capability[0] < 3:
        raise RuntimeError("Unsupported GPU")

     # context  
    global context
    context = device.make_context()
    attrs=device.get_attributes()
  
    print("Device #0: %s" % ( device.name()))
    print(" Compute Capability: %d.%d" % device.compute_capability())
    print(" Total Memory: %s GB" % (device.total_memory()//(1024*1024*1024)))
    print("Succesfully initialized PYCUDA")
    return

# Function for garbage collection in CUDA
def pycuda_finish():
    global context
    context.pop()
    from pycuda.tools import clear_context_caches
    clear_context_caches()
    print("Finishing up PYCUDA")
    return

def WriteCSV_OpS(nFun, elapsed,Ops, fCreate=False) :
    return
    nOpS = Ops / elapsed
    fName = "M5GP_OpS.csv"
    if fCreate == True :
        if os.path.exists(fName):
            os.remove(fName)
    with open(fName, 'a', newline='') as file:
        writer = csv.writer(file)       
        writer.writerow([nFun, elapsed, Ops, nOpS])  
        file.close()  
    #print(f"{nFun} Time lapsed: {elapsed}, Ops: {Ops}, nOpS : {nOpS}")
    return

def Truncate(f, n) :
    return math.floor(f * 10 ** n) / 10 ** n

def bestIndividualInfo(config,  
                        dInitialPopulation,  
                        indexBestIndividual_p) :

    BestIndividualLength = 0
    numOpNOOP = 0
    umOpIf = 0
    numVars = 0
    numConst = 0
    numOps = 0
    numOpSin = 0
    numOpCos = 0
    numOpExp = 0
    numOpLog = 0
    numOpAbs = 0
    Expr = ""

    for i in range(config.GenesIndividuals):
        gene = dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i];

        if (gene != NOOP) :
            BestIndividualLength =  BestIndividualLength + 1
        # if (gene == NOOP) :
        #     numOpNOOP = numOpNOOP + 1
        # if (gene == -10001) :
        #     numOps = numOps + 1
        #     Expr = Expr + "+\t"
        # if (gene == -10002) :
        #     numOps = numOps + 1
        #     Expr = Expr + "-\t"
        # if (gene == -10003) :
        #     numOps = numOps + 1
        #     Expr = Expr + "*\t"
        # if (gene == -10004) :
        #     numOps = numOps + 1
        #     Expr = Expr + "/\t"
        # if (gene == -10005) :
        #     numOpSin = numOpSin + 1
        #     Expr = Expr + "sin\t"
        # if (gene == -10006) :
        #     numOpCos = numOpCos + 1
        #     Expr = Expr + "cos\t"
        # if (gene == -10007) :
        #     numOpExp = numOpExp + 1
        #     Expr = Expr + "exp\t"
        # if (gene == -10008) :
        #     numOpLog = numOpLog + 1
        #     Expr = Expr + "log\t"
        # if (gene == -10009) :
        #     numOpAbs = numOpAbs +1
        #     Expr = Expr + "abs\t"
        # if ((gene == -10010) or (gene == -10011) or (gene == -10012)) :
        #     numOpIf = numOpIf + 1
        #     Expr = Expr + "if\t"
        # if ((gene <= -1000) and (gene > -10000)) :
        #     numVars = numVars + 1
        #     Expr = Expr + "X"
        #     Expr = Expr + str((int)((gene+1000) * (-1)))
        #     Expr = Expr + "\t"
        # if ((gene >= (config.maxRandomConstant * (-1) )) and (gene <= config.maxRandomConstant)) :
        #     numConst = numConst + 1
        #     Expr = Expr + " "
        #     Expr = Expr + str(gene)
        #     Expr = Expr + "\t"
        # End if
    # End for
    return BestIndividualLength

# Obtiene la expresion del individuo
def getIndividualExpr(config,  
                        dInitialPopulation,  
                        indexBestIndividual_p) :
    BestIndividualLength = 0
    numOpNOOP = 0
    umOpIf = 0
    numVars = 0
    numConst = 0
    numOps = 0
    numOpSin = 0
    numOpCos = 0
    numOpExp = 0
    numOpLog = 0
    numOpAbs = 0
    Expr = ""

    maxVar = float((1000 + config.nvar -1) * (-1))
    for i in range(config.GenesIndividuals):
        gene = dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i];

        if (gene != NOOP) :
            BestIndividualLength =  BestIndividualLength + 1
        elif (gene == NOOP) :
            numOpNOOP = numOpNOOP + 1
        elif (gene == -10001) :
            numOps = numOps + 1
            Expr = Expr + "+\t"
        elif (gene == -10002) :
            numOps = numOps + 1
            Expr = Expr + "-\t"
        elif (gene == -10003) :
            numOps = numOps + 1
            Expr = Expr + "*\t"
        elif (gene == -10004) :
            numOps = numOps + 1
            Expr = Expr + "/\t"
        elif (gene == -10005) :
            numOpSin = numOpSin + 1
            Expr = Expr + "sin\t"
        elif (gene == -10006) :
            numOpCos = numOpCos + 1
            Expr = Expr + "cos\t"
        elif (gene == -10007) :
            numOpExp = numOpExp + 1
            Expr = Expr + "exp\t"
        elif (gene == -10008) :
            numOpLog = numOpLog + 1
            Expr = Expr + "log\t"
        elif (gene == -10009) :
            numOpAbs = numOpAbs +1
            Expr = Expr + "Abs\t"
        elif ((gene == -10010) or (gene == -10011) or (gene == -10012)) :
            numOpIf = numOpIf + 1
            Expr = Expr + "if\t"       
        #if ((gene <= -1000) and (gene > -10000)) :
        elif ((gene <= -1000) and ((gene >= maxVar)) and (gene.is_integer())) :
            numVars = numVars + 1
            Expr = Expr + "X"
            Expr = Expr + str((int)((gene+1000) * (-1)))
            Expr = Expr + "\t"
        elif ((gene >= (MIN_CONSTANT )) and (gene <= MAX_CONSTANT)) :
            numConst = numConst + 1
            Expr = Expr + " "
            Expr = Expr + str(gene)
            Expr = Expr + "\t"
        else :
            numConst = numConst + 1
            Expr = Expr + " "
            Expr = Expr + str(gene)
            Expr = Expr + "\t"            
        #End if
    #End for

    return Expr

# Obtiene el gen como expresion
def getGeneExp(config, gene) :
    Expr = ""

    maxVar = float((1000 + config.nvar -1) * (-1))
    if (gene == -10001) :
        Expr += "+"
    elif (gene == -10002) :
        Expr += "-"  
    elif (gene == -10003) :
        Expr += "*"
    elif (gene == -10004) :
        Expr += "/"
    elif (gene == -10005) :
        Expr += "sin"
    elif (gene == -10006) :
        Expr += "cos"
    elif (gene == -10007) :
        Expr += "exp"
    elif (gene == -10008) :
        Expr += "log"
    elif (gene == -10009) :
        Expr += "Abs"
    elif ((gene <= -1000) and ((gene >= maxVar)) and (gene.is_integer())) :
        Expr += "X_"
        Expr += str(int(((gene+1000) * (-1))))
    elif ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) :
        Expr += str(gene)
    else :
        Expr += str(gene)
    
    return Expr

# Obtiene las expresiones completas del modelo dentro del stack
def getModelExpr(config, Model) :
    lenIndiv = 0
    stackModel = LifoQueue()
    Expr = ""
    tmpExpr = ""

    maxVar = float((1000 + config.nvar -1) * (-1))

    lenModel = len(Model)
    for i in range(lenModel):
        gene = Model[i]
        if (gene == -11111) :
            break
        geneExpr = getGeneExp(config, gene)
        lenIndiv += 1

        # ********************************* Es una constante ************************************/
        if ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) : # Es una constante
            tmpExpr = "1:"
            tmpExpr += "("+str(geneExpr)+")"
            stackModel.put(tmpExpr)

        # ********************************* Es una variable ************************************/
        elif ((gene <= -1000) and ((gene >= maxVar)) and (gene.is_integer())) :  # Es una variable
            tmpExpr= "1:"
            tmpExpr += geneExpr
            stackModel.put(tmpExpr)

        # ************ Es un operador de Suma,Resta,Division o Multiplicacion ******************/
        elif ((gene == -10001) or (gene == -10002) or (gene == -10003) or (gene == -10004)) :
            # Es Suma,Resta,Division o Multiplicacion
            if (not stackModel.empty()) :
                tmp = stackModel.get() #Obtenemos el ultimo elemento del stack
                strCont = tmp[0 : tmp.find(":")]
                if (strCont.isnumeric()) :
                    tmpT = tmp
                    cont1 = int(strCont)
                    tmp  = tmp[tmp.find(":") + 1 : len(tmp)]
                    if (not stackModel.empty()) :
                        tmp2 = stackModel.get()
                        strCont = tmp2[0 : tmp2.find(":")]
                        cont2 = int(strCont)
                        tmp2  = tmp2[tmp2.find(":") +1 : len(tmp2)]
                        tmpExpr= str(cont1 + cont2 + 1)
                        tmpExpr += ":("
                        tmpExpr += tmp
                        tmpExpr += geneExpr
                        tmpExpr += tmp2
                        tmpExpr += ")"
                        stackModel.put(tmpExpr)
                    else :
                        stackModel.put(tmpT)               
                    #End if
                # End if
            # End if

        # ********* Es un operador de seno, coseno, exponente, logaritmo y absoluto ************/
        elif ((gene == -10005) or (gene == -10006) or (gene == -10007) or (gene == -10008) or (gene == -10009)) :
            if (not stackModel.empty()) :
                tmp = stackModel.get()
                strCont = tmp[0 : tmp.find(":")]
                if (strCont.isnumeric()) :
                    cont1 = int(strCont)
                    tmp  = tmp[tmp.find(":") +1 : len(tmp)]
                    tmpExpr= str(cont1 + 1)
                    tmpExpr += ":("
                    tmpExpr += geneExpr
                    tmpExpr += "("
                    tmpExpr += tmp
                    tmpExpr += ")"
                    tmpExpr += ")"
                    stackModel.put(tmpExpr)
                #end if
            # End if
        elif (gene == NOOP) :  # Es NoOP, no hacemos nada
            if (not stackModel.empty()) :
                g1 = 0
            # End if
        else :
            tmpExpr = "1:"
            tmpExpr += "("+str(geneExpr)+")"
            stackModel.put(tmpExpr)
        # End if
    # End for (individuals)

    stackLen = stackModel.qsize()

    stackExpr = []
    if (not stackModel.empty()) :
        for j in range(stackLen):
            tmpExpr = str(lenIndiv)
            tmpExpr += ":"
            tmpExpr += str(stackLen)
            tmpExpr += ":"
            tmpExpr += stackModel.get()
            #stackExpr.insert(0,tmpExpr) 
            stackExpr.append(tmpExpr)
    else :
        tmpExpr = str(lenIndiv)
        tmpExpr += ":"
        tmpExpr += str(stackLen)
        tmpExpr += ":" 
        stackExpr.append(tmpExpr)      
    #end if
 
    #IndivLen:StackLen:ModelLen:ModelExpr
    return stackExpr



def m4gpModel(config, Model, Coef, Intercep) :
    lenIndiv = 0
    stackModel = LifoQueue()
    m4gpModel = []
    Expr = ""
    tmpExpr = ""

    maxVar = float((1000 + config.nvar -1) * (-1))

    #print("maxvar:", maxVar)
    lenModel = len(Model)
    #print("getModelExpr. nvar:", config.nvar," Genes:",config.GenesIndividuals)
    for i in range(lenModel):
        gene = Model[i]
        if (gene == -11111) :
            break

        geneExpr = getGeneExp(config, gene)
        #print("GeneExpr (", i, "): ", gene, " - ", geneExpr)
        lenIndiv += 1

        # ********************************* Es una constante ************************************/
        if ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) : # Es una constante
            stackModel.put(gene)

        # ********************************* Es una variable ************************************/
        elif ((gene >= maxVar) and (gene <= -1000)) :  # Es una variable
            stackModel.put(gene)

        # ************ Es un operador de Suma,Resta,Division o Multiplicacion ******************/
        elif ((gene == -10001) or (gene == -10002) or (gene == -10003) or (gene == -10004)) :
            # Es Suma,Resta,Division o Multiplicacion
            tmpArr = []
            if (not stackModel.empty()) :
                tmp = stackModel.get() #Obtenemos el ultimo elemento del stack
                
                if (not stackModel.empty()) :
                    tmp2 = stackModel.get()
                    
                    tmpArr.append(tmp2)
                    tmpArr.append(tmp)
                    tmpArr.append(gene)

                    stackModel.put(tmpArr)
                else :
                    stackModel.put(tmp)
                # End if
            # End if

        # ********* Es un operador de seno, coseno, exponente, logaritmo y absoluto ************/
        elif ((gene == -10005) or (gene == -10006) or (gene == -10007) or (gene == -10008) or (gene == -10009)) :
            tmpArr = []
            if (not stackModel.empty()) :
                tmp = stackModel.get()
                tmpArr.append(tmp)
                tmpArr.append(gene)
                stackModel.put(tmpArr)
            # End if
        elif (gene == NOOP) :  # Es NoOP, no hacemos nada
            if (not stackModel.empty()) :
                g1 = g1
            # End if
        else :
            g1 = gene
        # End if
    # End for (individuals)

    stackLen = stackModel.qsize()
  
    #IndivLen:StackLen:ModelLen:ModelExpr
    return stackModel

def m4gpBuildExpr(tmp1, nvoModel) :
    if isinstance(tmp1, list):
        lenTmp = len(tmp1)
        for k in range(lenTmp):
            tmp4 = tmp1[lenTmp-k-1]
            nvoModel = m4gpBuildExpr(tmp4, nvoModel)
        nvoModel.append(-10001)
    # End for
    else :    
        nvoModel.insert(0,tmp1)
    return nvoModel
