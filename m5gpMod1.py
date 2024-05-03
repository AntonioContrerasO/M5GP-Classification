# *********************************************************************
# Name: m5gpMod1.py
# Description: Modulo que implementa metodos tipo wrapper para ejecutar
# metodos CUDA y CuML a traves de llamadas comunes
# Se implementa la logica de ejecucion para funciones de numba y CuML
# *********************************************************************

import math
import numpy as np

import time
import gc

import cupy as cp

from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32)

import m5gpGlobals as gpG
import m5gpCudaMethods as gpCuda
import m5gpCumlMethods as gpCuM

# *************************** Initialize population ******************************** 
def initialize_population (
        numIndividuals,
        nvar,
        sizeMaxDepthIndividual,
        maxRandomConstant,
        genOperatorProb,
        genVariableProb,
        genConstantProb,
        genNoopProb,
        useOpIF ) :
    
    
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"]

    # Initialize a state for each thread
    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)

    hInitialPopulation = np.zeros((gpG.sizeMemPopulation), dtype=np.float32) 
    dInitialPopulation = cuda.to_device(hInitialPopulation)

    start_time = time.time()
    gpCuda.initialize_population[blocksize, gridsize](cu_states,
                                        dInitialPopulation,
                                        numIndividuals,
                                        nvar,
                                        sizeMaxDepthIndividual,
                                        maxRandomConstant,
                                        genOperatorProb,
                                        genVariableProb,
                                        genConstantProb,
                                        genNoopProb,
                                        useOpIF ) 
    elapsed = time.time() - start_time

    hInitialPopulation = dInitialPopulation.copy_to_host()

    Ops = (numIndividuals * sizeMaxDepthIndividual)
    gpG.WriteCSV_OpS("InitialPopulation", elapsed,Ops,True)
 
    return hInitialPopulation
# -- End of Initialize population --

# ***************************  Compute Individuals  ****************************
def compute_individuals(
        hInitialPopulation,
        hData,
        numIndividuals,
        GenesIndividuals,
        nrowTrain,
        nvar,
        getStackModel ) :

  # Total elements of the data train matrix to form
  totalElements = nrowTrain * nvar

  # Memory size of the number of individuals in the initial population
  sizeMemIndividuals = numIndividuals 
  
  # Memory size of semantics for the entire population with training data
  sizeMemIndividualsTrain = numIndividuals * nrowTrain 

  # Memory size of the training data
  sizeMemDataTrain = totalElements

  sizeMemModel = GenesIndividuals * numIndividuals * nrowTrain
  sizeMemPopulation = numIndividuals * GenesIndividuals
  sizeMemStack = sizeMemPopulation * nrowTrain
  sizeMemStackIdx = sizeMemIndividuals  * nrowTrain

  # Calculate the available memory for slide individuals blocks 
  memRequired = (np.dtype(float).itemsize) * (sizeMemPopulation + 
                          sizeMemIndividualsTrain + 
                          sizeMemDataTrain + 
                          sizeMemStack + 
                          sizeMemStackIdx + 
                          sizeMemModel)


  memRest = gpG.free_mem-memRequired
  memUsePercent = memRequired/gpG.free_mem
  memUsePercent2 = memUsePercent - math.floor(memUsePercent)
  #print("memUsePercent: ", memUsePercent, "memUsePercent2: ", memUsePercent2, )

  memUsePercent = math.ceil(memUsePercent)
  if (memUsePercent2 > 0.85):
    memUsePercent = memUsePercent + 1
  
  if (memUsePercent <= 1) :
    memUsePercent = 1
        
  numIndividualsBlock = math.ceil(numIndividuals / memUsePercent)
  initialBlock = 0
  finalBlock = numIndividualsBlock 
 
  #print("finalBlock:", finalBlock, " numIndividuals:", numIndividuals)
  hData = np.reshape(hData, -1)
  dDataTrain = cuda.to_device(hData)

  # ******************  Individuals Evaluation  ********************
  # Invokes the GPU to interpret the initial population with data train
  # divide initial population in blocks to fit in available memory 

  hOutIndividuals = [] 
  hOutIndividualsBlock = []
  hStack = np.zeros((sizeMemStack), dtype=np.float32)
  hStackIdx = np.zeros((sizeMemStackIdx), dtype=np.float32)
  hStackModel = []
  if (getStackModel == 1):
    hStackModel = np.zeros((sizeMemModel), dtype=np.float32)
  
  dOutIndividualsBlock = 0
  pBlock1 = 0
  pBlocki_ant = 0
  pBlocks_ant = 0

  start_time = time.time()
  Ops = 0

  #elapsed1 = time.time() - start_time
  #gpG.WriteCSV_OpS("compute_individuals 1 ", elapsed1,Ops)

  while(finalBlock <= numIndividuals) :      
    sizeMemPopulationBlock = numIndividualsBlock * GenesIndividuals
    sizeMemIndividualsBlock = numIndividualsBlock * nrowTrain
    memStackBlock = sizeMemPopulationBlock * nrowTrain
    memStackIdxBlock = sizeMemIndividualsBlock
    sizeMemModelBlock = numIndividualsBlock * GenesIndividuals * nrowTrain
    totalSemanticElementsBlock = numIndividualsBlock * nrowTrain
    sizeMemIndividualsBlock = numIndividualsBlock * nrowTrain

    hStackBlock = np.zeros((memStackBlock), dtype=np.float32)
    hStackIdxBlock = np.zeros((memStackIdxBlock), dtype=np.float32)
    hStackModelBlock = np.zeros((sizeMemModelBlock), dtype=np.float32)
    hOutIndividualsBlock = np.zeros((sizeMemIndividualsBlock), dtype=np.float32)   

    # Get initial population block for evaluate individuals
    if (finalBlock ==  numIndividuals and pBlock1 == 0):
      hInitialPopulationBlock = hInitialPopulation
      #print("Entro A")
    else:
      hInitialPopulationBlock = hInitialPopulation[(initialBlock*GenesIndividuals):(finalBlock*GenesIndividuals)]
      #print("Entro B")
       
    dInitialPopulationBlock = cuda.to_device(hInitialPopulationBlock)
    dOutIndividualsBlock = cuda.to_device(hOutIndividualsBlock)   
    dStackBlock = cuda.to_device(hStackBlock)
    dStackIdxBlock = cuda.to_device(hStackIdxBlock)
    dStackModelBlock = cuda.to_device(hStackModelBlock)
        
    MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"]    

    #elapsed2 = time.time() - start_time
    #gpG.WriteCSV_OpS("compute_individuals 2 ", elapsed2,Ops)
    
    gpCuda.compute_individuals[blocksize, gridsize](
                        dInitialPopulationBlock,
                        dOutIndividualsBlock,
                        dDataTrain,
                        numIndividualsBlock,
                        GenesIndividuals,
                        nrowTrain,
                        nvar,
                        dStackBlock,
                        dStackIdxBlock,
                        getStackModel,
                        dStackModelBlock
    )
    #elapsed3 = time.time() - start_time
    #gpG.WriteCSV_OpS("compute_individuals 3 ", elapsed3,Ops)

    #cuda.synchronize()
    
    #elapsed4 = time.time() - start_time
    #gpG.WriteCSV_OpS("compute_individuals 4 ", elapsed4,Ops)

    # Return blocks from Device to host
    hOutIndividualsBlock = dOutIndividualsBlock.copy_to_host()
    hStackBlock = dStackBlock.copy_to_host()
    hStackIdxBlock = dStackIdxBlock.copy_to_host()

    #elapsed5 = time.time() - start_time
    #gpG.WriteCSV_OpS("compute_individuals 5 ", elapsed5,Ops)

    if (finalBlock >= numIndividuals and pBlock1 == 0) :
      hOutIndividuals = hOutIndividualsBlock
      hStackIdx = hStackIdxBlock
      hStack = hStackBlock
      #print("Entro 1")
    else :
      # Join device blocks with in one local block 
      hOutIndividuals = np.hstack((hOutIndividuals, hOutIndividualsBlock))
      pBlocki = pBlocki_ant + hStackIdxBlock.shape[0]
      hStackIdx[pBlocki_ant:pBlocki] = hStackIdxBlock   
      pBlocki_ant = pBlocki

      pBlocks = pBlocks_ant + hStackBlock.shape[0]
      hStack[pBlocks_ant:pBlocks] = hStackBlock
      pBlocks_ant = pBlocks
      #print("Entro 2")

    pBlock1 = pBlock1 + 1

    #elapsed6 = time.time() - start_time
    #gpG.WriteCSV_OpS("compute_individuals 6 ", elapsed6,Ops)
        
    if (finalBlock >= numIndividuals) :
      break

    initialBlock = finalBlock 
    finalBlock = initialBlock + numIndividualsBlock
    if (finalBlock > numIndividuals) :
      numIndividualsBlock = numIndividuals - initialBlock
      finalBlock = numIndividuals
  # End while

  elapsed = time.time() - start_time
  Ops = (numIndividuals  * nrowTrain * GenesIndividuals)
  gpG.WriteCSV_OpS("compute_individuals (" + str(pBlock1) + ")", elapsed,Ops)

  del hStackModelBlock
  #Free local memory 
  del hStackBlock
  del hStackIdxBlock
  del hInitialPopulationBlock
  del hOutIndividualsBlock 
  
  #Free gpu vectors memory 
  del dStackBlock
  del dStackIdxBlock
  del dStackModelBlock
  del dOutIndividualsBlock
  del dInitialPopulationBlock
  gc.collect()

  return hOutIndividuals, hStack, hStackIdx, hStackModel
# *************************  End of Compute Individuals  **************************

# ****************************  Evaluate Individuals  *****************************
def ComputeError(self,
                hOutIndividuals, 
                hDataY, 
                numIndividuals, 
                nrowTrain,
                hStack, 
                hStackIdx,
                evaluationMethod) :
   
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []

    result_train_p = 0
    hFit = np.zeros((gpG.sizeMemIndividuals), dtype=np.float32)
    dFit = cuda.to_device(hFit)

    dOutIndividuals = cuda.to_device(hOutIndividuals)
    dDataY = cuda.to_device(hDataY)

    #gridsize = gpCuda.gpuMaxUseProc(self.Individuals, blocksize)
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"] 

    start_time = time.time()

    if evaluationMethod == 0 :  #0=RMSE
        #print("RMSE")
        gpCuda.computeRMSE[blocksize, gridsize](
                        dOutIndividuals, 
                        dDataY, 
                        dFit, 
                        numIndividuals, 
                        nrowTrain ) 

        hFit = dFit.copy_to_host()
        # This section makes use of the isamin of cublas function to determine
        # the position of the best individual in initial Population using RMSE
        result_off = gpG.np.argmin(dFit)        
        indexBestOffspring = result_off

        result_w = gpG.np.argmax(dFit)	
        indexWorstOffspring = result_w       

    elif evaluationMethod == 1 :  #0=R2 :
        gpCuda.computeR2[blocksize, gridsize](
                        dOutIndividuals, 
                        dDataY, 
                        dFit, 
                        numIndividuals, 
                        nrowTrain )       

        hFit = dFit.copy_to_host()

        # This section makes use of the isamax of cublas function to determine
        # the position of the best individual in initial Population using R2
        # #make a handle to the function of tf.cublas 
        result_off = gpG.np.argmax(dFit)        
        indexBestOffspring = result_off

        result_w = gpG.np.argmin(dFit)	
        indexWorstOffspring = result_w      

    elif (evaluationMethod == 2 or #M4GP - 2=cuML LinearRegression
        evaluationMethod == 3 or #M4GP - 3=cuML Lasso regularization
        evaluationMethod == 4 or #M4GP - 4=cuML Ridge regularization
        evaluationMethod == 5 or #M4GP - 5=cuML kernel Ridge Regression
        evaluationMethod == 6 or #M4GP - 6=cuML Elasticnet regularization 
        evaluationMethod == 7 or #M4GP - 7=cuML MiniBatch none regularization
        evaluationMethod == 8 or #M4GP - 8=cuML MiniBatch lasso regularization
        evaluationMethod == 9 or #M4GP - 9=cuML MiniBatch ridge regularization
        evaluationMethod == 10) : #M4GP - 10=cuML MiniBatch elasticnet regularization

        #start_time = time.time()
        coefArr = []
        intercepArr = []
        cuModel = []

        hFit, cuModel, coefArr, intercepArr = gpCuM.EvaluateCuml2(self, hStack, hStackIdx, hFit, hDataY)

        #elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        #print(f"Time cuML lapsed: {elapsed}")
  
        dFit = cuda.to_device(hFit)
        if (self.scorer==0) or (self.scorer==1):
          result_off = gpG.np.argmin(dFit)        
          indexBestOffspring = result_off

          result_w = gpG.np.argmax(dFit)	
          indexWorstOffspring = result_w   
        elif  (self.scorer==2) :
          result_off = gpG.np.argmax(dFit)        
          indexBestOffspring = result_off

          result_w = gpG.np.argmin(dFit)	
          indexWorstOffspring = result_w       

        # Obtenemos los coeficientes y el modelo del 
        # mejor individuo generados por cuML
        coefArr_p = coefArr[indexBestOffspring]
        intercepArr_p = intercepArr[indexBestOffspring]  
        cuModel_p = cuModel[indexBestOffspring]
     
    # end if (Evaluation methods)

    elapsed = time.time() - start_time
    Ops = (numIndividuals * nrowTrain)
    gpG.WriteCSV_OpS("compute_error", elapsed,Ops)    
 
    return hFit, indexBestOffspring,  indexWorstOffspring, coefArr_p, intercepArr_p, cuModel_p
# *************************  End of Evaluate Individuals  **************************

# *******************************  Select Tournament  ******************************
def select_tournament(
                    hInitialPopulation,
                    hFit,
                    numIndividuals,
                    GenesIndividuals ) :
    
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"]

    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    # Initialize a state for each thread
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)
       

    hNewPopulation  = np.zeros((gpG.sizeMemPopulation), dtype=np.float32) 
    hBestParentsTournament = np.zeros((gpG.sizeMemIndividuals), dtype=np.int)

    dBestParentsTournament = cuda.to_device(hBestParentsTournament)
    dInitialPopulation = cuda.to_device(hInitialPopulation)
    dNewPopulation = cuda.to_device(hNewPopulation)

    dFit = cuda.to_device(hFit)

    start_time = time.time()
    gpCuda.parent_select_tournament[blocksize, gridsize](cu_states,
                              dNewPopulation,
                              dInitialPopulation,
                              dFit,
                              dBestParentsTournament,
                              gpG.sizeTournament,
                              numIndividuals,
                              GenesIndividuals   )

    elapsed = time.time() - start_time
    Ops = (numIndividuals * gpG.sizeTournament) 
    gpG.WriteCSV_OpS("tournament("+str(gpG.sizeTournament)+")", elapsed,Ops)  

    hNewPopulation = dNewPopulation.copy_to_host()
    hBestParentsTournament = dBestParentsTournament.copy_to_host()   
   
    return hNewPopulation, hBestParentsTournament
# ****************************  End of Select Tournament  ****************************

# *********************************  UMAD Mutation  **********************************
def umadMutation(self,
                 hInitialPopulation,
                 hBestParentsTournament,
                 numIndividuals) :
       
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"]

    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    # Initialize a state for each thread
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)

    hNewPopulation  = np.zeros((gpG.sizeMemPopulation), dtype=np.float32) 
    dNewPopulation = cuda.to_device(hNewPopulation)
    dInitialPopulation = cuda.to_device(hInitialPopulation)
    dBestParentsTournament = cuda.to_device(hBestParentsTournament)

    start_time = time.time()
    gpCuda.umadMutation[blocksize, gridsize](cu_states,
                        dNewPopulation,
                        dInitialPopulation,
                        dBestParentsTournament,
                        numIndividuals,
                        self.GenesIndividuals,
                        self.nrowTrain,
                        self.nvar,
                        self.mutationProb,
                        self.mutationDeleteRateProb,
                        self.maxRandomConstant,
                        self.genOperatorProb,
                        self.genVariableProb,
                        self.genConstantProb,
                        self.genNoopProb,
                        self.useOpIF)

    elapsed = time.time() - start_time
    Ops = (numIndividuals * self.GenesIndividuals) 
    gpG.WriteCSV_OpS("umadMutation", elapsed,Ops) 

    hNewPopulation = dNewPopulation.copy_to_host()
  
    return hNewPopulation
# ****************************  End of UMAD Mutation  ******************************

# *****************************  Survival (Elitist)  *******************************
def Survival(self,
            indexBestIndividual_p,
            indexBestOffspring,
            indexWorstOffspring,
            hInitialPopulation,
            hNewPopulation,
            hFit,
            hFitNew,
            coefArr_p,
            intercepArr_p,
            cuModel_p,
            stackBestModel_p,
            coefArrNew,
            intercepArrNew,
            cuModelNew,
            stackBestModelNew) :
    
    idx_a1 = indexWorstOffspring * self.GenesIndividuals
    idx_b1 = indexWorstOffspring * self.GenesIndividuals + self.GenesIndividuals
    idx_a2 = indexBestIndividual_p * self.GenesIndividuals
    idx_b2 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals

    if (self.evaluationMethod == 0 or self.evaluationMethod >= 2) and  (self.scorer !=2) :
        
        # Checamos si la nueva generacion es mejor que la anterior
        if (hFit[indexBestIndividual_p] < hFitNew[indexBestOffspring]) :
          # La nueva generacion no fue mejor que la anterior
          # Copia el mejor individuo de la anterior generacion  (idx_a2:idx_b2)
          # al lugar del peor individuo de la nueva generacion (idx_a1:idx_b1)
          hNewPopulation[idx_a1:idx_b1] = hInitialPopulation[idx_a2:idx_b2]

          # Ahora el peor hijo es el mejor padre
          hFitNew[indexWorstOffspring] = hFit[indexBestIndividual_p]
          indexBestIndividual_p = indexWorstOffspring

        else :
          # La nueva generacion fue mejor que la anterior
          indexBestIndividual_p = indexBestOffspring

          # Si es un metodo decuML, copiamos los coeficientes del
          # mejor individuo de la nueva generacion como papa para
          # la siguiente generacion
          if (self.evaluationMethod >= 2) :
            coefArr_p = coefArrNew
            intercepArr_p = intercepArrNew 
            cuModel_p = cuModelNew
            stackBestModel_p = stackBestModelNew
    # End if
    elif (self.evaluationMethod == 1) or (self.scorer==2) :
        # Checamos si el mejor padre de la anterior poblacion es mejor que el mejor hijo en la nueva poblacion*/
        if (hFit[indexBestIndividual_p] > hFitNew[indexBestOffspring]) :
          # Pasa el mejor individuo de la anterior poblacion a la posicion del peor individuo dela nueva poblacion */
          hNewPopulation[idx_a1:idx_b1] = hInitialPopulation[idx_a2:idx_b2]
          # Ahora el peor hijo es el mejor padre
          hFitNew[indexWorstOffspring] = hFit[indexBestIndividual_p]
          indexBestIndividual_p = indexWorstOffspring
        else :
          indexBestIndividual_p = indexBestOffspring

          if (self.evaluationMethod >= 2) :
            coefArr_p = coefArrNew
            intercepArr_p = intercepArrNew 
            cuModel_p = cuModelNew
            stackBestModel_p = stackBestModelNew
                      
        # End if
    # End if
    return hNewPopulation, indexBestIndividual_p, coefArr_p, intercepArr_p, cuModel_p, stackBestModel_p
# ***************************  End of Survival (Elitist)  *****************************

     # ***********************    NEW REPLACE   ***********************
def replace(self,
              hInitialPopulation,
              hNewPopulation, 
              hFit,
              hFitNew) :
  
  dInitialPopulation = cuda.to_device(hInitialPopulation)
  dNewPopulation = cuda.to_device(hNewPopulation)
  dFit = cuda.to_device(hFit)
  dFitNew = cuda.to_device(hFitNew)  

  # Move new population to Initial population for individuals and Fits 
  MaxOcup = gpCuda.gpuMaxUseProc(self.Individuals)
  blocksize = MaxOcup["BlockSize"]
  gridsize = MaxOcup["GridSize"]
  gpCuda.replace[blocksize, gridsize](dInitialPopulation, 
          dNewPopulation, 
          dFit,
          dFitNew, 
          self.Individuals, 
          self.GenesIndividuals)

  # Copiamos valores del dispositivo GPU al  host (locales)
  hFit = dFit.copy_to_host()      
  hInitialPopulation = dInitialPopulation.copy_to_host()
       
  return hInitialPopulation, hFit
# *********************** END NEW REPLACE ***********************

def getStackBestModel(
        hModelPopulation,
        hData,
        numIndividuals,
        GenesIndividuals,
        nrowTrain,
        nvar) :

  numIndividuals = 1
  hData = np.reshape(hData, -1)
  #GenesIndiv = hInitialPopulation.shape[0] # self.GenesIndividuals

  # Calculate memory por size vectors
  sizeMemIndividuals = numIndividuals * nrowTrain 
  sizeMemPopulation = numIndividuals * GenesIndividuals
  memStack = sizeMemPopulation * nrowTrain
  memStackIdx = sizeMemIndividuals
  sizeMemModel = GenesIndividuals * numIndividuals * nrowTrain
  totalSemanticElements = numIndividuals * nrowTrain

  #local vector
  hStack = []
  hStackIdx = []
  hStackModel = []
  hStack = np.zeros((memStack), dtype=np.float32)
  hStackIdx = np.zeros((memStackIdx), dtype=np.float32)
  hStackModel = np.zeros((sizeMemModel), dtype=np.int32)
  hOutIndividuals = np.zeros((sizeMemIndividuals), dtype=np.float32) 

  # Copy vectors to gpu device
  dModelPopulation = cuda.to_device(hModelPopulation)
  dData = cuda.to_device(hData)
  dStack = cuda.to_device(hStack)
  dStackIdx = cuda.to_device(hStackIdx)
  dStackModel = cuda.to_device(hStackModel)
  dOutIndividuals = cuda.to_device(hOutIndividuals)    

  #print("hModelPopulation:", hModelPopulation)
  #print("hData:", hData)

  MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElements)
  blocksize = MaxOcup["BlockSize"]
  gridsize = MaxOcup["GridSize"]  
  gpCuda.compute_individuals[blocksize, gridsize](
                      dModelPopulation,
                      dOutIndividuals,
                      dData,
                      numIndividuals,
                      GenesIndividuals,
                      nrowTrain,
                      nvar,
                      dStack,
                      dStackIdx,
                      1,
                      dStackModel   
  )  

  #hOutIndividuals = dOutIndividuals.copy_to_host()
  #hStack = dStack.copy_to_host()
  #hStackIdx = dStackIdx.copy_to_host()
  hStackModel = dStackModel.copy_to_host()

  del hStack 
  del hStackIdx
  del hOutIndividuals 

  del dModelPopulation   
  del dData
  del dStack 
  del dStackIdx 
  del dStackModel 
  del dOutIndividuals
  gc.collect()

  return hStackModel