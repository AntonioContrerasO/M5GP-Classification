# *********************************************************************
# Name: m5gp.py
# Description: Modulo principal del sistema que implementa los
# metodos del ciclo evolutivo de GP, asi como la interface tipo SkLearn
# Se implementa la logica de ejecucion para funciones de numba y CuML
# *********************************************************************

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import os
import sys 
import math
import copy
import pandas as pd
import numpy as np
import time
import gc
import cupy as cp

from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32)

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(this_dir))
import m5gpGlobals as gpG
import m5gpGlobals as gpF
import m5gpCumlMethods as gpCuM
import m5gpMod1 as gp2


class m5gpRegressor(BaseEstimator):
  #method to initialize the class
  def __init__(self, 
            generations=50, 
            Individuals=500, 
            GenesIndividuals=1024, 
            mutationProb=0.15, 
            mutationDeleteRateProb=0.01,  
            evaluationMethod=0, 
            scorer=0,  
            sizeTournament=0.20, 
            maxRandomConstant=5, 
            genOperatorProb=0.54, 
            genVariableProb=0.35, 
            genConstantProb=0.10, 
            genNoopProb=0.001,  
            useOpIF=0,   
            log=1, 
            verbose=1, 
            logPath='log/',
            function_set = '' ):

    env = dict(os.environ)
    self.generations = generations
    self.Individuals=Individuals 
    self.GenesIndividuals=GenesIndividuals
    self.mutationProb=mutationProb 
    self.mutationDeleteRateProb=mutationDeleteRateProb  
    self.evaluationMethod=evaluationMethod
    self.scorer = scorer
    self.sizeTournament=sizeTournament 
    self.maxRandomConstant=maxRandomConstant 
    self.genOperatorProb=genOperatorProb 
    self.genVariableProb=genVariableProb 
    self.genConstantProb=genConstantProb 
    self.genNoopProb=genNoopProb  
    self.useOpIF=useOpIF
    self.nvar=0 
    self.nrowTrain=0 
    self.nrowTest=0
    self.log=log
    self.verbose=verbose
    self.logPath=logPath
    self.function_set=function_set
    self.model = ''
    self.m4gpModel = ''
    self.cuModel = 0
    self.bestIndividual = ''
    self.nodes = 0

    print("Initializing m5gp")

    # Usign pycuda, get GPU device memory information
    gpG.pycudasetup()
    gpG.gpu_memory = gpG.pycuda.mem_get_info()
    gpG.free_mem = gpG.gpu_memory[0]
    print("Initial memory info:")
    print("GPU Memory: ", gpG.gpu_memory)
    print("Free Memory: ", gpG.free_mem)
    gpG.pycuda_finish()

    fName = "M5GP_OpS.csv"
    if os.path.exists(fName):
        os.remove(fName)
    return

  #This method implement the evolution with M5GP  
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    # train data    
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train

    self.nrowTrain = len(data.index)
    self.nrowTest = len(data.index)
    self.nrowPredict = len(data.index)
    self.nvar = data.shape[1] - 1
 
    print("Executing Fit - Method(", self.evaluationMethod ,") - ", gpCuM.cuGetMethodName(self), " Scorer:", self.scorer)
    print("nRows:", self.nrowTrain, "nVars:", self.nvar)

    # Store the size in bytes for initial population
    gpG.sizeMemPopulation = self.Individuals * self.GenesIndividuals 
    gpG.sizeMemIndividuals = self.Individuals 
    gpG.sizeTournament = math.ceil(self.sizeTournament * self.Individuals)

    # Define vectors to work on device 
    self.model = np.zeros((self.GenesIndividuals ), dtype=np.float32) 

    #rint("Initialize Individual")
    # *************************** Initialize population ********************************* 
    hInitialPopulation = gp2.initialize_population(
                              self.Individuals,
                              self.nvar,
                              self.GenesIndividuals,
                              self.maxRandomConstant,
                              self.genOperatorProb,
                              self.genVariableProb,
                              self.genConstantProb,
                              self.genNoopProb,
                              self.useOpIF )
    # -- End of Initialize population --

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals = [] 
    hStack = []
    hStackIdx = []
    hStackModel = []
  
    #print ("Compute Individual")
    hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
            hInitialPopulation,
            X_train,
            self.Individuals,
            self.GenesIndividuals,
            self.nrowTrain,
            self.nvar,
            0 )
    # ****************** End of Compute Individuals **********************
    # Get the semantic matrix
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []  
    stackBestModel_p = []

    coefArrNew = []
    intercepArrNew = [] 
    cuModelNew = []
    stackBestModelNew = []

    hFit = np.zeros((gpG.sizeMemIndividuals), dtype=np.float32)
    hFitNew = np.zeros((gpG.sizeMemIndividuals), dtype=np.float32)
    indexBestOffspring = 0
    indexWorstOffspring = 0


    #print("Compute Error")
    # ***************************** Compute ERROR ***********************************
    hFit, indexBestOffspring, indexWorstOffspring, coefArr_p, intercepArr_p, cuModel_p = gp2.ComputeError(self,
                hOutIndividuals, 
                y_train, 
                self.Individuals, 
                self.nrowTrain,
                hStack, 
                hStackIdx,
                self.evaluationMethod)

    # Index of best individual of initialization
    indexBestIndividual_p = indexBestOffspring 

    del hStack
    del hStackIdx
    gc.collect()

    ajFit = 0
    if (self.evaluationMethod == 1) : #or (self.scorer == 2) :  
      ajFit = gpG.MAX_R2_NEG * (-1)
    trainFit = hFit[indexBestIndividual_p] - ajFit    
    print("Initial Index:", indexBestIndividual_p, " Initial Fit:", trainFit)

    # ***********************************************************************
    # ********************* GP Process Generation Cycle *********************
    # ***********************************************************************
    print ("Starting generational process")
    for generation in range(1,self.generations + 1):
      trainFit = 0
      testFit = 0
      coefArrNew = []
      intercepArrNew = [] 
      cuModelNew = []
      stackBestModelNew = []
      start_time = time.time()

      #print("Torneo")
      # *********************  Select Tournament  **********************
      hNewPopulation, hBestParentsTournament = gp2.select_tournament(
                    hInitialPopulation,
                    hFit,
                    self.Individuals, 
                    self.GenesIndividuals )

      #print("Mutacion")
      # *********************  UMAD Mutation  **********************
      hNewPopulation = gp2.umadMutation(self,
                                  hInitialPopulation,
                                  hBestParentsTournament,
                                  self.Individuals) 

      #print (hNewPopulation)
      # ***************************  Compute Individuals  ****************************
      hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
              hNewPopulation,
              X_train,
              self.Individuals,
              self.GenesIndividuals,
              self.nrowTrain,
              self.nvar,
              0 )
      
      # ***************************** Compute ERROR ***********************************
      hFitNew, indexBestOffspring, indexWorstOffspring, coefArrNew, intercepArrNew, cuModelNew = gp2.ComputeError(self,
              hOutIndividuals, 
              y_train, 
              self.Individuals, 
              self.nrowTrain,
              hStack, 
              hStackIdx,
              self.evaluationMethod)

      #print("hFit:", hFit[indexBestIndividual_p], " indexBestIndividual_p:", indexBestIndividual_p)
      #print("hFitNew:", hFitNew[indexBestOffspring], " indexBestOffspring:", indexBestOffspring)

      # *********************** NEW SURVIVAL (Elitist) ***********************
      hNewPopulation, indexBestIndividual_p, coefArr_p, intercepArr_p, cuModel_p, stackBestModel_p = gp2.Survival(self,
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
              stackBestModelNew)
      # *********************** END NEW SURVIVAL ***********************


      # ***********************    NEW REPLACE   ***********************
      hInitialPopulation, hFit = gp2.replace(self,
                      hInitialPopulation,
                      hNewPopulation, 
                      hFit,
                      hFitNew)
      # *********************** END NEW REPLACE ***********************
      #print (hInitialPopulation)
      
			# Validate Best Individual with Test file for generation
			#/*trainFit = checkFitness(config, handle, dataFile, dInitialPopulation, indexBestIndividual_p, 0);*/
      ajFit = 0
      if (self.evaluationMethod == 1) : #or (self.scorer == 2) : 
        ajFit = gpG.MAX_R2_NEG * (-1)

      bestFitGeneration = hFit[indexBestIndividual_p]
      trainFit = hFit[indexBestIndividual_p] - ajFit

      # Obtenemos la longitud del stack del mejor papa
      BestIndividualLength = gpG.bestIndividualInfo(self, hInitialPopulation,  indexBestIndividual_p)
     

      if self.verbose == 1 :
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Generation:",generation, " Best Index:",indexBestIndividual_p, " Length Indiv:",BestIndividualLength, " Train Fit:",trainFit, f" Time lapsed:{elapsed}")
			#end if

      del hStack
      del hStackIdx
      del hStackModel
      del hOutIndividuals 
      gc.collect()
      
      
      if hFitNew[indexBestIndividual_p] == 0 :
        break
    
    # ************* Fin de for (Ciclo Generacional) ****************

    # Obtenemos el mejor individuo
    idx_a1 = indexBestIndividual_p * self.GenesIndividuals
    idx_b1 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
    self.bestIndividual = hInitialPopulation[idx_a1:idx_b1]
    self.model = self.bestIndividual

    # Para caso de evaluaciones utilizando cuML se construye una 
    # expresion utilizando todas expresiones del stack que generan 
    # la matriz semantica 
    if (self.evaluationMethod >= 2 ) :
      self.cuModel = copy.deepcopy(cuModel_p)
      self.maxRandomConstant = gpG.MAX_CONSTANT

      #sacamos el mejor modelo del stack de expresiones 
      stackBestModel_p = gp2.getStackBestModel(
                  self.bestIndividual,
                  X_train,
                  self.Individuals,
                  self.GenesIndividuals,
                  self.nrowTrain,
                  self.nvar) 
    
      # Se construye una nueva pila con las todas expresiones 
      # generadas y almacenadas en el stack del mejor modelo
      allModelExpr = gpG.getModelExpr(self, stackBestModel_p)
      
      # De la cadena completa de expresiones obtenemos el numero  
      # de stacks de expresiones disponibles
      # 'X:Y:Z:<Expr1>', 'X:Y:Z:<Expr2>', .... , 'X:Y:Z:<ExprN'  
      # (0)X=Total de elementos, 
      # (1)Y=Numero de stacks, 
      # (2)Z=Elemento de este stack 
      tmpModelExpr = allModelExpr[0]
      tmp = tmpModelExpr.split(':')
      nStack = int(tmp[1])

      # Se crea una nueva pila para guardar las expresiones de cada 
      # elememento del stack
      nvoModel = [] 
      m4gpModel = gpG.m4gpModel(self, stackBestModel_p, 
                                coefArr_p, 
                                intercepArr_p) 
            

      #print("self.cuModel.coef_.shape:", self.cuModel.coef_.shape)
      #print("self.cuModel.coef_:", self.cuModel.coef_)

      # Reconstruccion del modelo.
      # Se agregan los coeficientes obtenidos del modelo de evaluacion cuML
      # Por cada expresion, tenemos un coefiente  
      
      for j in range(nStack):
        #Obtenemos las expresiones del stack
        tmp1 = m4gpModel.get()
        tmp2 = coefArr_p
        tmp3 = tmp2[nStack-j-1]
        if(math.isnan(tmp3) or math.isinf(tmp3)) :
          tmp3 = 0

        # Solo interesan expresiones cuyo coeficiente no sea cero
        if (tmp3 != 0) :
          #Se agrega el coeficiente al inicio de la expresion
          #Se le agrega un operador de multiplicacion (*)
          nvoModel.insert(0,float(-10003)) 
          nvoModel.insert(0,float(tmp3))
          nvoModel =  gpG.m4gpBuildExpr(tmp1, nvoModel)
          if (j >= 1):
            nvoModel.append(-10001)            
      #end for

      tmp3 = intercepArr_p
      if (math.isnan(float(tmp3)) or (math.isinf(float(tmp3)))) :
        tmp3 = 0

      # Solo interesan expresiones cuyo coeficiente no sea cero
      if (tmp3 != 0) :
      #insertamos el intercept a la expresion
        nvoModel.append(float(tmp3))
        nvoModel.append(-10001)  # Se agrega un operador de suma (+)
      
      nvoModel.append(-11111)
      self.m4gpModel = np.array(nvoModel)
      del nvoModel
    #end if

    #Free local memory
    del hFit
    del hFitNew
    del hInitialPopulation

    # Clear lists
    gpCuM.coefArr.clear()
    gpCuM.intercepArr.clear()
    gpCuM.cuModel.clear()
    gc.collect()
    
    print("Finished Fit.")
    return	 
  # Fin de def (fit)

  def predict(self, X_predict):
    print("Inicio predict: ", X_predict.shape)

    # Get number of data rows for predict
    self.nrowPredict = X_predict.shape[0]
    hDataPredict = np.reshape(X_predict, -1)

    numIndividuals = 1
    hModelPopulation = self.bestIndividual  
    GenesIndiv = hModelPopulation.shape[0] # self.GenesIndividuals

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
            hModelPopulation,
            hDataPredict,
            numIndividuals,
            GenesIndiv,
            self.nrowPredict,
            self.nvar,
            0 )

    y_pred=[]

    stackBestModel_p = gp2.getStackBestModel(
                hModelPopulation,
                X_predict,
                numIndividuals,
                GenesIndiv,
                self.nrowPredict,
                self.nvar) 
    #allModelExpr = gpG.getModelExpr(self, stackBestModel_p)   

    if (self.evaluationMethod < 2 ) :
      for i in range(self.nrowPredict):
        y_pred.append(hOutIndividuals[i])

      y_pred = np.array(y_pred)
    else :
      st = hStack.reshape(numIndividuals, self.nrowPredict * GenesIndiv)
      ind = st[0]
      ind2 = ind.reshape(self.nrowPredict, GenesIndiv)
      tt = int(hStackIdx[0])
      
      sX_train = ind2[:, :tt]     
      cX = cp.asarray(sX_train, dtype=cp.float64)
      y_predModel = self.cuModel.predict(cX)
      y_pred = cp.asnumpy(y_predModel)

      #Free local memory
      del st
      del ind
      del ind2
      del tt
      del sX_train
      del cX
      del y_predModel
    #End if

    #Free local objects memory
    del hStack
    del hStackIdx
    del hStackModel
    del hOutIndividuals 
    del hDataPredict
    del hModelPopulation
    gc.collect()

    return y_pred
  # Fin de def (predict)

  def best_individual(self):
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel

    allModelExpr = gpG.getModelExpr(self, model) 

    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]
    tmp = BestModelExpr.split(':')
    indivLenght = tmp[0]
    nStack = tmp[1]
    complexity = tmp[2] 
    modelExpr = tmp[3]

    return modelExpr  
  # Fin de def (best_individual) 

  def get_model(self):
    return self.best_individual()
  # Fin de def (get_model) 

  def get_n_nodes(self):
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel     

    allModelExpr = gpG.getModelExpr(self, model) 
    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]

    tmp = BestModelExpr.split(':')
    nStack = tmp[1]
    nodes = tmp[2] 

    return str(nodes)
  # Fin de def (get_n_nodes) 

  def complexity(self):
    return self.get_n_nodes()
  # Fin de def (complexity) 
   
  def meanSquaredError(self, cY, YPred) :
    npY = np.array(cY).astype('float32')

    npYPred = YPred
    mse = mean_squared_error(npY, npYPred, squared=False)
    return mse

  def rmse(self, cY, YPred) :
    mse = self.meanSquaredError(cY, YPred) 
    mse = math.sqrt(mse)
    return mse


class m5gpClassifier(BaseEstimator):
  """
    Modelo de programación genética combinado con un modelo de cuML.

    Parámetros
    ----------
    generations : int, opcional, default=2
      Número de generaciones (limitado por defecto).

    Individuals : int, opcional, default=16
      Número de individuos.

    GenesIndividuals : int, opcional, default=256
      Número de genes por individuo.

    mutationProb : float, opcional, default=0.1
      Probabilidad de tasa de mutación.

    mutationDeleteRateProb : float, opcional, default=0.01
      Probabilidad de tasa de eliminación de mutación.

    sizeTournament : float, opcional, default=0.15
      Tamaño del torneo.

    evaluationMethod : int, opcional, default=0
      Modelo de machine learning que se utiliza en combinación con la programación genética para la evaluación de error.
      Opciones:
        - 0: Logistic Regression
        - 1: Support Vector Classifier
        - 2: Random Forest Classifier
        - 3: K Neighbors Classifier

    scorer : int, opcional, default=2
    Métrica utilizada para evaluar el desempeño del modelo de machine learning en combinación con la programación genética.
    
    Opciones:
        - 0: Accuracy Score (accuracy de cuML)
        - 1: ROC AUC Score (cuROCAUC de cuML)
        - 2: F1 Score (f1_score de scikit-learn)
        - 3: Average Precision Score (average_precision_score de scikit-learn)

    averageMode : str, opcional, default='macro'
    Solo disponible para F1 Score & Average Precision Score

    Opciones:
        - 'micro': Calcula la puntuación F1 globalmente contando el total de verdaderos positivos, falsos negativos y falsos positivos.
        - 'macro': Calcula la puntuación F1 para cada clase y luego calcula la media sin ponderar de estas puntuaciones.
        - 'weighted': Calcula la puntuación F1 para cada clase y luego calcula la media ponderada de estas puntuaciones según el soporte de cada clase.
        - 'samples': Calcula la puntuación F1 para cada instancia y luego calcula la media de estas puntuaciones.
        

    maxRandomConstant : int, opcional, default=999
      Número de constantes (-maxRandomConstant a maxRandomConstant).

    genOperatorProb : float, opcional, default=0.50
      Probabilidad de generar operadores.

    genVariableProb : float, opcional, default=0.39
      Probabilidad de generar variables.

    genConstantProb : float, opcional, default=0.1
      Probabilidad de generar constantes.

    genNoopProb : float, opcional, default=0.01
      Probabilidad de generar operadores NOOP.

    useOpIF : int, opcional, default=0
      Establece si se utiliza el operador IF.

    verbose : int, opcional, default=1
      Muestra mensajes durante la ejecución.

    crossVal : bool, opcional, default=True
      Indica si se realiza validación cruzada.

    k : int, opcional, default=5
      Número de divisiones para la validación cruzada.
  """
  def __init__(self, 
            generations=50, 
            Individuals=500, 
            GenesIndividuals=1024, 
            mutationProb=0.15, 
            mutationDeleteRateProb=0.01,  
            evaluationMethod=0, 
            scorer=0,  
            sizeTournament=0.20, 
            maxRandomConstant=5, 
            genOperatorProb=0.54, 
            genVariableProb=0.35, 
            genConstantProb=0.10, 
            genNoopProb=0.001,  
            useOpIF=0,   
            log=1, 
            verbose=1, 
            logPath='log/',
            function_set = '',
            crossVal = True,
            k = 5 ,
            averageMode = "macro",
            params=None,
            **kwargs):

    env = dict(os.environ)
    self.generations = generations
    self.Individuals=Individuals 
    self.GenesIndividuals=GenesIndividuals
    self.mutationProb=mutationProb 
    self.mutationDeleteRateProb=mutationDeleteRateProb  
    self.evaluationMethod=evaluationMethod
    self.scorer = scorer
    self.sizeTournament=sizeTournament 
    self.maxRandomConstant=maxRandomConstant 
    self.genOperatorProb=genOperatorProb 
    self.genVariableProb=genVariableProb 
    self.genConstantProb=genConstantProb 
    self.genNoopProb=genNoopProb  
    self.useOpIF=useOpIF
    self.nvar=0 
    self.nrowTrain=0 
    self.nrowTest=0
    self.log=log
    self.verbose=verbose
    self.logPath=logPath
    self.function_set=function_set
    self.model = ''
    self.m4gpModel = ''
    self.cuModel = 0
    self.bestIndividual = ''
    self.crossVal = crossVal
    self.k = k
    self.averageMode = averageMode
    self.params = None
    # if params == None:
    #   self.params = gpCuM.validateParameters(kwargs, self.evaluationMethod)
    # else : 
    #   self.params = gpCuM.validateParameters(params, self.evaluationMethod)

    print(gpCuM.cuGetMethodNameClassification(self))
    

    print("Initializing m5gp")

    # Usign pycuda, get GPU device memory information
    gpG.pycudasetup()
    gpG.gpu_memory = gpG.pycuda.mem_get_info()
    gpG.free_mem = gpG.gpu_memory[0]
    print("Initial memory info:")
    print("GPU Memory: ", gpG.gpu_memory)
    print("Free Memory: ", gpG.free_mem)
    gpG.pycuda_finish()

    fName = "M5GP_OpS.csv"
    if os.path.exists(fName):
        os.remove(fName)
    return

  #This method implement the evolution with M5GP  
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    # train data    
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train

    self.nrowTrain = len(data.index)
    self.nrowTest = len(data.index)
    self.nrowPredict = len(data.index)
    self.nvar = data.shape[1] - 1
 
    print("Executing Fit - Method(", self.evaluationMethod ,") - ", gpCuM.cuGetMethodNameClassification(self), " Scorer:", self.scorer)
    print("nRows:", self.nrowTrain, "nVars:", self.nvar)

    # Store the size in bytes for initial population
    gpG.sizeMemPopulation = self.Individuals * self.GenesIndividuals 
    gpG.sizeMemIndividuals = self.Individuals 
    gpG.sizeTournament = math.ceil(self.sizeTournament * self.Individuals)

    # Define vectors to work on device 
    self.model = np.zeros((self.GenesIndividuals ), dtype=np.float32) 

    #rint("Initialize Individual")
    # *************************** Initialize population ********************************* 
    hInitialPopulation = gp2.initialize_population(
                              self.Individuals,
                              self.nvar,
                              self.GenesIndividuals,
                              self.maxRandomConstant,
                              self.genOperatorProb,
                              self.genVariableProb,
                              self.genConstantProb,
                              self.genNoopProb,
                              self.useOpIF )
    # -- End of Initialize population --
    
    # ***************************  Compute Individuals  ****************************
    hOutIndividuals = [] 
    hStack = []
    hStackIdx = []
    hStackModel = []
  
    #print ("Compute Individual")
    hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
            hInitialPopulation,
            X_train,
            self.Individuals,
            self.GenesIndividuals,
            self.nrowTrain,
            self.nvar,
            0 )
    # ****************** End of Compute Individuals **********************
    # Get the semantic matrix
    cuModel_p = []  
    stackBestModel_p = []

    cuModelNew = []
    stackBestModelNew = []

    hFit = np.zeros((gpG.sizeMemIndividuals), dtype=np.float32)
    hFitNew = np.zeros((gpG.sizeMemIndividuals), dtype=np.float32)
    indexBestOffspring = 0
    indexWorstOffspring = 0


    #print("Compute Error")
    # ***************************** Compute ERROR ***********************************
    hFit, indexBestOffspring, indexWorstOffspring, cuModel_p = gp2.ComputeErrorClassification(self,
                hOutIndividuals, 
                y_train, 
                self.Individuals, 
                self.nrowTrain,
                hStack, 
                hStackIdx,
                self.evaluationMethod)

    # Index of best individual of initialization
    indexBestIndividual_p = indexBestOffspring 

    del hStack
    del hStackIdx
    gc.collect()

    ajFit = 0
    trainFit = hFit[indexBestIndividual_p] - ajFit    
    print("Initial Index:", indexBestIndividual_p, " Initial Fit:", trainFit)

    # ***********************************************************************
    # ********************* GP Process Generation Cycle *********************
    # ***********************************************************************
    print ("Starting generational process")
    for generation in range(1,self.generations + 1):
      trainFit = 0
      testFit = 0
      cuModelNew = []
      stackBestModelNew = []
      start_time = time.time()

      #print("Torneo")
      # *********************  Select Tournament  **********************
      hNewPopulation, hBestParentsTournament = gp2.select_tournament(
                    hInitialPopulation,
                    hFit,
                    self.Individuals, 
                    self.GenesIndividuals )

      #print("Mutacion")
      # *********************  UMAD Mutation  **********************
      hNewPopulation = gp2.umadMutation(self,
                                  hInitialPopulation,
                                  hBestParentsTournament,
                                  self.Individuals) 

      #print (hNewPopulation)
      # ***************************  Compute Individuals  ****************************
      hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
              hNewPopulation,
              X_train,
              self.Individuals,
              self.GenesIndividuals,
              self.nrowTrain,
              self.nvar,
              0 )
      
      # ***************************** Compute ERROR ***********************************
      hFitNew, indexBestOffspring, indexWorstOffspring, cuModelNew = gp2.ComputeErrorClassification(self,
              hOutIndividuals, 
              y_train, 
              self.Individuals, 
              self.nrowTrain,
              hStack, 
              hStackIdx,
              self.evaluationMethod)

      #print("hFit:", hFit[indexBestIndividual_p], " indexBestIndividual_p:", indexBestIndividual_p)
      #print("hFitNew:", hFitNew[indexBestOffspring], " indexBestOffspring:", indexBestOffspring)

      # *********************** NEW SURVIVAL (Elitist) ***********************
      hNewPopulation, indexBestIndividual_p, cuModel_p, stackBestModel_p = gp2.SurvivalClassification(self,
              indexBestIndividual_p,
              indexBestOffspring,
              indexWorstOffspring,
              hInitialPopulation,
              hNewPopulation,
              hFit,
              hFitNew,
              cuModel_p,
              stackBestModel_p,
              cuModelNew,
              stackBestModelNew)
      # *********************** END NEW SURVIVAL ***********************


      # ***********************    NEW REPLACE   ***********************
      hInitialPopulation, hFit = gp2.replace(self,
                      hInitialPopulation,
                      hNewPopulation, 
                      hFit,
                      hFitNew)
      # *********************** END NEW REPLACE ***********************
      #print (hInitialPopulation)
      
			# Validate Best Individual with Test file for generation
			#/*trainFit = checkFitness(config, handle, dataFile, dInitialPopulation, indexBestIndividual_p, 0);*/
      ajFit = 0
      bestFitGeneration = hFit[indexBestIndividual_p]
      trainFit = hFit[indexBestIndividual_p] - ajFit

      # Obtenemos la longitud del stack del mejor papa
      BestIndividualLength = gpG.bestIndividualInfo(self, hInitialPopulation,  indexBestIndividual_p)
     

      if self.verbose == 1 :
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Generation:",generation, " Best Index:",indexBestIndividual_p, " Length Indiv:",BestIndividualLength, " Train Fit:",trainFit, f" Time lapsed:{elapsed}")
			#end if

      del hStack
      del hStackIdx
      del hStackModel
      del hOutIndividuals 
      gc.collect()
      
      
      if hFitNew[indexBestIndividual_p] == 0.99 :
        break
    
    # ************* Fin de for (Ciclo Generacional) ****************

    # Obtenemos el mejor individuo
    idx_a1 = indexBestIndividual_p * self.GenesIndividuals
    idx_b1 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
    self.bestIndividual = hInitialPopulation[idx_a1:idx_b1]
    self.model = self.bestIndividual

    # Para caso de evaluaciones utilizando cuML se construye una 
    # expresion utilizando todas expresiones del stack que generan 
    # la matriz semantica 
    
    self.cuModel = copy.deepcopy(cuModel_p)
    self.maxRandomConstant = gpG.MAX_CONSTANT

    #sacamos el mejor modelo del stack de expresiones 
    stackBestModel_p = gp2.getStackBestModel(
                self.bestIndividual,
                X_train,
                self.Individuals,
                self.GenesIndividuals,
                self.nrowTrain,
                self.nvar) 
  
    # Se construye una nueva pila con las todas expresiones 
    # generadas y almacenadas en el stack del mejor modelo
    allModelExpr = gpG.getModelExpr(self, stackBestModel_p)

    final_expression = []
    nodes = True
    for expression in allModelExpr:
      array = expression.split(":")
      if nodes:
        self.nodes = array[1]
        nodes = False

      final_expression.append(array[3])
    self.m4gpModel = final_expression

    #end if

    #Free local memory
    del hFit
    del hFitNew
    del hInitialPopulation

    # Clear lists
    gpCuM.coefArr.clear()
    gpCuM.intercepArr.clear()
    gpCuM.cuModel.clear()
    gc.collect()
    
    print("Finished Fit.")

    return	 
  # Fin de def (fit)

  def predict(self, X_predict, probability=False):
    print("Inicio predict: ", X_predict.shape)
    

    # Get number of data rows for predict
    self.nrowPredict = X_predict.shape[0]
    hDataPredict = np.reshape(X_predict, -1)
    numIndividuals = 1
    hModelPopulation = self.bestIndividual  
    GenesIndiv = hModelPopulation.shape[0] # self.GenesIndividuals

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals, hStack, hStackIdx, hStackModel = gp2.compute_individuals(
            hModelPopulation,
            hDataPredict,
            numIndividuals,
            GenesIndiv,
            self.nrowPredict,
            self.nvar,
            0 )

    y_pred = []

    # stackBestModel_p = gp2.getStackBestModel(
    #             hModelPopulation,
    #             X_predict,
    #             numIndividuals,
    #             GenesIndiv,
    #             self.nrowPredict,
    #             self.nvar) 
    # allModelExpr = gpG.getModelExpr(self, stackBestModel_p)   
    # print(allModelExpr)

  
    st = hStack.reshape(numIndividuals, self.nrowPredict * GenesIndiv)
    ind = st[0]
    ind2 = ind.reshape(self.nrowPredict, GenesIndiv)
    tt = int(hStackIdx[0])
    
    sX_train = ind2[:, :tt]     
    cX = cp.asarray(sX_train, dtype=cp.float64)
    if probability == True:
      y_predModel = self.cuModel.predict_proba(cX)
    else:
      y_predModel = self.cuModel.predict(cX)
    y_pred = cp.asnumpy(y_predModel)

    if X_predict.shape == (2,):
      y_pred = y_pred[0]

    #Free local memory
    del st
    del ind
    del ind2
    del tt
    del sX_train
    del cX
    del y_predModel
    #End if

    #Free local objects memory
    del hStack
    del hStackIdx
    del hStackModel
    del hOutIndividuals 
    del hDataPredict
    del hModelPopulation
    gc.collect()

    return y_pred
  # Fin de def (predict)

  def predict_proba(self, X_predict):
    return self.predict(X_predict,True)

  def best_individual_expression(self):
    return  self.m4gpModel  
  # Fin de def (best_individual) 

  def complexity(self):
    return self.nodes
  # Fin de def (complexity) 
   
  def score(self, X, y, metric=0, averageMode="macro") :

    if metric == 3:
      y_pred = self.predict_proba(X)
    else: 
      y_pred = self.predict(X)

    score = gpCuM.evaluationMetrics(metric, y, y_pred, averageMode)

    return score