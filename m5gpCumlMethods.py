# *********************************************************************
# Name: m5gpCumlMethods.py
# Description: Modulo que implementa las metodos para ejecutar multiples
# metodos de CuML
# Se utiliza la libreria cuml
# *********************************************************************


import math
import copy
import cupy as cp
import cudf
import gc
import numpy as np

import cuml as cu
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml import Ridge
from cuml.linear_model import Ridge
from cuml.linear_model import Lasso
from cuml.kernel_ridge import KernelRidge
from cuml.linear_model import ElasticNet

from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml.metrics.regression import mean_squared_error as cuMSE
from cuml.metrics.regression import r2_score as cuR2

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import cpu_count
from multiprocessing import Manager

#import skcuda.cublas as cublas
#import pycuda.gpuarray as gpuarray

import m5gpGlobals as gpG

coefArr =[]
intercepArr =[]
cuModel = []
slr = 0

cuMethodName = ""

def check_npzeros(arr):
    if np.all(arr == 0):
        return True
    return False

def cuGetMethodName(self) :
    global cuMethod

    if self.evaluationMethod == 0 :
        cuMethod = "m5gp RMSE"

    if self.evaluationMethod == 1 :
        cuMethod = "m5gp R2"

    if self.evaluationMethod == 2 :
        cuMethod = "cuML Linear Regression"

    if self.evaluationMethod == 3 :
        cuMethod = "cuML Lasso Regression"

    if self.evaluationMethod == 4 :
        cuMethod = "cuML Ridge Regression"

    if self.evaluationMethod == 5 :
        cuMethod = "cuML kernel Ridge Regression"

    if self.evaluationMethod == 6 :
        cuMethod = "cuML Elasticnet Regression"
 
    if self.evaluationMethod == 7 :
        cuMethod = "cuML MiniBatch Normal Regression"

    if self.evaluationMethod == 8 :
        cuMethod = "cuML MiniBatch Lasso Regression"

    if self.evaluationMethod == 9 :
        cuMethod = "cuML MiniBatch Ridge Regression"
 
    if self.evaluationMethod == 10 :
        cuMethod = "cuML MiniBatch Elasticnet Regression"

    return cuMethod

def createCumlMethod(mFitness) :
    rPenalty = 'none'
    if mFitness == 2 :
        slr = LinearRegression(fit_intercept = True, 
                               normalize = False, 
                               algorithm = "svd" # algorithm{‘svd’, ‘eig’, ‘qr’, ‘svd-qr’, ‘svd-jacobi’}, (default = ‘eig’)
                               ) 

    if mFitness == 3 :
        slr = Lasso(alpha = 1.0, # (default = 1.0)
                    normalize = True, # (default = False)
                    fit_intercept=True, # (default = True)
                    max_iter = 1000, #  (default = 1000)
                    solver =  'cd', # {‘cd’, ‘qn’} (default=’cd’)
                    selection = 'cyclic' # {‘cyclic’, ‘random’} (default=’cyclic’)
                    )

    if mFitness == 4 :
        slr = Ridge(alpha=1.0, # (default = 1.0)
                    fit_intercept=True, # (default = True)
                    normalize=True, # (default = False)
                    solver="svd", #solver {‘eig’, ‘svd’, ‘cd’} (default = ‘eig’)
                    verbose=6)

    if mFitness == 5 :
        slr = KernelRidge(kernel="linear")

    if mFitness == 6 :
        slr = ElasticNet(alpha = 1.0,  # (default = 1.0)
                         l1_ratio=0.5,  # (default = 0.5)
                         solver='cd', # {‘cd’, ‘qn’} (default=’cd’)
                         normalize=False, #  (default = False)
                         max_iter = 1000, #  (default = 1000)
                         tol=0.001, # (default = 1e-3)
                         fit_intercept=True, # (default = True)
                         selection= 'random' # {‘cyclic’, ‘random’} (default=’cyclic’)
                         )

    if mFitness == 7 :
        rPenalty = 'none' # normal - Linear regression

    if mFitness == 8 :
        rPenalty = 'l1'  # Lasso

    if mFitness == 9 :
        rPenalty = 'l2' #Ridge

    if mFitness == 10 :
        rPenalty = 'elasticnet'

    if (mFitness >= 7 and mFitness <= 10) :
        slr = cumlMBSGDRegressor(alpha=0.0001, # default = 0.0001)
                                learning_rate='adaptive', #learning_rate : {‘optimal’, ‘constant’, ‘invscaling’, ‘adaptive’} (default = ‘constant’)
                                eta0=0.001, # (default = 0.001)
                                epochs=1000, # (default = 1000)
                                fit_intercept=True, # (default = True)
                                l1_ratio = 0.15, # (default=0.15)
                                batch_size=1024, # (default = 32)
                                tol=0.001, # (default = 1e-3)
                                penalty=rPenalty, # {‘none’, ‘l1’, ‘l2’, ‘elasticnet’} (default = ‘l2’)
                                loss='squared_loss', # {‘hinge’, ‘log’, ‘squared_loss’} (default = ‘hinge’)
                                power_t=0.5, # (default = 0.5)
                                output_type = 'numpy', #output_type : {‘input’, ‘array’, ‘dataframe’, ‘series’, ‘df_obj’, ‘numba’, ‘cupy’, ‘numpy’, ‘cudf’, ‘pandas’}, default=None
                                verbose = True)
    #end if
    return slr


#Ejecuta la evaluacion del modelo utilizando CUML    
def ExecCuml(nProc, hFit,  st, mFitness, indiv, genes, nrows, hStackIdx, y_train, scorer):

    if (nProc > indiv):
        return
    
    slr = createCumlMethod(mFitness)
    #slr = copy.deepcopy(slr1)

    ind = st[nProc]

    #como los datos vienen como vector, lo convertimos como matriz de cupy
    ind2 = ind.reshape(nrows, genes)

    #Obtenemos el numero de columnas (elementos del stack)
    tt = int(hStackIdx[nProc*nrows])

    # Transformamos como matriz el vector del individuo obtenido del stack 
    sX_train = ind2[:, :tt]      

    sCols = sX_train.shape[1]
    cX = cudf.DataFrame()
    cY = cudf.DataFrame()

    # Verificamos que al menos tengamos una columna en el arreglo
    if (sCols >= 1) :
        cX = cp.asarray(sX_train, dtype=cp.float64)
        cY = cp.asarray(y_train, dtype=cp.float64)

        # Procesamos el Fit con el arreglo transformado
        reg = slr.fit(cX, cY)
   
        # Creamos un vector de coeficientes
        coefArr = reg.coef_
        
        #creamos un vector de valores de interceps
        if(math.isnan(reg.intercept_) or math.isinf(reg.intercept_)) :
            intercepArr = 0
        else :
            intercepArr = reg.intercept_

        yPred = slr.predict(cX)

        cuModel= copy.deepcopy(slr)

        if check_npzeros(yPred):
            if (scorer==0):
                mse = gpG.MAX_RMSE
            else :
                mse = gpG.MAX_R2_NEG
        else :
            if (scorer==0) or (scorer==1):
                # Se hace la evaluacion utilizando MSE
                mse = cuMSE(cY, yPred, squared=True)
            else :
                # Se hace la evaluacion utilizando R2
                mse = cuR2(cY, yPred)
    else :      
        if (scorer==0) or (scorer==1):
            mse = gpG.MAX_RMSE
        else :
            mse = gpG.MAX_R2_NEG
        coefArr = 0
        intercepArr = 0
        cuModel = copy.deepcopy(slr)
    #endif

    if math.isnan(mse) or math.isinf(mse):
        if (scorer==0) or (scorer==1):
            mse = gpG.MAX_RMSE
        else :
            mse = gpG.MAX_R2_NEG
    hFit[nProc] = mse
    
    cX = []
    cY = []
    sX_train = []
    return cuModel

# Ejecuta la evaluacion del modelo utilizando multiprocesamiento (CPU Cores)
def EvaluateCuml(self, hStack, hStackIdx, hFit, y_train) :
    global coefArr
    global intercepArr
    global cuMethod
    global cuModel
    global slr

    coefArr = []
    intercepArr = []
    cuModel = []
                    
    #Obtenemos todos los stack con todos los resultados de la matriz semantica
    st = hStack.reshape(self.Individuals, self.nrowTrain * self.GenesIndividuals)

    #slr = createCumlMethod(self.evaluationMethod)
    #slr = 0

    nCores = cpu_count()
    n_processes = int(nCores/3) 
    n_processes = 4
    #n_processes = 1
    set_start_method('spawn', force=True)

    manager = Manager()
    hFit_L = manager.list(hFit)
    st_L = manager.list(st)
    hStackIdx_L  = manager.list(hStackIdx)
    y_train_L  = manager.list(y_train)
    
    #Ejecuta la evaluacion de CUML utilizando nucleos de multiprocesamiento
    print("Inicio cuML multiprocess nCores:", nCores, "n_processes:", n_processes)
    with Pool(processes=n_processes) as pool:
            results = [pool.apply_async(ExecCuml, args=(nProc, hFit_L, st_L, self.evaluationMethod, self.Individuals, self.GenesIndividuals, self.nrowTrain, hStackIdx_L, y_train_L, self.scorer)) for nProc in range(self.Individuals)]
            try:
                hRes = [res.get(timeout=1000) for res in results]
                hFit_tmp = list(hFit_L)
            except :
                print("Timeout Multiprocessing")
                hFit_tmp = hFit.fill(gpG.MAX_RMSE)

    print("Termino execCores")
    
    for i in range(self.Individuals):
        # Regresa el Fit obtenido
        hFit[i] = hFit_tmp[i]

        # Regresa el modelo de CUML generado por cada individuo (hRes)
        slr2 = copy.deepcopy(hRes[i])
        cuModel.insert(i,slr2)
        intercepArr.insert(i,slr2.intercept_)
        coefArr.insert(i,slr2.coef_)

    return hFit, cuModel, coefArr, intercepArr

# Ejecuta la evaluacion del modelo de manera secuencial    
def EvaluateCuml2(self, hStack, hStackIdx, hFit, y_train) :
    global coefArr
    global intercepArr
    global cuMethod
    global cuModel
    global slr

    coefArr = []
    intercepArr = []
    cuModel = []
                    
    #Obtenemos todos los stack con todos los resultados de la matriz semantica
    st = hStack.reshape(self.Individuals, self.nrowTrain * self.GenesIndividuals)

    #Ejecuta la evaluacion de CUML de manera secuencial
    for i in range(self.Individuals):
        hRes = ExecCuml(i, hFit, st, self.evaluationMethod, self.Individuals, self.GenesIndividuals, self.nrowTrain, hStackIdx, y_train, self.scorer)
        
        # Regresa el modelo de CUML del individuo generado (hRes)
        slr2 = copy.deepcopy(hRes)

        # Agregamos el modelo CUML del individuo en un arreglo
        cuModel.insert(i,slr2)
        intercepArr.insert(i,slr2.intercept_)
        coefArr.insert(i,slr2.coef_)

    return hFit, cuModel, coefArr, intercepArr

