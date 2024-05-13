# *********************************************************************
# Name: m5gpCumlMethods.py
# Description: Modulo que implementa las metodos para ejecutar multiples
# metodos de CuML
# Se utiliza la libreria cuml.
# *********************************************************************
 
import math
import copy
import cupy as cp
import cudf
import gc
import numpy as np
import pandas as pd


# Import models from CUML
import cuml as cu
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml import Ridge
from cuml.linear_model import Ridge
from cuml.linear_model import Lasso
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
from cuml.kernel_ridge import KernelRidge
from cuml.linear_model import ElasticNet
from cuml import LogisticRegression
from cuml import SVC
from cuml import RandomForestClassifier
from cuml import KNeighborsClassifier
from cuml import MBSGDClassifier


# import metrics
from cuml.metrics.regression import mean_squared_error as cuMSE
from cuml.metrics.regression import r2_score as cuR2
from cuml.metrics.accuracy import accuracy_score 
from cuml.metrics import roc_auc_score 
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from cuml.model_selection import train_test_split

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import cpu_count
from multiprocessing import Manager

import logging

#import skcuda.cublas as cublas
#import pycuda.gpuarray as gpuarray

import m5gpGlobals as gpG

coefArr =[]
intercepArr =[]
cuModel = []
slr = 0

cuMethodName = ""

average = ""

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
                               copy_X = True,
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


def cuGetMethodNameClassification(self):
        global cuMethod

        if self.evaluationMethod == 0:
            cuMethod = "Logistic Regression"
        elif self.evaluationMethod == 1:
            cuMethod = "Support Vector Classifier"
        elif self.evaluationMethod == 2:
            cuMethod = "Random Forest Classifier"
        elif self.evaluationMethod == 3:
            cuMethod = "K Neighbors Classifier"
        elif self.evaluationMethod == 4:
            cuMethod = "Mini Batch Classifier"

        return cuMethod

def getDefaultParams(evaluationMethod):

    if evaluationMethod == 0:
            defaultParams = {
                "penalty": "l2",  # Default value
                "tol": 1e-4,  # Default value
                "C": 1.0,  # Default value
                "fit_intercept": True,  # Default value
                "class_weight": None,  # Default value
                "max_iter": 20000,  # Default value
                "linesearch_max_iter": 50,  # Default value
                "verbose": False,  # Default value
                "l1_ratio": None,  # Default value
                "solver": "qn",  # Default value
                "output_type": None
            }

    elif evaluationMethod == 1:
        defaultParams = {
                "C": 92.5,  # Default value
                "kernel": "rbf",  # Default value
                "degree": 2,  # Default value
                "gamma": "auto",  # Default value
                "coef0": 5.9,  # Default value
                "tol": 1e-3,  # Default value
                "cache_size": 1024.0,  # Default value
                "max_iter": -1,  # Default value
                "nochange_steps": 1000,  # Default value
                "verbose": False,  # Default value
                "output_type": None,  # Default value
                "class_weight" : 'balanced'
            }

    elif evaluationMethod == 2:
        defaultParams = {
                "n_estimators": 100,  # Default value
                "split_criterion": 0,  # Default value
                "bootstrap": True,  # Default value
                "max_samples": 1.0,  # Default value
                "max_depth": 16,  # Default value
                "max_leaves": -1,  # Default value
                "max_features": "auto",  # Default value
                "n_bins": 128,  # Default value
                "n_streams": 4,  # Default value
                "min_samples_leaf": 1,  # Default value
                "min_samples_split": 2,  # Default value
                "min_impurity_decrease": 0.0,  # Default value
                "max_batch_size": 4096,  # Default value
                "random_state": None,  # Default value
                "verbose": False,  # Default value
                "output_type": None  # Default value
            }

    elif evaluationMethod == 3:
        defaultParams = {
                "n_neighbors": 5,  # Default value
                "algorithm": "auto",  # Default value
                "metric": "euclidean",  # Default value
                "weights": "uniform",  # Default value
                "verbose": False,  # Default value
                "output_type": None  # Default value
            }
    elif evaluationMethod == 4:
        defaultParams = {
                "loss": 'hinge',
                "penalty": "l2",
                "alpha": 0.0001,
                "l1_ratio": 0.15,
                "batch_size": 32,
                "fit_intercept": True,
                "epochs": 1000,
                "tol": 1e-3,
                "shuffle": True,
                "eta0": 0.001,
                "power_t": 0.5,
                "learning_rate": 'constant',
                "n_iter_no_change": 5,
                "verbose": False,
                "output_type": None
            }
    
    return defaultParams


def validateParameters(params,evaluationMethod):
    defaultParams = getDefaultParams(evaluationMethod)

    keys_dict1 = set(params.keys())
    keys_dict2 = set(defaultParams.keys())

    if (keys_dict1.issubset(keys_dict2)):

        for key, value in params.items():
            defaultParams[key] = value
    else: 
        print("Check your params, One or more are incorrect")
        exit()
    
    return defaultParams


def createCumlMethodClassification(evaluationMethod, params=None):
    if params== None:
            params = getDefaultParams(evaluationMethod)
    if evaluationMethod == 0 :
        slr = LogisticRegression(**params)
    if evaluationMethod == 1:
        slr = SVC(**params, probability= True)
    if evaluationMethod == 2:
        slr = RandomForestClassifier(**params)
    if evaluationMethod == 3:
        slr = KNeighborsClassifier(**params)
    if evaluationMethod == 4: # es muy lento
        slr = MBSGDClassifier()
    return slr

def ExecCumlClassification(nProc, hFit,  st, evaluationMethod, indiv, genes, nrows, hStackIdx, y_train, scorer, params , CrossVal, k, averageMode):

    fit = 0
    if (nProc > indiv):
        
        return
    # Desactivar los mensajes de registro
    logging.getLogger('cuml').setLevel(logging.ERROR)
    
    slr = createCumlMethodClassification(evaluationMethod, params)
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

    if (sCols >= 1) :
        cX = cp.asarray(sX_train, dtype=cp.float64)
        cY = cp.asarray(y_train, dtype=cp.float64)

        if CrossVal:
            fit, slr =  CrossValidation(slr, cX, cY, scorer, k, averageMode)
        else:
            # Procesamos el Fit con el arreglo transformado
            reg = slr.fit(cX, cY)

            yPred = make_predictions(slr,scorer,cX.to_numpy())

            cuModel= copy.deepcopy(slr)

            fit = evaluationMetrics(scorer,cY.to_numpy(),yPred,averageMode)

        cuModel = copy.deepcopy(slr)
    else :      
        fit = 0
        cuModel = copy.deepcopy(slr)
    #endif

    if math.isnan(fit) or math.isinf(fit):
        fit = 0
     
    
    hFit[nProc] = fit
    
    cX = []
    cY = []
    sX_train = []
    return cuModel

def EvaluateCuml2Classification(self, hStack, hStackIdx, hFit, y_train) :
    global cuMethod
    global cuModel
    global slr

    cuModel = []
                    
    #Obtenemos todos los stack con todos los resultados de la matriz semantica
    st = hStack.reshape(self.Individuals, self.nrowTrain * self.GenesIndividuals)

    #Ejecuta la evaluacion de CUML de manera secuencial
    for i in range(self.Individuals):
        hRes = ExecCumlClassification(i, hFit, st, self.evaluationMethod, self.Individuals, self.GenesIndividuals, self.nrowTrain, hStackIdx, y_train, self.scorer, self.params, self.crossVal, self.k, self.averageMode)
        
        # Regresa el modelo de CUML del individuo generado (hRes)
        slr2 = copy.deepcopy(hRes)

        # Agregamos el modelo CUML del individuo en un arreglo
        cuModel.insert(i,slr2)

    return hFit, cuModel


def evaluationMetrics(scorer ,y_true, y_pred, averageMode):
    classes = pd.DataFrame(y_true)[0].unique().size
    average = 'binary'
    
    if(classes > 2 ):
        if (averageMode == ["micro", "macro", "weighted", "samples"]):
            average = averageMode
        else:
            average = "macro"

    if (scorer == 0):
            fit = accuracy_score(y_true, y_pred)      
    elif (scorer == 1):
        fit = roc_auc_score(y_true, y_pred)
    elif (scorer == 2):
        fit = f1_score(y_true, y_pred, average = average)
    elif (scorer == 3):
        fit = average_precision_score(y_true, y_pred, average = averageMode)
    return fit

def CrossValidation(slr, cX , cY, scorer, k, averageMode):
        # Load your data into a cuDF dataframe
        X = cudf.DataFrame(cX)  # Your features
        y = cudf.Series(cY)     # Your target variable
            

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0 )

        # Shuffle the data
        shuffle_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train.iloc[shuffle_indices]
        y_train_shuffled = y_train.iloc[shuffle_indices]

        # Determine the size of each fold
        fold_size = len(X_train) // k

        bestCrossScore = 0
        bestModel = 0
        
        # Perform k-fold cross-validation
        for i in range(k):
            # Determine the start and end indices for the current fold
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else len(X_train)
            
            # Split the data into training and validation sets for this fold
            X_val_fold = X_train_shuffled[start:end]
            y_val_fold = y_train_shuffled[start:end]
            X_train_fold = cudf.concat([X_train_shuffled[:start], X_train_shuffled[end:]])
            y_train_fold = cudf.concat([y_train_shuffled[:start], y_train_shuffled[end:]])
            
            # Train the model on the training fold
            slr.fit(X_train_fold.to_numpy(), y_train_fold.to_numpy())
            
            # Evaluate the model on the validation fold
            y_pred = make_predictions(slr,scorer,X_val_fold.to_numpy())
            
            if np.isnan(y_pred).any():
                nan_indices = np.isnan(y_pred)
                y_pred[nan_indices] = 0
            score = evaluationMetrics(scorer, y_val_fold.to_numpy(), y_pred, averageMode)

            if math.isnan(score) or math.isinf(score):
                score = 0.01
            
            if score > bestCrossScore:
                bestCrossScore = score
                bestModel = copy.deepcopy(slr)
     
        y_test_pred = make_predictions(bestModel, scorer, X_test.to_numpy())
        if np.isnan(y_test_pred).any():
            nan_indices = np.isnan(y_test_pred)
            y_test_pred[nan_indices] = 0

        test_score = evaluationMetrics(scorer, y_test.to_numpy(), y_test_pred, averageMode)
        return test_score, bestModel


def make_predictions(slr,scorer,X):
    if (scorer != 3):
        y_pred = slr.predict(X)
    else:
        y_pred = slr.predict_proba(X)
    return y_pred

