from m5gp import m5gpRegressor as m5gp
from   sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#load the data
#dataset = pd.DataFrame(pd.read_csv("/home/treelab/python-codes/data/Concrete/train_10107_1.txt" ,sep='\s+', header=None))
dataset = pd.DataFrame(pd.read_csv("/home/treelab/srbench1/pmlb/datasets5/589_fri_c2_1000_25/589_fri_c2_1000_25.tsv" ,sep='\s+', header=None))
nrows = len(dataset.index)
nvar = dataset.shape[1] - 1
#print("Leyo X")
X = dataset.iloc[0:nrows, 0:nvar-1]
y = dataset.iloc[:nrows, nvar-1]

x_train = dataset.iloc[0:nrows, 0:nvar-1].to_numpy().astype(np.float32)
y_train = dataset.iloc[:nrows, nvar-1].to_numpy().astype(np.float32)


#X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)

print('Running m5gp ...')
 
est = m5gp(
            generations=30, # number of generations (limited by default)
            Individuals=256, # number of individuals
            GenesIndividuals=128, # number of genes per individual
            mutationProb=0.1, # mutation rate probability
            mutationDeleteRateProb=0.01,  # mutation delete rate probality
            sizeTournament=0.15, # size of tournament
            evaluationMethod=0,  #error evaluation method 
                        # 0=RMSE, 
                        # 1=R2, 
                        #cuML Methods
                        # 2=LinearRegression, 3=Lasso Regression, 
                        # 4=Ridge regression, 5=kernel Ridge Regression,
                        # 6=ElasticNet Regression
                        #cuML MiniBatch options
                        # 7=MiniBatch none regularization (linear regression)
                        # 8=MiniBatch lasso regularization 
                        # 9=MiniBatch ridge regularization 
                        #10=MiniBatch elasticnet regularization 
            maxRandomConstant=999, #number of constants (-maxRandomConstant to maxRandomConstant)
            genOperatorProb=0.50, #probablity for generate Operators 
            genVariableProb=0.39, #probablity for generate variables 
            genConstantProb=0.1, #probablity for generate constants
            genNoopProb=0.01, #probablity for generate NOOP Operators 
			useOpIF=False, #Set if use IF operator
            log=1, #save log files
			verbose=1, #Show menssages on execution
            logPath='log/' #path for logs
 )


#ea.cudacapabilities()

est.fit(x_train, y_train)

print("Complexity: ", est.complexity())

yPredicted = est.predict(x_train)
#print("Y Data :\n", y_train)
#print("Y Predicted:\n", yPredicted)

mse = est.meanSquaredError(y_train, yPredicted)
print("mse: ", mse)
print("rmse:", est.rmse(y_train, yPredicted))