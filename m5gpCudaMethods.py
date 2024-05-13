# *********************************************************************
# Name: m5gpCudaMethods.py
# Description: Modulo que implementa las metodos para ejecutar codigo utilizando
# nucleos de CUDA para su ejecucion en paralelo
# Se utiliza la libreria de numba.
# *********************************************************************
 
from numba import cuda
from numba import njit, literal_unroll
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32,
                               xoroshiro128p_normal_float32,
							   xoroshiro128p_normal_float64)

import math
import numpy as np
import m5gpGlobals as gpG

def gpuMaxUseProc(Individuals) :
	blocksize = 1
	#gridsize = 2147483647
    
	gridsize=(Individuals + blocksize-1) // blocksize
	while(gridsize > 1024) :
		if  (blocksize == 1):
			blocksize = 0
		blocksize = blocksize + 32
		gridsize=(Individuals + blocksize-1) // blocksize
		
	MaxOcup = {}
	MaxOcup["BlockSize"] = gridsize
	MaxOcup["GridSize"] = blocksize	

	return MaxOcup

@cuda.jit()
def Truncate(f, n) :
	if (f < 0) :
		f2 = f * (-1)
	else :
		f2 = f
	T =  math.floor(f2 * 10 ** n) / 10 ** n
	if (f < 0) :
		T = T * (-1)
	return T

#@cuda.jit(nopython=True)
#def empty():
#    return np.empty(5, np.float64)  # np.float64 instead of np.float


@njit
def zeros(max):
    return np.zeros(max, dtype=np.float64)

@cuda.jit()
def initialize_population (cu_states,
        dInitialPopulation,
        numIndividuals,
        nvar,
        sizeMaxDepthIndividual,
        maxRandomConstant,
        genOperatorProb,
        genVariableProb,
        genConstantProb,
        genNoopProb,
        useOpIF ) :

    #const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x 
	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

	if (tid >= numIndividuals) :
		return

	if (tid >= (numIndividuals * sizeMaxDepthIndividual)) :
		return
	
	for j in range(sizeMaxDepthIndividual) :
		gene = gpG.OP_NOOP
		# Obtenemos operador o (variable/constante) o NOOP */
		prob = xoroshiro128p_uniform_float32(cu_states, tid)
	    
		# Verificamos la probabilidad de que sea un Operador */
		if (prob <= genOperatorProb) :
            # Es un Operador
			#  1 = Suma
			#  2 = Resta
			#  3 = Multiplicacion
			#  4 = Division
			#  5 = Seno
			#  6 = Coseno
			#  7 = Exponente
			#  8 = Logaritmo
			#  9 = Valor Absoluto

			# 10 = Sumatoria (* Future use)
			# 11 = Producto (* Future use)
			# 12 = Promedio (* Future use)
			# 13 = Desviacion Standard (* Future use)

			# ***************************************************************
			# Si hay nuevos operadores/funciones, ponerlas en este espacio,
			# entre el ultimo operador agregado y los IF. Los IF incrementan
			# su valor.
			# ***************************************************************
			# 10 = Operador IFMAYOR
			# 11 = Operador IFMENOR
			# 12 = Operador IFIGUAL
			# 99 = NOOP

			numOp = (gpG.OP_IFG * (-1)) - 10000 + useOpIF - 1
			op1 = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % numOp) + 1			
			op = Truncate(op1, 0)
			if (op == gpG.OP_IFG and useOpIF == 1) :   # Fue un IF
                # Fue un IF, obtenemos la condicion de manera aleatoria
				cond = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % 3) + 1
				cond = Truncate(cond, 0)
				if (cond == 1) : # IFMAYOR
					op = gpG.OP_IFG 
				elif (cond == 2) : # IFMENOR
					op = gpG.OP_IFL
				elif (cond == 3) : # IFIGUAL
					op = gpG.OP_IFE
				else :
					op = gpG.OP_NOOP # 13 - NOOP
				gene = op
			else :
				gene = ((op * (-1)) + gpG.OP_INI)
            #Fin de If

		elif ((prob > genOperatorProb) and (prob <= (genVariableProb+genOperatorProb))) :
            # Obtenemos la probabilidad de que sea una variable */
			gene = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % (nvar)+1000) * (-1)
			gene = Truncate(gene, 0)
		elif ((prob > (genVariableProb+genOperatorProb)) and (prob <= (genVariableProb+genOperatorProb+genConstantProb))) :
            # Obtenemos la probabilidad de que sea una constante */
			gene = ((xoroshiro128p_normal_float64(cu_states, tid)*1000)  % maxRandomConstant+1)
			gene = Truncate(gene, 0)
			prob = xoroshiro128p_uniform_float32(cu_states, tid)
            #  Probabilidad de que la constante sea positiva o negativa */
			if (prob < 0.5) :
				gene = gene * (-1)         
		else :
            # Obtenemos la probabilidad de que sea un Operador NOOP */
			gene = gpG.OP_NOOP 	# Obtenemos la probabilidad de que sea un Operador NOOP */

		dInitialPopulation[tid*sizeMaxDepthIndividual+j] = int(gene)
	# Fin del FOR
	return

@cuda.jit(device=True)
def isEmpty(pushGenes, sizeMaxDepthIndividual) :
    if (pushGenes <= 0) :
        return True
    else :
        return False

# Remove all elements from the stack so that in the next evaluations there are no previous values of other individuals """
#@numba.jit(nopython=True, nogil=True, cache=True) 
@cuda.jit(device=True)
def clearStack(sizeMaxDepthIndividual, dStack:np.array) :
	for i in range(sizeMaxDepthIndividual):
		dStack[i] = 0
	return 0


@cuda.jit(device=True)
def push(val, pushGenes, dStack) :
	dStack[pushGenes] = val
	return pushGenes+1

@cuda.jit(device=True)
def pop(pushGenes, dStack) :
	pushGenes = pushGenes - 1
	return dStack[pushGenes]

@cuda.jit(device=True)
def pushMod(val, pushModel, stackModel) :
	stackModel[pushModel] = val
	return pushModel+1


@cuda.jit(device=True)
def popMod(pushModel, stackModel) :
    pushModel = pushModel - 1
    return stackModel[pushModel]


@cuda.jit
def compute_individuals(inputPopulation,
                        outIndividuals,
                        data,
                        numIndividuals,
                        sizeMaxDepthIndividual,
                        nrow,
                        nvar,
                        uStack: np.ndarray,
                        uStackIdx: np.ndarray,
                        model,
                        stackModel: np.ndarray ) :

	#const unsigned int tidSem = threadIdx.x + blockIdx.x * blockDim.x 
	tidSem = cuda.grid(1)
	#tidSem = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	
	out = gpG.MAX_RMSE
	pushGenes = 0
	pushModel = 0

	if (tidSem >= (numIndividuals * nrow)) :
		return

	# Obtenemos el numero de renglon del individuo que corresponde 
	tid =  int(tidSem / nrow)
	if (tid < 0) :
		tid = 0

	# Obtenemos el numero de elemento o renglon de la matriz de entrenamiento 
	k = tidSem - (tid*nrow)

	maxVar = (1000 + nvar -1) * (-1)

	# Clear stack
	#for i in range(sizeMaxDepthIndividual):
	#		uStack[tidSem*sizeMaxDepthIndividual +i] = 0

	#nTotalGenes = 0
	for i in range(sizeMaxDepthIndividual) :
		t_ = 0
		tmp = 0
		tmp2 = 0
		inputPop = inputPopulation[tid*sizeMaxDepthIndividual+i]	

   	    # *************************** Es una constante ******************************
		if ((inputPop >= gpG.MIN_CONSTANT) and (inputPop <= gpG.MAX_CONSTANT)) : # Es una constante
			uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = inputPop
			pushGenes += 1
			out = inputPop 
			if (model == 1) :
				stackModel[tidSem*sizeMaxDepthIndividual+pushModel] = inputPop
				pushModel += 1
			continue
		# *************************** Es una variable ******************************
		elif ((inputPop <= -1000) and (inputPop >= maxVar) and ((inputPop - int(inputPop)) == 0)) : # Es una variable
			t = int(inputPop)
			t_ = (t+1000)*(-1)
			uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = data[t_+nvar*k]
			pushGenes += 1
			out = data[t_+nvar*k]
			if (model == 1) :
				stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
				pushModel += 1	
			continue
		# *************************** Es un operador de suma ******************************
		elif (inputPop == gpG.OP_ADD) :   # Es Suma
			if (not isEmpty(pushGenes, sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
				if (not isEmpty(pushGenes, sizeMaxDepthIndividual)) :
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]					
					if (not math.isnan(tmp) and not math.isinf(tmp) and not math.isnan(tmp2) and not math.isinf(tmp2)) :
						out = tmp + tmp2
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1
			continue
		# *************************** Es un operador de resta ******************************
		elif (inputPop == gpG.OP_SUB) :    # Es Resta
			if(not isEmpty(pushGenes, sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]				
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]					
					if (not math.isnan(tmp) and not math.isinf(tmp) and not math.isnan(tmp2) and not math.isinf(tmp2)) :
						out = tmp - tmp2
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1						
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1							
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1		
			continue			
    	# *************************** Es un operador de multiplicacion ******************************/
		elif (inputPop == gpG.OP_MUL) :    # Es multiplicacion
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]				
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]						
					if (not math.isnan(tmp) and not math.isinf(tmp) and not math.isnan(tmp2) and not math.isinf(tmp2)) :
						out = tmp * tmp2
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1							
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1							
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1		
			continue				
    	# *************************** Es un operador de division ******************************/
		elif (inputPop == gpG.OP_DIV) :   # Es division
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]				
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :	
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
					if (not math.isnan(tmp) and not math.isinf(tmp) and not math.isnan(tmp2) and not math.isinf(tmp2)) :
						out = tmp / math.sqrt(1 + (tmp2 * tmp2))
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1							
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1							
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1	
			continue				
		# *************************** Es un operador de seno ******************************/
		elif (inputPop == gpG.OP_SIN) :    # Es seno
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				#tmp = pop(pushGenes,uStack[tidSem*sizeMaxDepthIndividual])
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]				
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = math.sin(tmp)
					#out = sin(tmp * PI / 180.0) 
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
					pushGenes += 1						
					if (model == 1) :
						stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
						pushModel += 1		
			continue					
		# *************************** Es un operador de coseno ******************************/
		elif (inputPop == gpG.OP_COS) :   # Es coseno
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]				
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = math.cos(tmp)
					#out = cos(tmp * PI / 180.0 ) 
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
					pushGenes += 1						
					if (model == 1) :
						stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
						pushModel += 1	
			continue					
		# *************************** Es un operador de exponente ******************************/
		elif (inputPop == gpG.OP_EXP) :    # Es exponente
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = math.exp(tmp)
					if (not math.isnan(out) and not math.isinf(out) ) :
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1							
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1							
					else :
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
						pushGenes += 1		
			continue					
		# *************************** Es un operador de logaritmo ******************************/
		elif (inputPop == gpG.OP_LOG) :    # Es logaritmo
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					if (tmp > 0) :
						out = math.log(tmp) 
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
						pushGenes += 1							
						if (model == 1) :
							stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
							pushModel += 1							
					else :
						uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
						pushGenes += 1		
			continue					
		# *************************** Es un operador de absoluto ******************************/
		elif (inputPop == gpG.OP_ABS) :    #  Es absoluto
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]		
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = math.fabs(tmp) 
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
					pushGenes += 1						
					if (model == 1) :
						stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
						pushModel += 1	
			continue	
		# *************************** Es un operador de SUMATORIA ******************************/
		elif (inputPop == gpG.OP_SUM) :  
			cont = 0
			out = 0
			if (not isEmpty(pushGenes, sizeMaxDepthIndividual) and 
	   			pushGenes > 1) :
				cont = 1
				#Jalamos un primer elemento del stack
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = tmp
					while (pushGenes > 0):
						#Jalamos un siguiente elemento del stack si hay
						pushGenes -=  1
						tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
						if (not math.isnan(tmp2) and not math.isinf(tmp2)) :
							#print("Sumatoria tmp2:", tmp2, " pushgenes:", pushGenes)
							# Haz la sumatoria
							out += tmp2
							cont += 1
					# Fin del While
						
				if (model == 1) :
					stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
					pushModel += 1	
				#Fin del if

				uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
				pushGenes += 1					
			# Fin del if
			continue
		# *************************** Es un operador de PRODUCTO ******************************/
		elif (inputPop == gpG.OP_PRD) :  
			cont = 0
			out = 0
			if (not isEmpty(pushGenes, sizeMaxDepthIndividual)and 
	   			pushGenes > 1) :
				cont = 1
				#Jalamos un primer elemento del stack
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = tmp
					while (pushGenes > 0):
						#Jalamos un siguiente elemento del stack si hay
						pushGenes -=  1
						tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
						if (not math.isnan(tmp2) and not math.isinf(tmp2)) :
							#print("Producto tmp2:", tmp2, " pushgenes:", pushGenes)
							# Haz el producto
							out *= tmp2
							cont += 1
					# Fin del While
						
				if (model == 1) :
					stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
					pushModel += 1	
				#Fin del if

				uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
				pushGenes += 1					
			# Fin del if
			continue
		# *************************** Es un operador de PROMEDIO ******************************/
		elif (inputPop == gpG.OP_PRM) :  
			cont = 0
			out = 0
			if (not isEmpty(pushGenes, sizeMaxDepthIndividual)and 
	   			pushGenes > 1) :
				cont = 1
				#Jalamos un primer elemento del stack
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					out = tmp
					while (pushGenes > 0):
						#Jalamos un siguiente elemento del stack si hay
						pushGenes -=  1
						tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]	
						if (not math.isnan(tmp2) and not math.isinf(tmp2)) :
							#print("Promedio tmp2:", tmp2, " pushgenes:", pushGenes)
							# Haz la sumatoria
							out += tmp2
							cont += 1
					# Fin del While
						
				if (model == 1) :
					stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
					pushModel += 1	
				#Fin del if

				#Sacamos el promedio
				out = out / cont

				uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
				pushGenes += 1					
			# Fin del if

			continue
		# *************************** Es un operador de DESV ESTANDARD ************************/
		elif (inputPop == gpG.OP_DVS) :  
			#print("Operador DevStd")
			cont = 0
			out = 0
			suma = 0
			#pp = pushGenes 
			#arr = zeros(pp)
			arr = np.float64[:]

			if (not isEmpty(pushGenes, sizeMaxDepthIndividual) and 
	   			pushGenes > 1) :
				cont = 1
				#Jalamos un primer elemento del stack
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
				#print("tmp:", tmp, " pushgenes:", pushGenes)
				if (not math.isnan(tmp) and not math.isinf(tmp)) :
					arr[0] = tmp
					suma = tmp
					while (pushGenes > 0):
						#Jalamos un siguiente elemento del stack si hay
						pushGenes -=  1
						tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]						
						if (not math.isnan(tmp2) and not math.isinf(tmp2)) :
							#print("Desv Std tmp2:", tmp2, " pushgenes:", pushGenes)
							cont += 1
							arr[cont] = tmp2
							suma += tmp2					
				# Fin del While
						
				if (model == 1) :
					stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= inputPop
					pushModel += 1	
				#Fin del if

				#Sacamos la desviacion standard
				prom = suma / cont  #promedio

				s0 = 0
				for ds in range(cont):
					s1 = math.fabs((arr[ds] - prom) ** 2)
					#print("Desv Std:", arr[ds])
					s0 += s1
				out= math.sqrt(s0 / cont)
				
				uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
				pushGenes += 1	
			continue
    	# *************************** Es una condicion de IFMAYOR ******************************/
		elif (inputPop == gpG.OP_IFG) :    #  Es IF MAYOR
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : #  Verificamos que haya un primer elemento
				# Si hay, sacamos el primer elemento
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]					
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos si haya un segundo elemento
					# Si hay, sacamos un segundo elemento
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]
					if (not math.isnan(tmp) or not math.isnan(tmp2)) :
						if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos que haya un tercer elemento
							# Sacamos el tercer elemento
							pushGenes -=  1
							tmp3 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]							
							if (tmp > tmp2 ) :  # Evaluamos la condicion con los primeros dos elementos obtenidos IFMAYOR
								out = tmp3 	
								uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
								pushGenes += 1								
							else :
								if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Si es falso, verificamos que haya un cuarto elemento para el ELSE   
									# Sacamos el cuarto elemento
									pushGenes -=  1
									tmp4 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]										
									out = tmp4 
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
									pushGenes += 1										
								else : # No hay un cuarto elemento, regresamos los anteriores tres a la pila
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp3
									pushGenes += 1		
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
									pushGenes += 1									
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
									pushGenes += 1									
						else :
							# No hay un tercer elemento, regresamos los anteriores dos a la pila
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
							pushGenes += 1							
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
							pushGenes += 1
				else :
					#pushGenes = push(tmp,pushGenes,uStack[tidSem*sizeMaxDepthIndividual]) 
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1
			continue					
		# *************************** Es una condicion de IFMENOR ******************************/
		elif (inputPop == gpG.OP_IFL) : #  Es IF MENOR
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : #  Verificamos que haya un primer elemento
				# Si hay, sacamos el primer elemento
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]									
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos si haya un segundo elemento
					# Si hay, sacamos un segundo elemento
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]					
					if (not math.isnan(tmp) or not math.isnan(tmp2)) :
						if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos que haya un tercer elemento
							# Sacamos el tercer elemento
							pushGenes -=  1
							tmp3 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]								
							if (tmp < tmp2 ) :  # Evaluamos la condicion con los primeros dos elementos obtenidos IFMAYOR
								out = tmp3 
								uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
								pushGenes += 1								
							else :
								if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Si es falso, verificamos que haya un cuarto elemento para el ELSE
									# Sacamos el cuarto elemento
									pushGenes -=  1
									tmp4 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]										
									out = tmp4 
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
									pushGenes += 1									
								else : # No hay un cuarto elemento, regresamos los anteriores tres a la pila
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp3
									pushGenes += 1		
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
									pushGenes += 1									
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
									pushGenes += 1	
						else : # No hay un tercer elemento, regresamos los anteriores dos a la pila
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
							pushGenes += 1							
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
							pushGenes += 1
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1			
			continue							
		# *************************** Es una condicion de IFIGUAL ******************************/
		elif (inputPop == gpG.OP_IFE) : #  Es IF IGUAL
			if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : #  Verificamos que haya un primer elemento
				# Si hay, sacamos el primer elemento
				pushGenes -=  1
				tmp = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]									
				if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos si haya un segundo elemento
					# Si hay, sacamos un segundo elemento
					pushGenes -=  1
					tmp2 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]					
					if (not math.isnan(tmp) or not math.isnan(tmp2)) :
						if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Verificamos que haya un tercer elemento
							# Sacamos el tercer elemento
							pushGenes -=  1
							tmp3 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]								
							if (tmp == tmp2 ) :  # Evaluamos la condicion con los primeros dos elementos obtenidos IFMAYOR
								out = tmp3 
								uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
								pushGenes += 1								
							else :
								if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) : # Si es falso, verificamos que haya un cuarto elemento para el ELSE
									# Sacamos el cuarto elemento
									pushGenes -=  1
									tmp4 = uStack[tidSem*sizeMaxDepthIndividual+pushGenes]										
									out = tmp4 
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = out
									pushGenes += 1									
								else : # No hay un cuarto elemento, regresamos los anteriores tres a la pila
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp3
									pushGenes += 1		
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
									pushGenes += 1									
									uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
									pushGenes += 1	
						else : # No hay un tercer elemento, regresamos los anteriores dos a la pila
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp2
							pushGenes += 1							
							uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
							pushGenes += 1
				else :
					uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = tmp
					pushGenes += 1		
			continue								
		elif (inputPop == gpG.OP_NOOP or inputPop == gpG.OP_FIN) : #  Es NoOP, no hacemos nada
			#if (not isEmpty(pushGenes,sizeMaxDepthIndividual)) :
			out = gpG.MAX_RMSE
			#if (inputPop == -11111)	:
			#	print("inputPop:", inputPop)
		else : # No fue ninguno de los anteriores, es una constante
			uStack[tidSem*sizeMaxDepthIndividual+pushGenes] = inputPop
			pushGenes += 1
			out = inputPop 
			if (model == 1) :
				stackModel[tidSem*sizeMaxDepthIndividual+pushModel] = inputPop
				pushModel += 1		

		#if (tidSem == 0) :
		#	print("Input: ", inputPop, " Out(",i,"):",out, " Stack Idx: ", pushGenes)
	# Fin del for		

	if(math.isnan(out) or math.isinf(out)) :
		out = gpG.MAX_RMSE	

	outIndividuals[tidSem] = out 
	uStackIdx[tidSem] = pushGenes

	if (model == 1 and pushModel < sizeMaxDepthIndividual) :
		stackModel[tidSem*sizeMaxDepthIndividual + pushModel]= gpG.OP_FIN
		pushModel += 1	

	#if (model == 0) :
	#	print("outIndividuals[", tidSem, "]:", outIndividuals[tidSem])

	return


@cuda.jit
def computeRMSE(semantics,
				targetValues,
				fit,
				numIndividuals,
				nrow) :
	"""
	Function that calculates the fitness of an individual using the information stored in its semantic vector
	Args:
		semantics (NDArray): Vector of pointers that contains the semantics of the individuals of the initial population
		targetValues (NDArray): Contain the target values of train or test
		fit (NDArray): Vector that will store the error of each individual in the population
		numIndividuals (int): Number of individuals in the population
		nrow (int): Number of rows (instances) of the training and test dataset
	"""	

	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	if (tid >= numIndividuals) :
		return

	temp = 0
	for i in range(nrow):
		temp += (semantics[tid*nrow+i]-targetValues[i])*(semantics[tid*nrow+i]-targetValues[i])

	temp = math.sqrt(temp/nrow)
 
	if(math.isnan(temp) or math.isinf(temp) or (temp > gpG.MAX_RMSE)) :
		temp = gpG.MAX_RMSE

	fit[tid] = temp
	return

@cuda.jit
def computeR2(semantics,
				targetValues,
				fit,
				numIndividuals,
				nrow) :
	residual1=0 
	residual2=0
	total_m = 0
	y_mean =0
	sum_squared_residual=0
	sum_squared_total=0

	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	if (tid >= numIndividuals) :
		return

	# Calculate targets means
	for i in range(nrow):
		total_m += targetValues[i]

	y_mean = (total_m/nrow)

	# Calculate residual_sum_of_square and total_sum_of_square
	for i in range(nrow):
		# Calculate residual_sum_of_square 
		residual1 = (targetValues[i] - semantics[tid*nrow+i])
		sum_squared_residual = sum_squared_residual + (residual1 * residual1)		

		# Calculate total_sum_of_square
		residual2 = (targetValues[i] - y_mean)
		sum_squared_total = sum_squared_total + (residual2 * residual2)

	fit[tid] = (1 - (sum_squared_residual / sum_squared_total))

	if(math.isnan(fit[tid]) or math.isinf(fit[tid]) or  (fit[tid] > 2.0) or (fit[tid] < gpG.MAX_R2_NEG)) :
		fit[tid] = gpG.MAX_R2_NEG
		
	fit[tid] = fit[tid] + (gpG.MAX_R2_NEG * (-1))
	return


@cuda.jit
def parent_select_tournament(cu_states,  # states
							g_newPopulation, #dNewPopulation,
                            g_idata,  #dInitialPopulation,
                            g_uFit,  #dFit,
                            dBestParentsTournament,
                            tsizeTournament,  #sizeTournament,
                            numIndividuals,
                            sizeMaxDepthIndividual ) :
	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	if (tid >= numIndividuals) :
		return

	competitor = 0

	#print("numIndividuals:", numIndividuals)
	if (numIndividuals > 1) :

		#prob = xoroshiro128p_uniform_float32(cu_states, tid)
		#op = xoroshiro128p_normal_float32(cu_states, tid)  % numOp + 1
		# Indice aleatorio del padre considerando toda la poblacion
		id_best = (xoroshiro128p_normal_float64(cu_states, tid)*1000)  % numIndividuals
		id_best = int(Truncate(id_best, 0))

		for i in range(tsizeTournament):
			# * Vericamos que el padre no compita con el mismo */
			competitor = id_best
			while (competitor == id_best) :
				# se obtiene un competidor de manera aleatoria
				competitor = (xoroshiro128p_normal_float64(cu_states, tid)*1000) % numIndividuals
				competitor = int(Truncate(competitor, 0))

			# Si el competidor es mejor que el padre
			if (g_uFit[competitor] < g_uFit[id_best]) :  
				# g_fitBasedProb[tid] = g_idata[competitor]; // Se saca una copia del individuo competidor mejor y toma el lugar del padre
				id_best = competitor; # el indice del competidor es ahora el del padre

		dBestParentsTournament[tid] = int(id_best)
		# memcpy(&g_newPopulation[tid * sizeMaxDepthIndividual], &g_idata[tid * sizeMaxDepthIndividual], sizeof(float) * sizeMaxDepthIndividual);
	else :
		dBestParentsTournament[tid] = 0
	

	for i in range(sizeMaxDepthIndividual):
		g_newPopulation[tid * sizeMaxDepthIndividual + i] = g_idata[tid * sizeMaxDepthIndividual + i]

	return

@cuda.jit
def umadMutation(cu_states,  # states
			g_Population, # dNewPopulation
			g_idata,   # dInitialPopulation
			dBestParentsTournament, 
			numIndividuals,
			sizeMaxDepthIndividual, 
			nrow, 
			nvar,
			mutationProb, 
			mutationDeleteRateProb, 
			maxRandomConstant, 
			genOperatorProb, 
			genVariableProb,
			genConstantProb, 
			genNoopProb, 
			useOpIF) :

	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	if (tid >= numIndividuals) :
		return

	additionRate = mutationProb
	deletionRate = additionRate / (1 + additionRate)

	bestParent = int(dBestParentsTournament[tid])
	prob1 = 0 
	prob2 = 0

	for j in range(sizeMaxDepthIndividual):
		g_Population[tid * sizeMaxDepthIndividual + j] = g_idata[bestParent
				* sizeMaxDepthIndividual + j]

		prob1 = xoroshiro128p_uniform_float32(cu_states, tid)
		if (prob1 <= additionRate) :
			# Cae en la probabilidad de ser modificado
			# Obtenemos un nuevo gen, el actual gen es modificado
			gene = gpG.OP_NOOP

			gene = gpG.OP_NOOP
			# Obtenemos operador o (variable/constante) o NOOP */
			prob = xoroshiro128p_uniform_float32(cu_states, tid)
			
			# Verificamos la probabilidad de que sea un Operador */
			if (prob <= genOperatorProb) :
				# Es un Operador
				#  1 = Suma
				#  2 = Resta
				#  3 = Multiplicacion
				#  4 = Division
				#  5 = Seno
				#  6 = Coseno
				#  7 = Exponente
				#  8 = Logaritmo
				#  9 = Valor Absoluto
				# 10 = Sumatoria (* Future use)
				# 11 = Producto (* Future use)
				# 12 = Promedio (* Future use)
				# 13 = Desviacion Standard (* Future use)
				# ***************************************************************
				# Si hay nuevos operadores/funciones, ponerlas en este espacio,
				# entre el ultimo operador agregado y los IF. Los IF incrementan
				# su valor.
				# ***************************************************************
				# 10 = Operador IFMAYOR
				# 11 = Operador IFMENOR
				# 12 = Operador IFIGUAL
				# 99 = NOOP

				numOp = (gpG.OP_IFG * (-1)) - 10000 + useOpIF - 1
				op1 = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % numOp) + 1
				op = Truncate(op1, 0)
				if (op == gpG.OP_IFG and useOpIF == 1) :   # Fue un IF
					# Fue un IF, obtenemos la condicion de manera aleatoria
					cond = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % 3) + 1
					cond = Truncate(cond, 0)
					if (cond == 1) : # IFMAYOR
						op = gpG.OP_IFG 
					elif (cond == 2) : # IFMENOR
						op = gpG.OP_IFL
					elif (cond == 3) : # IFIGUAL
						op = gpG.OP_IFE
					else :
						op = gpG.OP_NOOP # 13 - NOOP
					gene = op
				else :
					gene = ((op * (-1)) + gpG.OP_INI)
				#Fin de If

			elif ((prob > genOperatorProb) and (prob <= (genVariableProb+genOperatorProb))) :
				# Obtenemos la probabilidad de que sea una variable */
				gene = ((xoroshiro128p_normal_float64(cu_states, tid)*1000) % (nvar)+1000) * (-1)
				gene = Truncate(gene, 0)
			elif ((prob > (genVariableProb+genOperatorProb)) and (prob <= (genVariableProb+genOperatorProb+genConstantProb))) :
				# Obtenemos la probabilidad de que sea una constante */
				gene = ((xoroshiro128p_normal_float64(cu_states, tid)*1000)  % maxRandomConstant+1)
				gene = Truncate(gene, 0)
				prob = xoroshiro128p_uniform_float32(cu_states, tid)
				#  Probabilidad de que la constante sea positiva o negativa */
				if (prob < 0.5) :
					gene = gene * (-1)         
			else :
				# Obtenemos la probabilidad de que sea un Operador NOOP */
				gene = gpG.OP_NOOP 	# Obtenemos la probabilidad de que sea un Operador NOOP */

			g_Population[tid * sizeMaxDepthIndividual + j] = int(gene)
		#end if

		prob2 = xoroshiro128p_uniform_float32(cu_states, tid)

		if (mutationDeleteRateProb >= 0) :
			deletionRate = mutationDeleteRateProb

		if (prob2 <= deletionRate) :
			# Cae en la probabilidad de ser eliminado (NOOP)
			g_Population[tid * sizeMaxDepthIndividual + j] = int(gpG.OP_NOOP)
	# Fin de for
	return

@cuda.jit
def replace(g_parents, 
			g_newPopulation,
			rmse_parents, 
			rmse_offspring, 
			numIndividuals,
			sizeMaxDepthIndividual) :

	tid = cuda.grid(1)
	#tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	if (tid >= numIndividuals) :
		return

	rmse_parents[tid] = rmse_offspring[tid]
	for i in range(sizeMaxDepthIndividual):
		g_parents[tid * sizeMaxDepthIndividual + i] =	g_newPopulation[tid * sizeMaxDepthIndividual + i]

	return
