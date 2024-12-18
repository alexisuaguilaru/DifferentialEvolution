from multiprocess import Pool
from time import time
import os
import pandas as pd

from typing import Callable

def SimulateOptimizer(Optimizer:Callable,KwargsOptimizer:dict,NumberSimulations:int=1000,Processes_Jobs:int=None) -> tuple[list,list,float]:
    """
        Function for simulating several runs/callings of the optimizer
    
        -- Optimizer:Callable :: Optimizer to simulate
        
        -- KwargsOptimizer:dict :: Keyword arguments passed 
        to the optimizer 

        -- NumberSimulations:int :: Number of executions or 
        simulations of the optimizer

        -- Processes_Jobs:int :: Number of simulations that 
        are executing at same time

        Return two lists which contain data about function 
        evaluations and optimal values at each generation 
        and simulation/run. Also return the time of execution
    """
    def FunctionOptimizer(KwargsOptimizer:dict) -> list[tuple[int,float]]:
        """
            Auxiliar function for wrap optimizer and 
            catch its results
            
            -- KwargsOptimizer:dict :: Keyword arguments passed 
            to the optimizer 

            Return a list with the relevant results at each generation
        """
        _ , snapshots = Optimizer(**KwargsOptimizer)
        
        functionEvaluations_Optimals = []
        for snapshot in snapshots:
            functionEvaluations_Optimals.append([snapshot[1],snapshot[2]])
        
        return functionEvaluations_Optimals

    with Pool(Processes_Jobs) as poolExecutions:
        startTime = time()
        simulationResults = poolExecutions.map(FunctionOptimizer,[KwargsOptimizer for _ in range(NumberSimulations)])
        endTime = time()

    simulationsFunctionEvaluations = []
    simulationsOptimals = []
    for simulationResult in simulationResults:
        simulationsFunctionEvaluations.append([generation[0] for generation in simulationResult])
        simulationsOptimals.append([generation[1] for generation in simulationResult])

    timeExecution = endTime - startTime

    return simulationsFunctionEvaluations , simulationsOptimals , timeExecution

def ConvertResultsCSV(OptimizerName:str,SimulationResults:list[list],TypeResult:str,FunctionNumber:str,Dimension:int,YearCEC:str) -> None:
    """
        Function for converting a list of results into a csv file

        -- OptimizerName:str :: Name of the optimizer
    
        -- SimulationResults:list :: Results to convert
        
        -- TypeResult:str :: Type of result. Function evaluations or optimal values
        
        -- FunctionNumber:str :: Function's name associated to results
        
        -- Dimension:int :: Problem's dimension
        
        -- YearCEC:str :: CEC problem's year
    """
    fileName , dataType = (f'FunctionEvaluations_F{FunctionNumber}' , int) if TypeResult == 'F' else (f'Optimals_F{FunctionNumber}' , float)

    try:
        os.open(f'Results/CEC_{YearCEC}/Dim_{Dimension}/{OptimizerName}')
    except:
        if f'CEC_{YearCEC}' not in os.listdir('Results'):
            os.mkdir(f'Results/CEC_{YearCEC}')
        
        if f'Dim_{Dimension}' not in os.listdir(f'Results/CEC_{YearCEC}'):
            os.mkdir(f'Results/CEC_{YearCEC}/Dim_{Dimension}')
        
        if f'{OptimizerName}' not in os.listdir(f'Results/CEC_{YearCEC}/Dim_{Dimension}'):
            os.mkdir(f'Results/CEC_{YearCEC}/Dim_{Dimension}/{OptimizerName}')
        
    pd.DataFrame(SimulationResults,dtype=dataType).to_csv(f'Results/CEC_{YearCEC}/Dim_{Dimension}/{OptimizerName}/{fileName+'.csv'}',header=None,index=None)