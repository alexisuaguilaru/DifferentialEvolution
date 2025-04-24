from joblib import Parallel , delayed
import numpy as np

from time import time
import os

from typing import Iterable

class SummaryResults:
    def __init__(self,Optimizer,*ArgsInitOptimizer,**KwargsInitOptimizer):
        """
            Class for generating a results summary of 
            simulations of a optimizer 
            
            -- Optimizer :: Optimizer class from which the summary is generated. 
            It is expected that when called, it returns a tuple where the second 
            element is the snapshots of the optimal values.

            -- ArgsInitOptimizer :: Arguments for initialize the optimizer 

            -- KwargsInitOptimizer :: Keyword arguments for initialize the optimizer 
        """
        self.Optimizer = Optimizer(*ArgsInitOptimizer,**KwargsInitOptimizer)

    def __call__(self,SimulationTimes:int,*ArgsOptimizer,DirResults:str='./',NameResults:str='results.csv',ThreadPool:int=1,**KwargsOptimizer) -> float:
        """
            Method for resuming results of several 
            simulations of the optimizer with 
            given parameters

            -- SimulationTimes:int :: Amount of simulations

            -- ArgsOptimizer :: Optimizer parameters as arguments

            -- DirResults:str :: Directory path where the results are saved

            -- NameResults:str :: File name of the results

            -- ThreadPool:int :: Number of thread/processors being use to execute simulations

            -- KwargsOptimizer :: Optimizer parameters as keywords arguments

            Returns the time taken for the execution of simulations
        """
        self.ArgsOptimizer = ArgsOptimizer
        self.KwargsOptimizer = KwargsOptimizer

        start_time = time()
        results_simulations = self.GetResults(SimulationTimes,ThreadPool)
        end_time = time()

        processed_results = self.ProcessResults(results_simulations)
        
        if DirResults != '.':
            self.NameResults = NameResults
            self.DirResults = DirResults
            self.SaveResults(processed_results)

            return end_time-start_time
    
        else:
            return processed_results[2]

    def GetResults(self,SimulationTimes:int,ThreadPool:int) -> list[np.ndarray]:
        """
            Method for getting raw results of 
            simulations

            -- SimulationTimes:int :: Amount of simulations

            -- ThreadPool:int :: Number of thread/processors being use to execute simulations

            Return a list of snapshots of the optimal 
            values in each simulation
        """
        optimize_function = lambda args_optimizer , kwargs_optimizer : self.Optimizer(*args_optimizer,**kwargs_optimizer)[1]

        with Parallel(n_jobs=ThreadPool) as pool_execution:
            results = pool_execution(delayed(optimize_function)(args_optimizer,kwargs_optimizer) 
                                     for args_optimizer , kwargs_optimizer in self.ParametersOptimizer(SimulationTimes))
        
        return results

    def ParametersOptimizer(self,SimulationTimes:int) -> Iterable[tuple[tuple[float],dict[str,float]]]:
        """
            Method for yielding arguments and 
            keyword arguments for the optimizer

            -- SimulationTimes:int :: Amount of simulations

            Return a tuple, where the first element 
            is the arguments and the second is the 
            keywords arguments
        """
        for _ in range(SimulationTimes):
            yield (self.ArgsOptimizer,self.KwargsOptimizer)

    def ProcessResults(self,ResultsSimulations:list[np.ndarray]) -> np.ndarray:
        """
            Method for processing results, obtaining 
            their minimum, quartiles and maximum

            -- ResultsSimulations:list[np.ndarray] :: List of snapshots of the optimal values in each simulation

            Return an array where the previous measurements are summarized
        """
        return np.quantile(ResultsSimulations,[0.0,0.25,0.5,0.75,1.0],axis=0)
    
    def SaveResults(self,ProcessedResults:np.ndarray) -> None:
        """
            Method for saving the summarized results 
            as csv file

            -- ProcessedResults:np.ndarray :: Summary of simulations measurements
        """
        os.makedirs(self.DirResults,exist_ok=True)
        np.savetxt(self.DirResults+self.NameResults,ProcessedResults,delimiter=',')