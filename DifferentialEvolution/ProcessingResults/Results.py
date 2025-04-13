from joblib import Parallel , delayed
import numpy as np

from time import time
import os

from typing import Iterable

class ResumeResults:
    def __init__(self,Optimizer,*ArgsInitOptimizer,**KwargsInitOptimizer):
        self.Optimizer = Optimizer(*ArgsInitOptimizer,**KwargsInitOptimizer)

    def __call__(self,SimulationTimes:int,*ArgsOptimizer,DirResults:str='./',NameResults:str='results.csv',ThreadPool:int=1,**KwargsOptimizer):
        self.ArgsOptimizer = ArgsOptimizer
        self.KwargsOptimizer = KwargsOptimizer

        start_time = time()
        results_simulations = self.GetResults(SimulationTimes,ThreadPool)
        end_time = time()

        processed_results = self.ProcessResults(results_simulations)
        print((processed_results).shape)
        print(processed_results[:,-1])
        
        self.DirResults = DirResults
        self.NameResults = NameResults
        self.SaveResults(processed_results)

        return end_time-start_time

    def GetResults(self,SimulationTimes,ThreadPool) -> np.ndarray:
        optimize_function = lambda args_optimizer , kwargs_optimizer : self.Optimizer(*args_optimizer,**kwargs_optimizer)[1]
        with Parallel(n_jobs=ThreadPool) as pool_execution:
            results = pool_execution(delayed(optimize_function)(args_optimizer,kwargs_optimizer) 
                                     for args_optimizer , kwargs_optimizer in self.ParametersOptimizer(SimulationTimes))
        return results

    def ParametersOptimizer(self,SimulationTimes:int) -> Iterable[tuple[tuple[float],dict[str,float]]]:
        for _ in range(SimulationTimes):
            yield (self.ArgsOptimizer,self.KwargsOptimizer)

    def ProcessResults(self,ResultsSimulations:list[np.ndarray]) -> np.ndarray:
        return np.quantile(ResultsSimulations,[0.0,0.25,0.5,0.75,1.0],axis=0)
    
    def SaveResults(self,ProcessedResults:np.ndarray) -> None:
        os.makedirs(self.DirResults,exist_ok=True)
        np.savetxt(self.DirResults+self.NameResults,ProcessedResults,delimiter=',')