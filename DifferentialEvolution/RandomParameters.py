import numpy as np

from typing import Callable

from .Base import DifferentialEvolution

class DifferentialEvolution_RandomParameters(DifferentialEvolution):
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic where 
            their Parameters (scaling factor and crossover rate)
            Randomly change each Generation
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,RangeScalingFactor:list[float,float]=[0,1],RangeCrossoverRate:list[float,float]=[0,1]) -> tuple[np.ndarray,list[list]]:
        """
            Method for searching optimal solution for objective function
            
            -- FunctionEvaluations:int :: Amount of function evaluations
            
            -- PopulationSize:int :: Parameter NP. Size of solutions population

            -- ScalingFactor:list :: Range of value for Parameter F [Scaling factor for difference vector]
            
            -- CrossoverRate:list :: Range of value for Parameter Cr [Crossover rate for crossover operation]

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each generation
        """
        self.RangeScalingFactor = RangeScalingFactor
        self.RangeCrossoverRate = RangeCrossoverRate

        return super().__call__(FunctionEvaluations,PopulationSize,-1,-1)
    
    def FindOptimal(self,FunctionEvaluations:int) -> None:
        """
            Method for finding the optimal solution for the objective function

            -- FunctionEvaluations:int :: Amount of function evaluations
        """
        for numberFunctionEvaluation in range(FunctionEvaluations):
            indexIndividual = numberFunctionEvaluation%self.PopulationSize

            if indexIndividual == 0:
                self.RandomizeParameters()
                self.NextPopulation()

            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()

    def RandomizeParameters(self) -> None:
        """
            Method for randomizing scaling and crossover parameters
        """
        self.ScalingFactor = np.random.uniform(*self.RangeScalingFactor)
        self.CrossoverRate = np.random.uniform(*self.RangeCrossoverRate)