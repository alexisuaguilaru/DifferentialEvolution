from math import ceil
import numpy as np

from typing import Callable

from .Base import DifferentialEvolution

class DifferentialEvolution_Reduction(DifferentialEvolution):
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic with Population Reduction 
            based on Clustering Algorithms
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,ScalingFactor:float,CrossoverRate:float,PercentageEvaluations:list[float]) -> tuple[np.ndarray,list[list]]:
        """
            Method for searching optimal solution for objective function
            
            -- FunctionEvaluations:int :: Amount of function evaluations
            
            -- PopulationSize:int :: Parameter NP. Size of solutions population
            
            -- ScalingFactor:float :: Parameter F. Scaling factor for difference vector 
            
            -- SrossoverRate:float :: Parameter Cr. Crossover rate for crossover operation

            -- PercentageEvaluations:list :: List of percentages of function evaluations 
            where a reduction to population is applied

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each generation
        """
        self.ApplyReductionEvaluations = set(ceil(FunctionEvaluations*percentage) for percentage in PercentageEvaluations)
        return super().__call__(FunctionEvaluations,PopulationSize,ScalingFactor,CrossoverRate)
    
    def FindOptimal(self,FunctionEvaluations:int) -> None:
        """
            Method for finding the optimal solution for the objective function

            -- FunctionEvaluations:int :: Amount of function evaluations
        """
        numberFunctionEvaluationInGeneration = 0
        for numberFunctionEvaluation in range(FunctionEvaluations):
            indexIndividual = numberFunctionEvaluation%self.PopulationSize
            self.IterativeImproveIndividual(indexIndividual)
            
            if (numberFunctionEvaluation+1) in self.ApplyReductionEvaluations:
                self.ApplyPopulationReduction()
                self.SnapshotPopulation(numberFunctionEvaluation+1)
                numberFunctionEvaluationInGeneration = -1

            elif (numberFunctionEvaluationInGeneration+1)%self.PopulationSize == 0:
                self.SnapshotPopulation(numberFunctionEvaluation+1)
                numberFunctionEvaluationInGeneration = -1

            numberFunctionEvaluationInGeneration += 1

        if numberFunctionEvaluationInGeneration != 0:
            self.SnapshotPopulation(numberFunctionEvaluation+1)
    
    def ApplyPopulationReduction(self):
        """
            Method to apply a reduction of the population based 
            on some clustering algorithm
        """
        clustersRepresentativeIndividuals = self.GetClustersRepresentatives()
        
        population = []
        fitnessValuesPopulation = []
        for individual , fitnessValue in clustersRepresentativeIndividuals:
            population.append(individual)
            fitnessValuesPopulation.append(fitnessValue)
        
        self.PopulationSize = len(population)
        self.Population = np.array(population)
        self.FitnessValuesPopulation = np.array(fitnessValuesPopulation)
        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()

    def GetClustersRepresentatives(self) -> list[list[np.ndarray,float]]:
        """
            Method to get the representative individuals 
            from each cluster using a given clustering algorithm 
            and a una policy of selection
        """
        pass