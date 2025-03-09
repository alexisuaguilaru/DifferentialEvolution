import numpy as np
from math import floor
from sklearn.cluster import k_means
from collections import defaultdict
from random import sample

from typing import Callable

from .Reduction_Base import DifferentialEvolution_Reduction

class DifferentialEvolution_FixedRandomSample(DifferentialEvolution_Reduction):
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic with Population Reduction 
            based on K Means Algorithm with Random Sample of Representatives
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,ScalingFactor:float,CrossoverRate:float,PercentageEvaluations:list[float],SampledIndividuals:int=3) -> tuple[np.ndarray,list[float]]:
        """
            Method for searching optimal solution for objective function
            
            -- FunctionEvaluations:int :: Amount of function evaluations
            
            -- PopulationSize:int :: Parameter NP. Size of solutions population
            
            -- ScalingFactor:float :: Parameter F. Scaling factor for difference vector 
            
            -- CrossoverRate:float :: Parameter Cr. Crossover rate for crossover operation

            -- PercentageEvaluations:list :: List of percentages of function evaluations 
            where a reduction to population is applied

            -- SampledIndividuals:int :: Number of sampled individuals in each cluster

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each generation
        """
        self.SampledIndividuals = SampledIndividuals
        return super().__call__(FunctionEvaluations,PopulationSize,ScalingFactor,CrossoverRate,PercentageEvaluations)

    def GetClustersRepresentatives(self) -> list[int]:
        population_fitness = np.concat([self.Population,self.FitnessValuesPopulation.reshape((self.PopulationSize,1))],axis=1)
        numberClusters = floor(np.sqrt(self.PopulationSize))
        populationLabels = k_means(population_fitness,n_clusters=numberClusters)[1]

        individualsClusters = defaultdict(list)
        for individual , label in zip(np.arange(self.PopulationSize),populationLabels):
            individualsClusters[label].append(individual)

        indexRepresentative = []
        for individuals in individualsClusters.values():
            if len(individuals) > self.SampledIndividuals:
                individuals = np.random.choice(individuals,self.SampledIndividuals)
            indexRepresentative.extend(individuals)

        return indexRepresentative