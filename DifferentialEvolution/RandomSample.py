import numpy as np
from sklearn.cluster import k_means
from math import floor , sqrt
from collections import defaultdict
from random import sample

from typing import Callable

from .Reduction_Base import DifferentialEvolution_Reduction

class DifferentialEvolution_RandomSample(DifferentialEvolution_Reduction):
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic with Population Reduction 
            based on K Means Algorithm with Random Sample of Representatives
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def GetClustersRepresentatives(self) -> list[list[np.ndarray,float]]:
        populationFitness = np.concat([self.Population,np.reshape(self.FitnessValuesPopulation,shape=(self.PopulationSize,1))],axis=1)
        numberClusters = floor(sqrt(self.PopulationSize))
        populationClusters = k_means(populationFitness,n_clusters=numberClusters)[1]

        individualsClusters = defaultdict(list)
        for cluster , individualFitness in zip(populationClusters,populationFitness):
            individual , fitnessValue = individualFitness[:-1] , individualFitness[-1]
            individualsClusters[cluster].append([individual,fitnessValue])

        clustersRepresentativeIndividuals = []
        for individuals in individualsClusters.values():
            if len(individuals) > 3:
                individuals = sample(individuals,3)
            clustersRepresentativeIndividuals.extend(individuals)

        return clustersRepresentativeIndividuals