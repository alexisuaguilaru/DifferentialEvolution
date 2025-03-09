import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

from typing import Callable

from .Reduction_Base import DifferentialEvolution_Reduction

class DifferentialEvolution_Agglomerative(DifferentialEvolution_Reduction):
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic with Population Reduction 
            based on Agglomerative [Clustering] Algorithm to get the Best 
            Representatives at each Cluster of Size Two
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def GetClustersRepresentatives(self) -> list[int]:
        population_fitness = np.concat([self.Population,self.FitnessValuesPopulation.reshape((self.PopulationSize,1))],axis=1)
        numberClusters = self.PopulationSize//2
        populationClusters = AgglomerativeClustering(n_clusters=numberClusters).fit_predict(population_fitness)

        individualsClusters = defaultdict(list)
        for individual , label in zip(np.arange(self.PopulationSize),populationClusters):
            individualsClusters[label].append(individual)

        indexRepresentative = []
        for individuals in individualsClusters.values():
            indexBest = np.argmin(self.FitnessValuesPopulation[individuals])
            indexRepresentative.append(individuals[indexBest])

        return indexRepresentative