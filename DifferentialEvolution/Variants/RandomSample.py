import numpy as np
from math import floor , ceil
from sklearn.cluster import k_means
from collections import defaultdict

from typing import Callable

from .Reduction_Base import DifferentialEvolution_Reduction

class DifferentialEvolution_FixedRandomSample(DifferentialEvolution_Reduction):
    def __init__(
            self,
            ObjectiveFunction:Callable,
            InitializeIndividual:Callable,
        ):
        """
        Class for Differential Evolution Metaheuristic with Population Reduction 
        based on K-Means Algorithm with Random Sample of Fixed Number of Representatives
            
        Parameters
        ----------
        ObjectiveFunction : Callable 
            Function being optimized 

        InitializeIndividual : Callable 
            Function to create individuals
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def __call__(
            self,
            FunctionEvaluations:int,
            PopulationSize:int,
            RangeScalingFactor:tuple[float,float]|float=[0,1],
            RangeCrossoverRate:tuple[float,float]|float=[0,1],
            PercentageEvaluations:list[float]=[0.5],
            SampledIndividuals:int=3
        ) -> tuple[np.ndarray,list[float]]:
        """
        Method for searching optimal solution for a give objective 
        function. Return the best optimal solution, because of 
        implementation will be the minimum.

        If both numbers in one of the ranges are equal, the parameter 
        is no random.

        Parameters
        ----------
        FunctionEvaluations : int 
            Number of function evaluations

        PopulationSize : int 
            Parameter NP. Size of population of solutions

        RangeScalingFactor : tuple[float,float] | float
            Range of values for Parameter F. Scaling factor 
            for difference between vector.

        RangeCrossoverRate : tuple[float,float] | float
            Range of values for Parameter Cr. Crossover rate 
            for crossover operation

        PercentageEvaluations : list[float]
            List of percentages of function evaluations 
            where a reduction to population is applied
        
        SampledIndividuals : int
            Number of sampled individuals in each cluster

        Returns
        -------
        OptimalIndividual : np.ndarray
            Best solution that was founded

        Snapshots : list[float] 
            List of the optimal values at each function evaluation
        """
        self.SampledIndividuals = SampledIndividuals
        return super().__call__(FunctionEvaluations,PopulationSize,RangeScalingFactor,RangeCrossoverRate,PercentageEvaluations)

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
    
class DifferentialEvolution_ProportionalRandomSample(DifferentialEvolution_Reduction):
    def __init__(
            self,
            ObjectiveFunction:Callable,
            InitializeIndividual:Callable,
        ):
        """
        Class for Differential Evolution Metaheuristic with Population Reduction 
        based on K Means Algorithm with Random Sample of Proportional Number of Representatives
            
        Parameters
        ----------
        ObjectiveFunction : Callable 
            Function being optimized 

        InitializeIndividual : Callable 
            Function to create individuals
        """
        super().__init__(ObjectiveFunction,InitializeIndividual)

    def __call__(
            self,
            FunctionEvaluations:int,
            PopulationSize:int,
            RangeScalingFactor:tuple[float,float]|float=[0,1],
            RangeCrossoverRate:tuple[float,float]|float=[0,1],
            PercentageEvaluations:list[float]=[0.5],
            ProportionIndividuals:float=1/2,
        ) -> tuple[np.ndarray,list[float]]:
        """
        Method for searching optimal solution for a give objective 
        function. Return the best optimal solution, because of 
        implementation will be the minimum.

        If both numbers in one of the ranges are equal, the parameter 
        is no random.

        Parameters
        ----------
        FunctionEvaluations : int 
            Number of function evaluations

        PopulationSize : int 
            Parameter NP. Size of population of solutions

        RangeScalingFactor : tuple[float,float] | float
            Range of values for Parameter F. Scaling factor 
            for difference between vector.

        RangeCrossoverRate : tuple[float,float] | float
            Range of values for Parameter Cr. Crossover rate 
            for crossover operation

        PercentageEvaluations : list[float]
            List of percentages of function evaluations 
            where a reduction to population is applied
        
        ProportionIndividuals : float
            Proportion of sampled individuals in each cluster

        Returns
        -------
        OptimalIndividual : np.ndarray
            Best solution that was founded

        Snapshots : list[float] 
            List of the optimal values at each function evaluation
        """
        self.ProportionIndividuals = ProportionIndividuals
        return super().__call__(FunctionEvaluations,PopulationSize,RangeScalingFactor,RangeCrossoverRate,PercentageEvaluations)

    def GetClustersRepresentatives(self) -> list[int]:
        population_fitness = np.concat([self.Population,self.FitnessValuesPopulation.reshape((self.PopulationSize,1))],axis=1)
        numberClusters = floor(np.sqrt(self.PopulationSize))
        populationLabels = k_means(population_fitness,n_clusters=numberClusters)[1]

        individualsClusters = defaultdict(list)
        for individual , label in zip(np.arange(self.PopulationSize),populationLabels):
            individualsClusters[label].append(individual)

        indexRepresentative = []
        for individuals in individualsClusters.values():
            individuals = np.random.choice(individuals,ceil(self.ProportionIndividuals*len(individuals)))
            indexRepresentative.extend(individuals)

        return indexRepresentative