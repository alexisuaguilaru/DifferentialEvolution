import numpy as np

from typing import Callable

from .Base import DifferentialEvolution

class DifferentialEvolution_Reduction(DifferentialEvolution):
    def __init__(
            self,
            ObjectiveFunction:Callable,
            InitializeIndividual:Callable,
        ):
        """
        Class for Differential Evolution Metaheuristic 
        with Population Reduction based on Clustering Algorithms

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
        """
        self.ApplyReductionEvaluations = set(np.ceil(FunctionEvaluations*percentage) for percentage in PercentageEvaluations)
        return super().__call__(FunctionEvaluations,PopulationSize,RangeScalingFactor,RangeCrossoverRate)
    
    def FindOptimal(
            self,
            FunctionEvaluations:int,
        ) -> None:
        """
        Method for finding the optimal solution 
        for the objective function

        Parameter
        ---------
        FunctionEvaluations : int
            Number of function evaluations
        """
        for numberFunctionEvaluation in range(FunctionEvaluations):
            if (indexIndividual:=numberFunctionEvaluation%self.PopulationSize) == 0:
                self.MutationOperation()
                self.CrossoverOperation()

            if numberFunctionEvaluation in self.ApplyReductionEvaluations:
                self.ApplyPopulationReduction()
                indexIndividual = numberFunctionEvaluation%self.PopulationSize
            
            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()
    
    def ApplyPopulationReduction(
            self,
        ) -> None:
        """
            Method to apply a reduction of the population based 
            on some clustering algorithm
        """
        indexRepresentative = self.GetClustersRepresentatives()
        
        self.PopulationSize = len(indexRepresentative)
        self.PopulationIndexes = np.arange(self.PopulationSize)
        
        self.Population = self.Population[indexRepresentative]
        self.FitnessValuesPopulation = self.FitnessValuesPopulation[indexRepresentative]

        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()

    def GetClustersRepresentatives(
            self,
        ) -> list[int]:
        """
        Method to get the representative individuals 
        from each cluster using a given clustering algorithm 
        and a policy of selection

        Return
        ------
        IndexRepresentative : list[int]
            List of index of representative solutions 
            in each cluster
        """
        pass