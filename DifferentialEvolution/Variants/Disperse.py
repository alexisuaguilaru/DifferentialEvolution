import numpy as np

from typing import Callable

from .Base import DifferentialEvolution

class DifferentialEvolution_Disperse(DifferentialEvolution):
    def __init__(
            self,
            ObjectiveFunction:Callable,
            InitializeIndividual:Callable,
        ):
        """
        Class for Differential Evolution Metaheuristic 
        with Population Dispersion (Augmentation) based 
        on Duplicate the Population and apply it Normal 
        Noise
        
        Parameters
        ----------
        ObjectiveFunction : Callable 
            Function being optimized 

        InitializeIndividual : Callable 
            Function to create individuals
        """
        super().__init__(ObjectiveFunction, InitializeIndividual)

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
            where a dispersion to population is applied

        Returns
        -------
        OptimalIndividual : np.ndarray
            Best solution that was founded

        Snapshots : list[float] 
            List of the optimal values at each function evaluation
        """
        self.ApplyDispersionEvaluations = set(np.ceil(FunctionEvaluations*percentage) for percentage in PercentageEvaluations)
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
            if numberFunctionEvaluation in self.ApplyDispersionEvaluations:
                self.ApplyPopulationDispersion()
                self.MutationOperation()
                self.CrossoverOperation()
                indexIndividual = numberFunctionEvaluation%self.PopulationSize
            
            elif (indexIndividual:=numberFunctionEvaluation%self.PopulationSize) == 0:
                self.MutationOperation()
                self.CrossoverOperation()
            
            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()
    
    def ApplyPopulationDispersion(
            self,
        ) -> None:
        """
        Method to apply a dispersion of the population based 
        on duplicate it and user normal noise
        """
        dispersedPopulation = self.Population.copy() + np.random.normal(size=self.Population.shape)
        fitnessValuesDispersedPopulation = np.apply_along_axis(self.ObjectiveFunction,1,dispersedPopulation)

        self.Population = np.concat([self.Population,dispersedPopulation],axis=0)
        self.FitnessValuesPopulation = np.concat([self.FitnessValuesPopulation,fitnessValuesDispersedPopulation],axis=0)

        self.PopulationSize *= 2
        self.PopulationIndexes = np.arange(self.PopulationSize)

        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()