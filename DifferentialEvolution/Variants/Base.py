import numpy as np

from .Abstract import AbstractDifferentialEvolution

from typing import Callable

class DifferentialEvolution(AbstractDifferentialEvolution):
    def __init__(
            self,
            ObjectiveFunction:Callable,
            InitializeIndividual:Callable,
        ):
        """
        Class for Differential Evolution Metaheuristic. Based on 
        DE/rand/1/bin using number of function evaluations instead 
        of iterations.

        Where its parameters (scaling factor and crossover rate)
        are randoms in each generation optionally
            
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
        """
        self.RangeScalingFactor = self.ValidateRangeParameter(RangeScalingFactor)
        self.RangeCrossoverRate = self.ValidateRangeParameter(RangeCrossoverRate)

        return super().__call__(FunctionEvaluations,PopulationSize,-1,-1)
    
    def ValidateRangeParameter(
            self,
            RangeParameter:tuple[float,float]|float,
        ) -> tuple[float,float]:
        """
        Method for validate a range of values for a 
        parameter. If it is a float, returns a 
        validated range.

        Parameter
        --------
        RangeParameter : tuple[float,float] | float
            Range of values being validated. It can 
            be a tuple of values or a float

        Return
        ------
        RangeParameter : tuple[float,float]
            A validated range of values
        """
        if type(RangeParameter) == float:
            RangeParameter = (RangeParameter,RangeParameter)
        return RangeParameter

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
            indexIndividual = numberFunctionEvaluation%self.PopulationSize

            if indexIndividual == 0:
                self.RandomizeParameters()
                self.MutationOperation()
                self.CrossoverOperation()

            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()

    def RandomizeParameters(
            self,
        ) -> None:
        """
        Method for randomizing scaling and crossover parameters
        """
        self.ScalingFactor = np.random.uniform(*self.RangeScalingFactor)
        self.CrossoverRate = np.random.uniform(*self.RangeCrossoverRate)