import numpy as np
from typing import Callable

class AbstractDifferentialEvolution:
    def __init__(self,ObjectiveFunction:Callable,InitializePopulation:Callable):
        """
        Abstract Class for implementation for variants of 
        Differential Evolution Metaheuristic. Based on 
        DE/rand/1/bin using number of function evaluations instead 
        of iterations.
        
        Parameters
        ----------
        ObjectiveFunction : Callable 
            Function being optimized 

        InitializeIndividual : Callable 
            Function to create individuals
        """
        self.ObjectiveFunction = ObjectiveFunction
        self.InitializePopulation = InitializePopulation

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,ScalingFactor:float,CrossoverRate:float) -> tuple[np.ndarray,list[float]]:
        """
        Method for searching optimal solution for a give objective 
        function. Return the best optimal solution, because of 
        implementation will be the minimum.
            
        Parameters
        ----------
        FunctionEvaluations : int 
            Number of function evaluations

        PopulationSize : int 
            Parameter NP. Size of population of solutions

        ScalingFactor : float 
            Parameter F. Scaling factor for difference between vector 

        CrossoverRate : float 
            Parameter Cr. Crossover rate for crossover operation

        Returns
        -------
        OptimalIndividual : np.ndarray
            Best solution that was founded

        Snapshots : list[float] 
            List of the optimal values at each function evaluation
        """
        self.PopulationSize = PopulationSize
        self.ScalingFactor = ScalingFactor
        self.CrossoverRate = CrossoverRate
        
        self.InitializeOptimization()
        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()
        self.Dimension = self.OptimalIndividual.shape[0]

        self.Snapshots = []
        self.WriteSnapshot()

        self.FindOptimal(FunctionEvaluations)
        
        return self.OptimalIndividual , self.Snapshots

    def InitializeOptimization(self) -> None:
        """
        Method to initialize Population and 
        FitnessValuesPopulation attributes  
        """
        self.Population = self.InitializePopulation(self.PopulationSize)
        self.FitnessValuesPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.Population)

        self.PopulationIndexes = np.arange(self.PopulationSize)

    def BestOptimalIndividual(self) -> tuple[np.ndarray,float]:
        """
        Method for finding the best optimal individual up to the current generation 
            
        Returns
        -------
        BestOptimalSoluton : np.ndarray
            Current best optimal solution
        
        BestOptimalValue : float
            Current best optimal value
        """
        indexOptimalIndividual = np.argmin(self.FitnessValuesPopulation)
        return self.Population[indexOptimalIndividual] , self.FitnessValuesPopulation[indexOptimalIndividual]

    def WriteSnapshot(self) -> None:
        """
        Method to save a snapshot of the optimal/best 
        value at current iteration
        """
        self.Snapshots.append(self.OptimalValue)

    def FindOptimal(self,FunctionEvaluations:int) -> None:
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
                self.MutationOperation()
                self.CrossoverOperation()

            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()    

    def MutationOperation(self) -> None:
        """
        Method to apply Differential Evolution 
        Mutation Operation to the population
        """
        self.MutatedPopulation = self.RandomSampleSolutions()
        self.MutatedPopulation += self.ScalingFactor*(self.RandomSampleSolutions()-self.RandomSampleSolutions())

    def RandomSampleSolutions(self) -> np.ndarray:
        """
        Method to get a sample of random individuals 
        from the population of solutions

        Return
        ------
        RandomSamplint : np.ndarray
            A random sample of solutions from 
            the population of solutions
        """
        randomIndexes =  np.random.randint(self.PopulationSize,size=self.PopulationSize)
        return self.Population[randomIndexes]

    def CrossoverOperation(self) -> None:
        """
        Method to apply Differential Evolution 
        Crossover Operation to the population
        """
        crossoverThreshold = np.random.random((self.PopulationSize,self.Dimension)) <= self.CrossoverRate
        
        indexesMutated = np.random.randint(self.Dimension,size=self.PopulationSize)
        crossoverThreshold[self.PopulationIndexes,indexesMutated] = True
        
        self.CrossoverPopulation = self.Population.copy()
        self.CrossoverPopulation[crossoverThreshold] = self.MutatedPopulation[crossoverThreshold]

        self.FitnessCrossoverPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.CrossoverPopulation)

    def IterativeImproveIndividual(self,IndexIndividual:int) -> None:
        """
        Method to improve a given individual in the population

        Parameter
        ---------
        IndexIndividual:int :: 
            Index of the solution/individual being improved
        """
        fitness_crossovered = self.FitnessCrossoverPopulation[IndexIndividual]
        fitness_population = self.FitnessValuesPopulation[IndexIndividual]
        if fitness_crossovered <= fitness_population:
            crossovered_solution = self.CrossoverPopulation[IndexIndividual]
            self.Population[IndexIndividual] = crossovered_solution
            self.FitnessValuesPopulation[IndexIndividual] = fitness_crossovered
            
            if self.FitnessValuesPopulation[IndexIndividual] < self.OptimalValue:
                self.OptimalValue = fitness_crossovered
                self.OptimalIndividual = crossovered_solution