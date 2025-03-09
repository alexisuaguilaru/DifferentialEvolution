import numpy as np

from typing import Callable

class DifferentialEvolution:
    def __init__(self,ObjectiveFunction:Callable,InitializePopulation:Callable):
        """
            Class for Differential Evolution Metaheuristic
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        self.ObjectiveFunction = ObjectiveFunction
        self.InitializePopulation = InitializePopulation

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,ScalingFactor:float,CrossoverRate:float) -> tuple[np.ndarray,list[float]]:
        """
            Method for searching optimal solution for objective function
            
            -- FunctionEvaluations:int :: Amount of function evaluations
            
            -- PopulationSize:int :: Parameter NP. Size of solutions population
            
            -- ScalingFactor:float :: Parameter F. Scaling factor for difference vector 
            
            -- CrossoverRate:float :: Parameter Cr. Crossover rate for crossover operation

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the optimal at each function evaluation
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
            Method to initialize Population and FitnessValuesPopulation attributes  
        """
        self.Population = self.InitializePopulation(self.PopulationSize)
        self.FitnessValuesPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.Population)

    def BestOptimalIndividual(self) -> tuple[np.ndarray,float]:
        """
            Method for finding the best optimal individual up to the current generation 
            
            Return the best optimal found's individual and function value
        """
        indexOptimalIndividual = np.argmin(self.FitnessValuesPopulation)
        return self.Population[indexOptimalIndividual] , self.FitnessValuesPopulation[indexOptimalIndividual]

    def WriteSnapshot(self) -> None:
        """
            Method to save a snapshot of the optimal solution at iteration
        """
        self.Snapshots.append(self.OptimalValue)

    def FindOptimal(self,FunctionEvaluations:int) -> None:
        """
            Method for finding the optimal solution for the objective function

            -- FunctionEvaluations:int :: Amount of function evaluations
        """
        for numberFunctionEvaluation in range(FunctionEvaluations):
            indexIndividual = numberFunctionEvaluation%self.PopulationSize

            if indexIndividual == 0:
                self.NextPopulation()

            self.IterativeImproveIndividual(indexIndividual)
            self.WriteSnapshot()
        
        del self.MutatedPopulation
        del self.CrossoverPopulation
        del self.FitnessCrossoverPopulation
        del self.Population
    
    def NextPopulation(self) -> None:
        """
            Method to generate a mutated, crossover population 
        """
        self.MutationOperation()
        self.CrossoverOperation()

    def MutationOperation(self) -> None:
        """
            Method to apply Differential Evolution 
            Mutation Operation to the population
        """
        self.MutatedPopulation = self.Population[np.random.randint(self.PopulationSize,size=self.PopulationSize)]
        self.MutatedPopulation += self.ScalingFactor*(self.Population[np.random.randint(self.PopulationSize,size=self.PopulationSize)]-self.Population[np.random.randint(self.PopulationSize,size=self.PopulationSize)])

    def CrossoverOperation(self) -> None:
        """
            Method to apply Differential Evolution 
            Crossover Operation to the population
        """
        crossoverThreshold = np.random.random((self.PopulationSize,self.Dimension)) <= self.CrossoverRate
        
        indexesMutated = np.random.randint(self.Dimension,size=self.PopulationSize)
        crossoverThreshold[np.arange(self.PopulationSize),indexesMutated] = True
        
        self.CrossoverPopulation = self.Population.copy()
        self.CrossoverPopulation[crossoverThreshold] = self.MutatedPopulation[crossoverThreshold]

        self.FitnessCrossoverPopulation = np.apply_along_axis(self.ObjectiveFunction,1,self.CrossoverPopulation)

    def IterativeImproveIndividual(self,IndexIndividual:int) -> None:
        """
            Method to improve a given individual in the population

            -- IndexIndividual:int :: Individual's index to improve
        """
        if self.FitnessCrossoverPopulation[IndexIndividual] <= self.FitnessValuesPopulation[IndexIndividual]:
            self.Population[IndexIndividual] = self.CrossoverPopulation[IndexIndividual]
            self.FitnessValuesPopulation[IndexIndividual] = self.FitnessCrossoverPopulation[IndexIndividual]
            
            if self.FitnessValuesPopulation[IndexIndividual] < self.OptimalValue:
                self.OptimalValue = self.FitnessValuesPopulation[IndexIndividual]
                self.OptimalIndividual = self.Population[IndexIndividual]