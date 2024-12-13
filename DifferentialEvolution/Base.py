import numpy as np
from random import sample , random , randrange
from copy import deepcopy

from typing import Callable

class DifferentialEvolution:
    def __init__(self,ObjectiveFunction:Callable,InitializeIndividual:Callable):
        """
            Class for Differential Evolution Metaheuristic
            
            -- ObjectiveFunction:Callable :: Function being optimized 

            -- InitializeIndividual:Callable :: Function to create individuals

            Based on DE/rand/1/bin using number of function evaluations instead of iterations
        """
        self.ObjectiveFunction = ObjectiveFunction
        self.InitializeIndividual = InitializeIndividual

    def __call__(self,FunctionEvaluations:int,PopulationSize:int,ScalingFactor:float,CrossoverRate:float) -> tuple[np.ndarray,list[list]]:
        """
            Method for searching optimal solution for objective function
            
            -- FunctionEvaluations:int :: Amount of function evaluations
            
            -- PopulationSize:int :: Parameter NP. Size of solutions population
            
            -- ScalingFactor:float :: Parameter F. Scaling factor for difference vector 
            
            -- CrossoverRate:float :: Parameter Cr. Crossover rate for crossover operation

            Return the best optimal solution, because of implementation will be the minimum, and snapshots of the population at each generation
        """
        self.PopulationSize = PopulationSize
        self.ScalingFactor = ScalingFactor
        self.CrossoverRate = CrossoverRate
        self.InitializePopulation()
        self.OptimalIndividual , self.OptimalValue = self.BestOptimalIndividual()
        
        self.Generation = -1
        self.Snapshots = []
        self.SnapshotPopulation(0)

        self.FindOptimal(FunctionEvaluations)
        
        return self.OptimalIndividual , self.Snapshots

    def InitializePopulation(self) -> None:
        """
            Method to initialize Population and FitnessValuesPopulation attributes  
        """
        self.Population = np.array([self.InitializeIndividual() for _ in range(self.PopulationSize)])
        self.FitnessValuesPopulation = np.array([self.ObjectiveFunction(individual) for individual in self.Population])

    def BestOptimalIndividual(self) -> tuple[np.ndarray,float]:
        """
            Method for finding the best optimal individual up to the current generation 
            
            Return the best optimal found's individual and function value
        """
        indexOptimalIndividual = 0
        for indexIndividual in range(1,self.PopulationSize):
            if self.FitnessValuesPopulation[indexIndividual] < self.FitnessValuesPopulation[indexOptimalIndividual]:
                indexOptimalIndividual = indexIndividual
        
        return self.Population[indexOptimalIndividual] , self.FitnessValuesPopulation[indexOptimalIndividual]

    def SnapshotPopulation(self,NumberFunctionEvaluations_Generation:int) -> None:
        """
            Method to save a snapshot of the population at generation-st

            -- NumberFunctionEvaluations_Generation:int :: Number of function 
            evaluations at generation-st 
        """
        self.Generation += 1
        self.Snapshots.append((self.Generation,NumberFunctionEvaluations_Generation,self.OptimalValue,self.OptimalIndividual,deepcopy(self.Population)))

    def FindOptimal(self,FunctionEvaluations:int) -> None:
        """
            Method for finding the optimal solution for the objective function

            -- FunctionEvaluations:int :: Amount of function evaluations
        """
        for numberFunctionEvaluation in range(FunctionEvaluations):
            indexIndividual = numberFunctionEvaluation%self.PopulationSize
            self.IterativeImproveIndividual(indexIndividual)

            if (numberFunctionEvaluation+1)%self.PopulationSize == 0:
                self.SnapshotPopulation(numberFunctionEvaluation+1)
    
    def IterativeImproveIndividual(self,indexIndividual:int) -> None:
        """
            Method to improve a given individual in the population

            -- indexIndividual:int :: Individual's index to improve
        """
        mutatedIndividual = self.MutationOperation()
        crossoverIndividual = self.CrossoverOperation(indexIndividual,mutatedIndividual)
        
        if (fitnessValue:=self.ObjectiveFunction(crossoverIndividual)) <= self.FitnessValuesPopulation[indexIndividual]:
            self.Population[indexIndividual] = crossoverIndividual
            self.FitnessValuesPopulation[indexIndividual] = fitnessValue
            if self.FitnessValuesPopulation[indexIndividual] < self.OptimalValue:
                self.OptimalValue = self.FitnessValuesPopulation[indexIndividual]
                self.OptimalIndividual = self.Population[indexIndividual]

    def MutationOperation(self) -> np.ndarray:
        """
            Method to apply Differential Evolution Mutation Operation to a random individual

            Return the mutated individual
        """
        randomIndex_1 , randomIndex_2 , randomIndex_3 = sample(range(self.PopulationSize),k=3)
        randomIndividual_1 , randomIndividual_2 , randomIndividual_3 = self.Population[randomIndex_1] , self.Population[randomIndex_2] , self.Population[randomIndex_3]

        return  randomIndividual_1 + self.ScalingFactor*(randomIndividual_2 - randomIndividual_3)

    def CrossoverOperation(self,indexIndividual:int,mutatedIndividual:np.ndarray) -> np.ndarray:
        """
            Method to apply Differential Evolution Crossover Operation
            
            -- indexIndividual:int :: Individual's index to be crossover
            
            -- mutantIndividual:np.ndarray :: Individual to be crossover

            Return the crossover individual product of base and mutated individuals 
        """
        baseIndividual = self.Population[indexIndividual]
        crossoverIndividual = deepcopy(baseIndividual)

        indexMutated = randrange(0,len(crossoverIndividual))
        for indexComponent , componentMutatedIndividual in enumerate(mutatedIndividual):
            if (random() <= self.CrossoverRate) or (indexMutated == indexComponent):
                crossoverIndividual[indexComponent] = componentMutatedIndividual
        
        return crossoverIndividual