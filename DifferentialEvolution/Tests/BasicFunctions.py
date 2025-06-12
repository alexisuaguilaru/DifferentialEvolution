"""
Basic test functions for evaluate 
convergence and functionality of 
variants of Differential Evolution
"""

import numpy as np

def F1(Solution:np.ndarray) -> float:
    return np.sum(Solution*Solution+100)

def F2(Solution:np.ndarray) -> float:
    return np.exp((np.sum(np.abs(Solution)))%10)

def F3(Solution:np.ndarray) -> float:
    return np.tanh(np.prod(Solution/100))

def Individuals(PopulationSize:int) -> np.ndarray:
    individual = np.random.uniform(-10,10,size=(PopulationSize,10))
    return individual