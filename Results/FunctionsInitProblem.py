import numpy as np
from opfunu import get_functions_by_classname

from typing import Callable

def GetValidatedObjectiveFunction(YearCEC:str='2017',Dimension:int=2) -> list[int]:
    """
        Function to get a list of validated function numbers with the criteria
        
        -- YearCEC:str :: Function's year which belongs
        
        -- Dimension:int :: Function's dimensions

        Return a list with the function's number which exist with the given query
    """
    objectiveFunctionNumber = []
    for functionNumber in range(1,30):
        try:
            get_functions_by_classname(f'F{functionNumber}{YearCEC}')[0](ndim=Dimension)
            objectiveFunctionNumber.append(functionNumber)
        except:
            continue

    return objectiveFunctionNumber

def ObjectiveFunctionCEC(FunctionNumber:str,YearCEC:str='2017',Dimension:int=2) -> Callable:
    """
        Function wrapper to return the CEC's function with the given parameters

        -- FunctionNumber:str :: Function's number
        
        -- YearCEC:str :: Function's year which belongs
        
        -- Dimension:int :: Function's dimensions

        Return objective function with the given query
    """    
    function = get_functions_by_classname(f'F{FunctionNumber}{YearCEC}')[0](ndim=Dimension)
    
    def ObjectiveFunction_inner(SolutionVector:np.ndarray) -> float:
        """
            Inner function to evaluate CEC's function with the given parameters

            -- SolutionVector:np.ndarray :: Vector represents a solution

            Return objective function evaluate's value
        """
        return function.evaluate(SolutionVector)
    
    return ObjectiveFunction_inner

def Individual(LowerBound:float=-100,UpperBound:float=100,Dimension:int=2) -> Callable:
    """
        Function to create a random individual
        
        -- LowerBound:float :: Minimum value to each individual's component
        
        -- UpperBound : Maximum value to each individual's component
        
        -- Dimension : Individual's dimension

        Return a function to create a new random individual
    """
    def Individual_inner() -> np.ndarray:
        """
            Inner function to create a random individual with the given range

            Return a new random individual
        """
        individual = np.random.default_rng().uniform(LowerBound,UpperBound,Dimension)
        return individual
    
    return Individual_inner