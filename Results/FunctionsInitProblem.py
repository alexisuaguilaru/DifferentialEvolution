import numpy as np
from opfunu import get_functions_by_classname

from typing import Callable

def ObjectiveFunctionCEC(FunctionNumber:str,YearCEC:str='2017',Dimension:int=2) -> Callable:
    """
        Function wrapper to return the CEC's function with the given parameters

        -- FunctionNumber:str :: Function's number
        
        -- YearCEC:str :: Function's year which belongs
        
        -- Dimension:int :: Function's dimensions
    """    
    function = get_functions_by_classname(f'F{FunctionNumber}{YearCEC}')[0](ndim=Dimension)
    
    def ObjectiveFunction_inner(SolutionVector:np.ndarray) -> float:
        """
            Inner function to evaluate CEC's function with the given parameters

            -- SolutionVector:np.ndarray :: Vector represents a solution
        """
        return function.evaluate(SolutionVector)
    
    return ObjectiveFunction_inner

def Individual(LowerBound:float=-100,UpperBound:float=100,Dimension:int=2) -> Callable:
    """
        Function to create a random individual
        
        -- LowerBound:float :: Minimum value to each individual's component
        
        -- UpperBound : Maximum value to each individual's component
        
        -- Dimension : Individual's dimension
    """
    def Individual_inner() -> np.ndarray:
        """
            Inner function to create a random individual with the given range
        """
        individual = np.random.default_rng().uniform(LowerBound,UpperBound,Dimension)
        return individual
    
    return Individual_inner