from .Base import DifferentialEvolution
from .RandomSample import DifferentialEvolution_RandomSample
from .Agglomerative import DifferentialEvolution_Agglomerative
from .RandomParameters import DifferentialEvolution_RandomParameters

def DifferentialEvolutionVariant(Variant:str):
    """
        Function to select between different variants of 
        Differential Evolution based on a string

        -- Variant:str :: Variant's name

        Return variant's class
    """
    if Variant == 'Base':
        return DifferentialEvolution
    elif Variant == 'RandomSample':
        return DifferentialEvolution_RandomSample
    elif Variant == 'Agglomerative':
        return DifferentialEvolution_Agglomerative
    elif Variant == 'RandomParameters':
        return DifferentialEvolution_RandomParameters