from .Base import DifferentialEvolution
from .RandomSample import DifferentialEvolution_FixedRandomSample , DifferentialEvolution_ProportionalRandomSample
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
    elif Variant == 'FixedRandomSample':
        return DifferentialEvolution_FixedRandomSample
    elif Variant ==  'ProportionalRandomSample':
        return DifferentialEvolution_ProportionalRandomSample
    elif Variant == 'Agglomerative':
        return DifferentialEvolution_Agglomerative
    elif Variant == 'RandomParameters':
        return DifferentialEvolution_RandomParameters