from .Base import DifferentialEvolution
from .RandomSample import DifferentialEvolution_FixedRandomSample , DifferentialEvolution_ProportionalRandomSample
from .Agglomerative import DifferentialEvolution_Agglomerative

def DifferentialEvolutionVariant(Variant:str):
    """
    Function to select between different variants of 
    Differential Evolution based on a string

    Parameter
    ---------
    Variant : str
        Name of the variant which being gotten

    Return
    ------
    Variant_Class : DifferentialEvolution
        Class of the variant which is requested        
    """
    return __DifferentialEvolutionVariant[Variant]
    
__DifferentialEvolutionVariant = {
    'Base' : DifferentialEvolution, 
    'Agglomerative' : DifferentialEvolution_Agglomerative,
    'FixedRandomSample' : DifferentialEvolution_FixedRandomSample, 
    'ProportionalRandomSample' : DifferentialEvolution_ProportionalRandomSample, 
}