import numpy as np

from DifferentialEvolution import DifferentialEvolutionVariant , SummaryResults
from TestingFunctions import Population , ObjectiveFunctionCEC

def IterationsOptimos(DiffEvoVariant:str,FunctionNumer:int,Dimension:int,**Parameters) -> np.ndarray:
    diff_evol_class = DifferentialEvolutionVariant(DiffEvoVariant)
    lower_bound = -100
    upper_bound = 100
    initialize_population = Population(lower_bound,upper_bound,Dimension)
    objective_function = ObjectiveFunctionCEC(FunctionNumer,Dimension=Dimension)

    summary_results = SummaryResults(diff_evol_class,objective_function,initialize_population)
    dir_result = '.'
    number_simulation = 8
    threads = 6
    results_medians = summary_results(number_simulation,DirResults=dir_result,ThreadPool=threads,**Parameters)

    return np.concat([[np.arange(results_medians.shape[0])],[results_medians]],axis=0)