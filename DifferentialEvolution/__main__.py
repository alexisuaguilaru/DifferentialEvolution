from DifferentialEvolution import ResumeResults , DifferentialEvolutionVariant
import numpy as np

def func_obj(solution):
    return np.sum(solution**2)

dim = 2
def init_pop(size_pop): 
    return np.random.uniform(-10,10,size=(size_pop,dim))

if __name__ == '__main__':
    optimizer = DifferentialEvolutionVariant('Base')
    
    resumen_results = ResumeResults(optimizer,func_obj,init_pop)

    simulation = 1000
    FunctionEvaluations = 10000
    PopulationSize = 100
    ScalingFactor = 0.5
    CrossoverRate = 0.5
    time = resumen_results(simulation,FunctionEvaluations,PopulationSize,ScalingFactor,CrossoverRate,ThreadPool=16)

    print(f'{time=}')