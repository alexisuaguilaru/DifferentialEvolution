from ..DifferentialEvolution.Base import DifferentialEvolution
from .FunctionsInitProblem import ObjectiveFunctionCEC , Individual

from .SimulationsCSV import SimulateOptimizer , ConvertResultsCSV

if __name__ == '__main__':
    yearCEC = '2017'
    dimension = 2
    numberSimulations = 200
    processes_jobs = 12

    lowerBound , upperBound = -100 , 100
    initializeIndividual = Individual(lowerBound,upperBound,dimension)

    optimizerName = 'Base'
    kwargs_DiffEvol = {
                        'FunctionEvaluations': 5000,
                        'PopulationSize': 100,
                        'ScalingFactor': 0.5,
                        'CrossoverRate': 0.5
                      }
    for functionNumber in ['1','2']:
        objectiveFunction = ObjectiveFunctionCEC(functionNumber,yearCEC,dimension)
        optimizerDiffEvol = DifferentialEvolution(objectiveFunction,initializeIndividual)

        simulationsFunctionEvaluations , simulationsOptimals = SimulateOptimizer(optimizerDiffEvol,kwargs_DiffEvol,numberSimulations,processes_jobs)

        ConvertResultsCSV(optimizerName,simulationsFunctionEvaluations,'F',functionNumber,dimension,yearCEC)
        ConvertResultsCSV(optimizerName,simulationsOptimals,'O',functionNumber,dimension,yearCEC)

        print(f'FINISH F{functionNumber}')