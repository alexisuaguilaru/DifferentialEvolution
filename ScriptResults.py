from multiprocess import Pool

from DifferentialEvolution.DispatchVariants import DifferentialEvolutionVariant

from Results.FunctionsInitProblem import GetValidatedObjectiveFunction , ObjectiveFunctionCEC , Individual
from Results.SimulationsCSV import SimulateOptimizer , ConvertResultsCSV , ConvertTimeExecutionCSV

if __name__ == '__main__':
    yearCEC = '2017'
    dimension = 2
    numberSimulations = 1000
    processes_jobs = 12

    lowerBound , upperBound = -100 , 100
    initializeIndividual = Individual(lowerBound,upperBound,dimension)

    variantNames = ['Base','RandomSample','Agglomerative','RandomParameters']
    kwargs_Base = {
                    'FunctionEvaluations': 5000,
                    'PopulationSize': 100,
                    'ScalingFactor': 0.5,
                    'CrossoverRate': 0.5
                  }
    kwargs_RandomSample = { 
        					'FunctionEvaluations': 5000,
                    		'PopulationSize': 100,
                    		'ScalingFactor': 0.5,
                    		'CrossoverRate': 0.5,
                            'PercentageEvaluations': [0.5]
    					  }
    kwargs_Agglomerative = {
        					 'FunctionEvaluations': 5000,
                    		 'PopulationSize': 100,
                    		 'ScalingFactor': 0.5,
                    		 'CrossoverRate': 0.5,
                             'PercentageEvaluations': [0.5]
    					   }
    kwargs_RandomParameters = {
                    			'FunctionEvaluations': 5000,
                    			'PopulationSize': 100,
                  			  }
    kwargs_VariantDiffEvol = {variantName:kwargs_Variant for variantName , kwargs_Variant in zip(variantNames,[kwargs_Base,kwargs_RandomSample,kwargs_Agglomerative,kwargs_RandomParameters])}

    validatedFunctionNumbers = GetValidatedObjectiveFunction(yearCEC,dimension)
	
    with Pool(processes_jobs) as poolExecutions:
        timeExecution_VariantFunctions = dict()
        for variantName in variantNames:
            print(f'START :: {variantName}\n')

            variantDiffEvol = DifferentialEvolutionVariant(variantName)

            timeExecution_Functions = []
            for functionNumber in validatedFunctionNumbers:
                objectiveFunction = ObjectiveFunctionCEC(functionNumber,yearCEC,dimension)
                optimizerDiffEvol = variantDiffEvol(objectiveFunction,initializeIndividual)

                print(f'START F{functionNumber}')
                simulationsFunctionEvaluations , simulationsOptimals , timeExecution = SimulateOptimizer(optimizerDiffEvol,kwargs_VariantDiffEvol[variantName],numberSimulations,poolExecutions)
                print(f'FINISH F{functionNumber}. {timeExecution}sec\n')

                timeExecution_Functions.append(timeExecution)

                ConvertResultsCSV(variantName,simulationsFunctionEvaluations,'F',functionNumber,dimension,yearCEC)
                ConvertResultsCSV(variantName,simulationsOptimals,'O',functionNumber,dimension,yearCEC)

            timeExecution_VariantFunctions[variantName] = timeExecution_Functions

            print(f'END :: {variantName}\n')
    
    ConvertTimeExecutionCSV(timeExecution_VariantFunctions,validatedFunctionNumbers,dimension,yearCEC)