from multiprocess import Pool

from DifferentialEvolution.DispatchVariants import DifferentialEvolutionVariant

from Results.FunctionsInitProblem import GetValidatedObjectiveFunction , ObjectiveFunctionCEC , Population
from Results.SimulationsCSV import SimulateOptimizer , ConvertResultsCSV , ConvertTimeExecutionCSV

if __name__ == '__main__':
    yearCEC = '2017'
    dimension = 2
    numberSimulations = 1000
    processes_jobs = 16

    lowerBound , upperBound = -100 , 100
    initializePopulation = Population(lowerBound,upperBound,dimension)

    variantNames = ['Base','FixedRandomSample','ProportionalRandomSample','Agglomerative','RandomParameters']
    kwargs_Base = {
                    'FunctionEvaluations': 5000,
                    'PopulationSize': 100,
                    'ScalingFactor': 0.5,
                    'CrossoverRate': 0.5
                  }
    kwargs_FixedRandomSample = { 
        					     'FunctionEvaluations': 5000,
                    		     'PopulationSize': 100,
                    		     'ScalingFactor': 0.5,
                    		     'CrossoverRate': 0.5,
                                 'PercentageEvaluations': [0.5]
    					       }
    kwargs_ProportionalRandomSample = { 
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
                             'PercentageEvaluations': [0.25,0.5,0.75]
    					   }
    kwargs_RandomParameters = {
                    			'FunctionEvaluations': 5000,
                    			'PopulationSize': 100,
                                'RangeScalingFactor': [0.4,0.6],
                                'RangeCrossoverRate': [0.4,0.6]
                  			  }
    kwargs_VariantDiffEvol = {variantName:kwargs_Variant for variantName , kwargs_Variant in zip(variantNames,[kwargs_Base,kwargs_FixedRandomSample,kwargs_ProportionalRandomSample,kwargs_Agglomerative,kwargs_RandomParameters])}

    validatedFunctionNumbers = GetValidatedObjectiveFunction(yearCEC,dimension)
	
    print("\nSTART SIMULATIONS/RUNS")
    timeExecution_VariantFunctions = dict()
    for variantName in variantNames:
        print(f'START :: {variantName}\n')
        with Pool(processes_jobs) as poolExecutions:

            variantDiffEvol = DifferentialEvolutionVariant(variantName)

            timeExecution_Functions = []
            for functionNumber in validatedFunctionNumbers:
                objectiveFunction = ObjectiveFunctionCEC(functionNumber,yearCEC,dimension)
                optimizerDiffEvol = variantDiffEvol(objectiveFunction,initializePopulation)

                print(f'START F{functionNumber}')
                simulationsResultsOptimal , timeExecution = SimulateOptimizer(optimizerDiffEvol,kwargs_VariantDiffEvol[variantName],numberSimulations,poolExecutions)
                print(f'FINISH F{functionNumber}. {timeExecution}sec\n')

                timeExecution_Functions.append(timeExecution)

                ConvertResultsCSV(variantName,simulationsResultsOptimal,functionNumber,dimension,yearCEC)

            timeExecution_VariantFunctions[variantName] = timeExecution_Functions

        print(f'END :: {variantName}\n')
    
    ConvertTimeExecutionCSV(timeExecution_VariantFunctions,validatedFunctionNumbers,dimension,yearCEC)