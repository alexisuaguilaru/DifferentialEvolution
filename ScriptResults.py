from DifferentialEvolution.DispatchVariants import DifferentialEvolutionVariant

from Results.FunctionsInitProblem import ObjectiveFunctionCEC , Individual
from Results.SimulationsCSV import SimulateOptimizer , ConvertResultsCSV

if __name__ == '__main__':
    yearCEC = '2017'
    dimension = 2
    numberSimulations = 50
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
	# ADD save time execution for each variant,function
    for variantName in variantNames:
        print(f'START :: {variantName}\n')

        variantDiffEvol = DifferentialEvolutionVariant(variantName)
        
        for functionNumber in range(1,4):
            ## ADD try - except statement
            objectiveFunction = ObjectiveFunctionCEC(functionNumber,yearCEC,dimension)
            optimizerDiffEvol = variantDiffEvol(objectiveFunction,initializeIndividual)

            print(f'START F{functionNumber}')
            simulationsFunctionEvaluations , simulationsOptimals , timeExecution = SimulateOptimizer(optimizerDiffEvol,kwargs_VariantDiffEvol[variantName],numberSimulations,processes_jobs)
            print(f'FINISH F{functionNumber}. {timeExecution}sec\n')

            ConvertResultsCSV(variantName,simulationsFunctionEvaluations,'F',functionNumber,dimension,yearCEC)
            ConvertResultsCSV(variantName,simulationsOptimals,'O',functionNumber,dimension,yearCEC)

        print(f'END :: {variantName}\n')