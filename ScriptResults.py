from DifferentialEvolution import DifferentialEvolutionVariant , ResumeResults
from TestingFunctions import GetValidatedObjectiveFunction , ObjectiveFunctionCEC , Population

if __name__ == '__main__':
    YEAR_CEC = '2017'
    DIMENSION = 2
    DIR_RESULTS = './test_results/'

    NUMBER_SIMULATIONS = 1000
    THREADS_JOBS = -1

    lower_bound , upper_bound = -100 , 100
    initialize_population = Population(lower_bound,upper_bound,DIMENSION)

    VariantNames = ['Base','FixedRandomSample','ProportionalRandomSample','Agglomerative','RandomParameters']
    kwargs_universal = {
                        'FunctionEvaluations': 5000,
                        'PopulationSize': 100,
                     }
    kwargs_Base = {
                    'ScalingFactor': 0.5,
                    'CrossoverRate': 0.5
                  }
    kwargs_FixedRandomSample = { 
                    		     'ScalingFactor': 0.5,
                    		     'CrossoverRate': 0.5,
                                 'PercentageEvaluations': [0.5]
    					       }
    kwargs_ProportionalRandomSample = { 
                    		            'ScalingFactor': 0.5,
                    		            'CrossoverRate': 0.5,
                                        'PercentageEvaluations': [0.5]
    					              }
    kwargs_Agglomerative = {
                    		 'ScalingFactor': 0.5,
                    		 'CrossoverRate': 0.5,
                             'PercentageEvaluations': [0.25,0.5,0.75]
    					   }
    kwargs_RandomParameters = {
                                'RangeScalingFactor': [0.4,0.6],
                                'RangeCrossoverRate': [0.4,0.6]
                  			  }
    
    validated_function_numbers = GetValidatedObjectiveFunction(YEAR_CEC,DIMENSION)

    print("\nSTART SIMULATIONS/RUNS")
    for variant_name in VariantNames:
        print(f'START :: {variant_name}\n')

        kwargs_variant = globals()[f'kwargs_{variant_name}']
        variant_diff_evol = DifferentialEvolutionVariant(variant_name)

        dir_results = DIR_RESULTS + f'{variant_name}/'
        for function_number in validated_function_numbers[:2]:
            objective_function = ObjectiveFunctionCEC(function_number,YEAR_CEC,DIMENSION)
            simulation_results = ResumeResults(variant_diff_evol,objective_function,initialize_population)

            print(f'START F{function_number}')
            name_results = f'F{function_number}.csv'
            time_execution = simulation_results(NUMBER_SIMULATIONS,DirResults=dir_results,NameResults=name_results,ThreadPool=THREADS_JOBS,**kwargs_universal,**kwargs_variant)
            print(f'FINISH F{function_number}. {time_execution}sec\n')
        
        print(f'END :: {variant_name}\n')