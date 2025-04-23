import json

from DifferentialEvolution import DifferentialEvolutionVariant , SummaryResults , CompressSummaryResults
from TestingFunctions import GetValidatedObjectiveFunction , ObjectiveFunctionCEC , Population

if __name__ == '__main__':
    with open('./Experiments/ConfigExperiments.json') as config_file:
        Configuration = json.load(config_file)
    
    YEAR_CEC = Configuration['YEAR_CEC']
    DIMENSION = Configuration['DIMENSION']
    DIR_RESULTS = "./Experiments/"

    NUMBER_SIMULATIONS = Configuration['NUMBER_SIMULATIONS']
    THREADS_JOBS = Configuration['THREADS_JOBS']

    lower_bound , upper_bound = Configuration['lower_bound'] , Configuration['upper_bound']
    initialize_population = Population(lower_bound,upper_bound,DIMENSION)

    validated_function_numbers = GetValidatedObjectiveFunction(YEAR_CEC,DIMENSION)

    kwargs_universal = {'FunctionEvaluations' : Configuration['FunctionEvaluations']}
    for population_size in Configuration['PopulationSize']:
        kwargs_universal['PopulationSize'] = population_size

        print("\nSTART SIMULATIONS/RUNS")
        for variant_name , kwargs_variant in Configuration['Variants'].items():
            print(f'START :: {variant_name}\n')

            variant_diff_evol = DifferentialEvolutionVariant(variant_name)

            dir_results = DIR_RESULTS + f'{variant_name}/'
            for function_number in validated_function_numbers:
                objective_function = ObjectiveFunctionCEC(function_number,YEAR_CEC,DIMENSION)
                simulation_results = SummaryResults(variant_diff_evol,objective_function,initialize_population)

                print(f'START F{function_number}')
                name_results = f'F{function_number}.csv'
                time_execution = simulation_results(NUMBER_SIMULATIONS,DirResults=dir_results,NameResults=name_results,ThreadPool=THREADS_JOBS,**kwargs_universal,**kwargs_variant)
                print(f'FINISH F{function_number}. {time_execution}sec\n')

            print(f'END :: {variant_name}\n')

        print("START COMPRESSION OF RESULTS\n")
        CompressSummaryResults(Dimension=DIMENSION,PopulationSize=population_size)