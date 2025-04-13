import numpy as np

from DifferentialEvolution import DifferentialEvolutionVariant , ResumeResults

# Create or define some objective function 
def ObjectiveFunction(Solution:np.ndarray) -> float:
    return np.sum(Solution**2)

# Define function to init a population of solutions with (size,dimension)
dimension = 2
def InitPopulation(PopulationSize):
    return np.random.uniform(-10,10,(PopulationSize,dimension))

if __name__ == '__main__':

    # Get variant of differential evolution 
    variant_name = 'Base'
    optimizer = DifferentialEvolutionVariant(variant_name)


    # Init class to get results with optimizer (differential 
    # evolution), objective function and function to init population
    get_results = ResumeResults(optimizer,ObjectiveFunction,InitPopulation)


    # Define arguments for save simulations/results and threading
    simulation_times = 50
    dir_result = './'
    name_results = 'example_results.csv'
    threads = -1 # Because use of joblib, -1 is equal to use all available threads

    # Define args and kwargs to call optimizer in each simulation
    function_evaluations = 5000
    population_size = 100
    scaling_factor = 0.5
    crossover_rate = 0.5

    # Call function to process results and simulations
    time_execution = get_results(simulation_times,function_evaluations,population_size,scaling_factor,crossover_rate,DirResults=dir_result,NameResults=name_results,ThreadPool=threads)


    # Print time of execution
    print(f'{time_execution=}')