###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras                                                          #
# karine.smiras@gmail.com                                                       #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import random

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#function that starts population with random values between -1, 1 for the weights
def initialize_population(population_size, n_vars):
    return np.random.uniform(-1, 1, (population_size, n_vars))

#alpha blend
def crossover(parents, crossover_rate, alpha=0.5):
    offspring = []
    population_size = len(parents)
    
    for i in range(0, population_size - 1, 2):  
        parent1, parent2 = parents[i], parents[i + 1] 
        
        if np.random.uniform(0, 1) < crossover_rate:
            child1, child2 = np.copy(parent1), np.copy(parent2)
            
            for j in range(len(parent1)):  
                d = abs(parent1[j] - parent2[j])
                low = min(parent1[j], parent2[j]) - alpha * d
                high = max(parent1[j], parent2[j]) + alpha * d
                
                child1[j] = np.random.uniform(low, high)
                child2[j] = np.random.uniform(low, high)
            
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)
    
    return np.array(offspring)


                    
#selects each time from a poole of 3 and then selects the solution with highest fitness
def tournament_selection(population, fitness, k):
    selected = []
    for i in range(len(population)):
      
        indices = random.sample(range(len(population)), k)

        best_individual = indices[np.argmax(fitness[indices])]
        selected.append(population[best_individual])
    return np.array(selected)
    
    
#mutation that uses standard cauchy distribution and mutates one chromosome 
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.uniform(0, 1) < mutation_rate:
            mutation_value = np.random.standard_cauchy() * 0.1
            individual[i] = individual[i] + mutation_value
            if individual[i] < -1: #make new individual is within the constraines
                individual[i] = -1
            elif individual[i] > 1:
                individual[i] = 1
    return individual



def main(experiment_name, opponent, test_train = 'train'):

    counter = 0
    n_hidden_neurons = 10

    if test_train == 'train':
        for i in range(10):
            counter += 1
           
            headless = True
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        
            experiment_name = experiment_name #voor nieuw experiment geef nieuwe naam en kijk naar nieuwe folder
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)
        
        
            
            run_mode = test_train #switch for testing and training
            
            if run_mode == "train": #training phase
                #
                #Dit is de train omgeving. Hier moeten we op een gegeven moment plotten for de optimalisatie
                #
                
                env = Environment(experiment_name=experiment_name,
                            enemies=opponent,
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False,
                            multiplemode="yes")
        
                # number of weights for multilayer with 10 hidden neurons
                n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        
                # start writing your own code from here
                generations = 100
                # na 100 generaties heb ik optimale fitness van 87.837. Op groepsapp was besproken dat beste solution rond 90 zou moeten zitten na ongeveer 300
                
                population_size = 100
                mutation_rate = 0.05
                crossover_rate = 0.2
                elitism_rate = 0.05
                alpha = 0.5
                nr_elites = int(population_size * elitism_rate)
                k = 3
        
                population = initialize_population(population_size=population_size, n_vars=n_vars)
                fitness = evaluate(env, population)
                global_best = np.max(fitness)
                global_best_index = np.argmax(fitness)
                global_best_individual = population[global_best_index]
                
                mean = np.mean(fitness)
                std = np.std(fitness)
                solutions = [population, fitness]
                env.update_solutions(solutions)
                
                
                for gen in range(generations):
                    
                    parents = tournament_selection(population=population, fitness=fitness, k=k) #selection
                    offspring_population = crossover(parents, crossover_rate, alpha) #crossover
                    
                    next_generation = np.array([mutate(individual=ind, mutation_rate=mutation_rate) for ind in offspring_population])
        
                    # Elitism houdt de beste 5% vast in de populatie. Is voor ons handig omdat onze crossover vrij gefocused is op diversiteit in de populatie
                    elite_indices = np.argsort(fitness)[-nr_elites:]  
                    elites = population[elite_indices]  
        
                    next_generation[:nr_elites] = elites
        
                    population = next_generation
                    
                    fitness = evaluate(env, population)
                    
                    best = np.max(fitness)
                    mean = np.mean(fitness)
                    std = np.std(fitness)
                    
                    if best > global_best:
                        global_best = best
                        global_best_index = np.argmax(fitness)
                        global_best_individual = population[global_best_index]
                        
                    
                    file_aux = open(experiment_name+'/results.txt', 'a')
                    file_aux.write('\n'+str(gen)+' '+str(round(best,6))+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(counter))
                    file_aux.close()
                    
                    np.savetxt(experiment_name+'/best.txt', global_best_individual) #gewichten van beste individu/solution
                    
                    solutions = [population, fitness]
                    env.update_solutions(solutions=solutions)
                    env.save_state()
            np.savetxt(experiment_name+f'/best{i}.txt', global_best_individual)

            
    if test_train == 'test':
       #
        #This is test environment where we load the individual with the highest fitness and test on the enemies
        #
        for i in range(10):
            headless = True
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"

            env = Environment(experiment_name=experiment_name,
                    enemies=opponent,
                    multiplemode='yes',
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini='yes')
            sol = np.loadtxt(experiment_name+f'/best{i}.txt')
            fitness = simulation(env, sol) #dit is de fitness van een spelletje met nieuwe enemies

            player_life = env.get_playerlife()
            enemy_life = env.get_enemylife()

            test_name = f'AB_{opponent}'
            if not os.path.exists(test_name):
                os.makedirs(test_name)

            file = open(test_name+'/results.txt', 'a')
            file.write(str(fitness)+' '+str(player_life)+' '+str(enemy_life)+'\n')
            file.close()

        

def simulate_wrapper(args):
    experiment_name, opponent = args
    return main(experiment_name, opponent)

experiment_names = [('GA_AB_G1',[1, 2, 5, 7]), ('GA_AB_G2',[3, 4, 6, 8])]

import multiprocessing as mp

test_train = 'test'

if test_train == 'train':
    if __name__ == '__main__':
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(simulate_wrapper, experiment_names)
            
if test_train == 'test':
    for tup in experiment_names:
        experiment_name, enemy = tup
        main(experiment_name, enemy, 'test')