import sys
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import random

def initialize(population_size, n_vars): # initializes population
    return (np.random.uniform(-1, 1, (population_size, n_vars)))

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def evaluate(env,x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def mutation(population, a, F=0.8):
    b, c = random.sample(list(population), k=2)
    
    while np.array_equal(b, c):
        b, c = random.sample(list(population), k=2)
    
    a = np.array(a)  # base vector
    b = np.array(b)  # differential vector
    c = np.array(c)
    
    new_individual = a + F * (b - c)

    return np.array(new_individual)


def crossover(old_individual, new_individual):
    crossover_child = []
    for i in range(len(old_individual)):
        crossover_child.append(random.choice([old_individual[i],new_individual[i]]))
    return np.array(crossover_child)

def selection(env, old_individual, new_child):
    old_fitness = simulation(env, old_individual)
    new_fitness = simulation(env, new_child)

    if new_fitness > old_fitness:
        return new_child
    else:
        return old_individual

def main(experiment_name,opponent_group,test_train='train'):
    
    n_hidden_neurons = 10
    counter = 0
    
    if test_train == 'train':
        for i in range(10):
            counter += 1
            # choose this for not using visuals and thus making experiments faster
            headless = True
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        
            experiment_name = experiment_name
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)
        
            
        
            # initializes simulation in individual evolution mode, for single static enemy.
    
            run_mode = test_train
            
            if run_mode == "train":
                env = Environment(experiment_name=experiment_name,
                    enemies=opponent_group,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini='yes',
                    multiplemode='yes')
        
                # number of weights for multilayer with 10 hidden neurons
                n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
        
                population_size = 100
                generations = 100
                
                population = initialize(population_size, n_vars)
                fitness = evaluate(env, population)

                global_best = np.max(fitness)
                global_best_index = np.argmax(fitness)
                global_best_individual = population[global_best_index]
                best = global_best_individual

                mean = np.mean(fitness)
                std = np.std(fitness)
                solutions = [population, fitness]
                env.update_solutions(solutions)
                
                for gen in range(generations):
                    print('Gen: ', gen)
                    # get new population
                    new_population = []
                    for j in range(population_size):
                        offspring = mutation(population, best)
                        offspring = crossover(population[j],offspring)
                        selected_offspring = selection(env,population[j],offspring)
                        new_population.append(selected_offspring)
                    
                    population = new_population
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
                    
                    np.savetxt(experiment_name+'/best.txt', global_best_individual)
                    
                    solutions = [population, fitness]
                    env.update_solutions(solutions=solutions)
                    env.save_state()
                
    if test_train == 'test':
            #
            #Dit is de test omgeving. Hier kunnen we de resultaten tegen test enemies zien
            #Hier moet wel nog code worden geschreven om iets te kunnen doen met testen
            #check misschien competition_results_v2.py

        for i in range(5):
            
            env = Environment(experiment_name=experiment_name,
                        enemies=opponent_group,
                        multiplemode='yes',
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False,
                        randomini='yes')
            

            sol = np.loadtxt(experiment_name+'/best.txt')
            fitness = simulation(env, sol) #dit is de fitness van een spelletje met nieuwe enemies

            player_life = env.get_playerlife()
            enemy_life = env.get_enemylife()

            test_name = f'TEST_NORM_{opponent_group}'
            if not os.path.exists(test_name):
                os.makedirs(test_name)

            file = open(test_name+'/results.txt', 'a')
            file.write(str(fitness)+' '+str(player_life)+' '+str(enemy_life)+'\n')
            file.close()
                    

def simulate_wrapper(args):
    experiment_name, enemy = args
    return main(experiment_name, enemy)
        
experiment_names = [('DE_NORM_G1',[1,2,3,4]), ('DE_NORM_G2',[5,6,7,8])]

import multiprocessing as mp

test_train = 'train'

if test_train == 'train':
    if __name__ == '__main__':
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(simulate_wrapper, experiment_names)
if test_train == 'test':
    for tup in experiment_names:
        experiment_name, enemy = tup
        main(experiment_name, enemy, 'test')