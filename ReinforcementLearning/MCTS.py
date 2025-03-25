# Functions

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

DEPTH = 20
ITERATIONS = 50
B = 10
TARGET_ADDRESS = 'LLLLLLLLLLLLLLLLLLLL'  # Example target leaf
MAX_SNOWCAP_ROLLOUTS = 1
FILENAME = 'results_MCTS.txt'

class Node():
    def __init__(self, depth=0, address='', parent=None):
        self.depth = depth
        self.address = address # LRLRLRLRLR
        self.visits = 0 
        self.value = 0 # edit distance
        self.total_reward = 0 # cumulative reward
        self.parent = parent
        self.children = [] # list of nodes
        self.epsilon = np.random.normal(0,1)

    def update_node(self, value):
        self.total_reward += value
        self.visits += 1

    def backpropagate(self, node, value):
        while node.parent != None:
            node.update_node(value)
            node = node.parent

def calc_edit_distance(Ai, At):
    return sum(char1 != char2 for char1, char2 in zip(Ai, At))

def value_of_state(dmax, di, node):
    t = dmax / 5
    return B * (math.e ** (-di/t)) + node.epsilon

def calculate_UCB(child, c):
    if child.parent is None or child.parent.visits == 0:
        return float('inf')
    if child.visits == 0:
        return float('inf')

    exploitation = child.total_reward / child.visits
    exploration = c * math.sqrt(math.log(child.parent.visits) / child.visits)
    return exploitation + exploration


def expand_node(node):
    if node.depth < 20:
        node.children = [
            Node(depth=node.depth+1, address=node.address+'L',parent=node),
            Node(depth=node.depth+1, address=node.address+'R',parent=node)
        ]

    if not node.children:
        return

    for i in range(MAX_SNOWCAP_ROLLOUTS):
        # print(f'SNOWCAP: {i}')
        new_child = random.choice(node.children)
        reward = rollout(new_child)
        backpropagate(new_child, reward)

def rollout(node, target_address=TARGET_ADDRESS, dmax=DEPTH):
    while node.depth < DEPTH:
        if not node.children:
            expand_node(node)
        node = random.choice(node.children)
    
    di = calc_edit_distance(node.address, target_address)
    return value_of_state(dmax, di, node)

def selection(root, c):
    node = root
    while node.depth < DEPTH:
        if not node.children:
            return node
        best_child = None
        best_ucb = float('-inf')
        for child in node.children:
            ucb_val = calculate_UCB(child, c)
            if ucb_val > best_ucb:
                best_ucb = ucb_val
                best_child = child
        node = best_child
    return node

def backpropagate(node, reward):
    while node is not None:
        node.update_node(reward)
        node = node.parent

def uct_search(root, iterations, c):
    target_found_iteration = None
    for i in range(iterations):
        print(f'Iteration {i}')
        node_to_expand = selection(root, c)
        expand_node(node_to_expand)
        reward = rollout(node_to_expand)
        if node_to_expand.address == TARGET_ADDRESS and target_found_iteration is None:
            target_found_iteration = i + 1  
            print('Target found')
        backpropagate(node_to_expand, reward)
    return root, reward, target_found_iteration

def find_best_leaf(node):
    if not node.children:  # Leaf
        if node.visits > 0:
            return node, node.total_reward / node.visits
        else:
            return node, float('-inf')
    best_node, best_score = None, float('-inf')
    for child in node.children:
        c_node, c_score = find_best_leaf(child)
        if c_score > best_score:
            best_node, best_score = c_node, c_score
    return best_node, best_score

def run_trial(c, iterations, trial_number, total_trials):
    # print(f"Running trial {trial_number}/{total_trials} for c = {c:.1f}\n")
    root = Node(depth=0, address="")
    _, reward, target_iteration = uct_search(root, iterations, c)
    best_leaf, best_avg_reward = find_best_leaf(root)
    print(c,reward,target_iteration,best_avg_reward)
    with open(FILENAME, 'a') as file:
        file.write(f'{c},{reward},{target_iteration},{best_avg_reward}\n')
    return c, reward, target_iteration, best_avg_reward

def run_experiment_parallel(c_values, iterations, num_trials):
    results = []
    jobs = []
    trial_number = 1
    total_trials = len(c_values) * num_trials
    for c in c_values:
        for _ in range(num_trials):
            jobs.append((c, iterations, trial_number, total_trials))
            trial_number += 1
    with Pool() as pool:
        outcomes = list(tqdm(pool.starmap(run_trial, jobs), total=total_trials, desc="Running Trials"))
    idx = 0
    for c in c_values:
        for _ in range(num_trials):
            c_value, reward, target_iteration, best_reward = outcomes[idx]
            results.append([c_value, reward, target_iteration, best_reward])
            idx += 1
    return results

def analyze_results(filename):
    results = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(',')
            results.append(line)
    df = pd.DataFrame(results, columns=["c", "reward", "iteration", "average_reward"])
    return df

def plot_results(results):

    results[['reward', 'iteration']].describe()

    avg_rewards = results.groupby('c')['average_reward'].mean().reset_index()
    plt.figure(figsize=(12, 4))

    plt.subplot(1,2,1)
    plt.plot(avg_rewards['c'], avg_rewards['average_reward'], marker='o', label="Average Best Reward")
    plt.xlabel("Exploration Parameter (c)")
    plt.ylabel("Average Best Reward")
    plt.title("Performance of MCTS for Different c Values")
    plt.grid(True)
    plt.legend()
    

    plt.subplot(1,2,2)
    labels = results['c']
    data = results['iteration']
    # plt.figure(figsize=(10, 6))
    sns.violinplot(x=labels, y=data, scale="width")
    plt.xlabel("Exploration Parameter (c)")
    plt.ylabel("Iterations to Reach Target")
    plt.title("Iterations to Reach Target State for Different c Values")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    c_values = [0.1, 1, 2, 3, 4]
    num_trials = 50
    results = run_experiment_parallel(c_values, ITERATIONS, num_trials)
    results = analyze_results(FILENAME)
    results = results.fillna(0)
    results.replace(['None', 'N/A', '', ' '], np.nan, inplace=True)
    results = results.astype(float, errors='ignore')
    plot_results(results)