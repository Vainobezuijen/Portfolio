nodes = 5  # Number of nodes
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]  # List of edges

# import necessary modules
import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)  # Layout for visual spacing

    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='black', font_size=14, font_weight='bold')
    plt.title("Sample Graph Visualization")
    plt.show()

def graph_to_cnf(k, nodes, edges):
    """
    Generates a CNF formula for the k-coloring problem on graph G.
    Each variable var(v,c) represents that node v has color c.
    
    Parameters:
    k (int): The number of colors
    nodes(int): The number nodes of the graph
    edges(list): The number of edges of the graph

    Returns:
    clauses (list of list of int): The CNF clauses
    """
    
    clauses = []
    # map each node v with color c to a unique variable (integer)
    def var(v, c):
        return (v - 1) * k + c
    
    for v in range(1, nodes + 1):
        # each node has to have at least one color
        clauses.append([var(v, c) for c in range(1, k + 1)])
        # a node cannot have two colors at the same time
        for c1 in range(1, k + 1):
            for c2 in range(c1 + 1, k + 1):
                # for each combination of two colors, at most one of these can be true
                clauses.append([-var(v, c1), -var(v, c2)])

    for (u, v) in edges:
        for c in range(1, k + 1):
            clauses.append([-var(u, c), -var(v, c)])

    return clauses

def parse_sat_output(result_file, k, nodes):
    """
    Parses the output.txt file output from the SAT solver and extracts
    the variable assignments, then maps them to vertex-color pairs.
    """
    node_color_pairs = []
    with open(result_file, 'r') as f:
        lines = f.readlines()
    if lines[0].strip() == "UNSAT":
        return -1
    assignments = list(map(int, lines[1].split()))
    for assignment in assignments:
        if assignment > 0:  # A positive value means the variable is True
            variable = assignment
            vertex = (variable - 1) // k + 1  # Find the vertex from variable
            color = (variable - 1) % k + 1    # Find the color from variable
            node_color_pairs.append((vertex, color))

    return node_color_pairs


def output(node_color_pairs):
    """
    Outputs the node-color pairs in the required format.
    """
    if node_color_pairs == -1:
        print("-1")
    else:
        for node, color in sorted(node_color_pairs):
            print(f"{node} {color}")

    # write results to a file:
    with open("sample_solutions.txt", 'w') as f:
        if node_color_pairs == -1:
            f.write("-1\n")
        else:
            for node, color in sorted(node_color_pairs):
                f.write(f"{node} {color}\n")

def plot_colored_graph(edges):

    G = nx.Graph()
    G.add_edges_from(edges)

    node_color_pairs = parse_sat_output(result_file, k, nodes)
    if node_color_pairs == -1:
        print("-1")
        return

    color_map = []
    
    for node, color in sorted(node_color_pairs):
        if color == 1:
            color_map.append('red')
        elif color == 2:
            color_map.append('blue')
        elif color == 3:
            color_map.append('green')
        else:
            color_map.append('magenta')  
    
    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, edge_color='black', font_size=14, font_weight='bold')
    plt.title("Sample Graph Visualization")
    plt.show()



edges = [
(0,2), (0,3), (0,5),(1,3), (1,4), (1,6),(2,4), (2,7),(3,8),(4,9),(5,6), (5,9),(6,7),(7,8),(8,9)
]

# Turn into DIMACS file
k = 4
clauses = graph_to_cnf(k, nodes, edges)
num_variables = nodes * k
num_clauses = len(clauses)
cnf_output = f"p cnf {num_variables} {num_clauses}\n"
for clause in clauses:
    cnf_output += " ".join(map(str, clause)) + " 0\n"
print(cnf_output)
with open("sample_graph_4color.cnf", 'w') as f:
    f.write(cnf_output)

# SAT solving
import subprocess

result = subprocess.run(['minisat', "sample_graph_4color.cnf", 'sample_output.txt'], capture_output=True)
print(result.stdout.decode())

# parse the result.txt file
result_file = "sample_output.txt"
node_color_pairs = parse_sat_output(result_file, k, nodes)

# Output the node-color assignments
output(node_color_pairs)

# Call plot_graph function
plot_colored_graph(edges)