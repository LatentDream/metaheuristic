import numpy as np
import random as r
import networkx as nx
import matplotlib.pyplot as plt

class Atelier():

    def __init__(self,filename):
        """ Creates an instance of the problem using the data contained in `filename`"""

        with open(filename,'r') as f:
            lines = f.readlines()
        
        # Dimension of the instance
        self.n_jobs = int(lines[0])
        self.n_machines = int(lines[1])

        self.dist = np.zeros((self.n_machines,self.n_machines),dtype=int)
        self.orders = dict()

        # Distances between sites
        curr = 3
        for i in range(self.n_machines):
            l = lines[curr].split()
            for j in range(self.n_machines):
                self.dist[i,j]=l[j]
            curr+=1
        
        # Orders related to each job
        curr+=1
        for i in range(self.n_jobs):
            l = lines[curr].split()
            l = [int(a) for a in l]
            self.orders[i] = l[::]
            curr+=1

    def get_total_cost(self,solution):
        """Computes the cost of a solution
        :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines
        :return: the cost"""
        cost = 0
        for i in range(self.n_jobs):
            for j in range(len(self.orders[i])-1):
                cost += self.dist[solution[self.orders[i][j]],solution[self.orders[i][j+1]]]
        return cost
    
    def generate_random_solution(self):
        """Creates a random solution for the problem

        :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

        solution = dict()
        sites = list(range(self.n_machines))
        r.shuffle(sites)

        for i in range(len(sites)):
            solution[i] = sites[i]
        return solution
    
    def generate_greedy_solution(self):
        """Creates a greedy initial solution for the problem
        :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""
        
        # Initialisation
        solution = dict()
        for i in range(self.n_machines):
            solution[i] = 0
        
        # Randomly determining the orders of the sites
        sites = list(range(1,self.n_machines))
        r.shuffle(sites)

        for j in sites:
            best_cost = 10**10
            best_mach = -1
            possible = [a for a in range(self.n_machines) if solution[a]==0]
            for i in possible:
                solution[i] = j
                cost = self.get_total_cost(solution)
                if cost < best_cost:
                    best_cost = cost
                    best_mach = i
                solution[i] = 0
            solution[best_mach] = j

        return solution
    
    def generate_initial_solution(self,args):
        """Creates an initial solution

        args
        :param mode: mode chosen for the generation (random or greedy)"""

        if args.mode == "random":
            solution = self.generate_random_solution()
        elif args.mode == "greedy":
            solution = self.generate_greedy_solution()
        else:
            raise Exception("Generation mode not implemented")
        
        return solution
    
    def save_solution(self, solution, args):
        """Saves the solution as a txt file.
        :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines
        
        args
        :param outfile: the file in which to write the solution"""
        
        with open(args.outfile,'w') as f:
            f.write("%s\n" % self.get_total_cost(solution))
            for i in solution:
                f.write("%s " % solution[i])
    
    def display_solution(self, solution, args):
        """Displays a solution as a png
        :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines
        
        args
        :param visufile: the file in which to draw the solution"""
        g = nx.MultiDiGraph()
        for i in range(self.n_machines):
            g.add_node(i, label=[k for k in solution if solution[k]==i][0])
        for i in range(self.n_jobs):
            col = (r.randint(100,255)/255,r.randint(100,255)/255,r.randint(100,255)/255)
            for j in range(len(self.orders[i])-1):
                s1 = [v for k,v in solution.items() if k == self.orders[i][j]][0]
                s2 = [v for k,v in solution.items() if k == self.orders[i][j+1]][0]
                g.add_edge(s1, s2, weight=self.dist[s1,s2], color = col)

        nodes = g.nodes(data=True)
        edges = g.edges(data=True)
        lab = {u[0]:u[1]['label'] for u in nodes}
        colors = [g[u][v][0]['color'] for u,v,_ in edges]
        weights = [g[u][v][0]['weight'] for u,v,_ in edges]
        nx.draw_networkx_edge_labels(g, pos=nx.spring_layout(g,seed=10), edge_labels={(u,v):g[u][v][0]['weight'] for u,v,_ in edges})
        nx.draw(g, pos=nx.spring_layout(g,seed=10), with_labels=True, labels = lab, edges=edges, edge_color=colors, edge_labels=weights)
        plt.title('n_jobs : %s, n_machines : %s\n solution cost : %s' % (self.n_machines, self.n_jobs, str(self.get_total_cost(solution))), size=15, color='red')
        plt.savefig(args.visufile)