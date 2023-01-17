class Knapsack():

    def __init__(self,filename):
        ''' Lit un fichier contenant une instance du probleme du sac a dos sous forme 
            N C
            u_0 w_0
            u_1 w_1
            ...
            u_{N-1} w_{N-1}
            
            @pre:  filename: contient un String qui indique le fichier qui decrit l'instance
            @post: retourne un tuple (N,C,I), I étant la liste contenant tout les tuples (u_i,w_i)
        '''
        with open(filename) as file:
            lines = file.readlines()
            header = lines[0].strip().split()
            self.n = int(header[0]) # nombre d'elements
            self.c = int(header[1]) # capacite du sac a dos
            self.items = [(int(l[0]),int(l[1])) for l in [l.strip().split() for l in lines[1:]]] # liste des tuples (u_i,w_i) pour chaque element
              
