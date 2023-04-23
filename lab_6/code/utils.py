from math import *
import random as r
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rcParams['figure.figsize'] = [12, 6] # default = [6.0, 4.0]
plt.rcParams['figure.dpi']     = 100     # default = 72.0
plt.rcParams['font.size']      = 7.5     # default = 10.0



def fitness_function(particles):
    """The fitness function which will be used in default run"""
    return  np.array([-20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20 for x,y in particles])

def generate_initial_swarm(args):
    particles = np.zeros((args.n_particles, args.size))
    velocities = np.zeros((args.n_particles, args.size))

    for i in range(args.n_particles):
        for j in range(args.size):
            particles[i,j] = args.x_min[j]+r.random()*(args.x_max[j]-args.x_min[j])
            velocities[i,j] = args.v_min[j]+r.random()*(args.v_max[j]-args.v_min[j])
    
    return particles, velocities

def get_random_sol(args, fitness_function):
    particles=np.zeros((1000,args.size))
    for i in range(1000):
        for j in range(args.size):
            particles[i,j] = args.x_min[j]+r.random()*(args.x_max[j]-args.x_min[j])

    values = fitness_function(particles)
    idx, cost = min([(k,v) for (k,v) in enumerate(values)], key=lambda a:a[1])
    return particles[idx], cost


def plot_function(args, particles_in, particles_out):
    x_list = np.arange(args.x_min[0], args.x_max[0], 0.1)
    y_list = np.arange(args.x_min[0], args.x_max[1], 0.1)
    X, Y =  np.meshgrid(x_list, y_list)
    ackley_function = lambda x, y: -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * pi * x)+np.cos(2 * pi * y))) + e + 20
    Z = ackley_function(X,Y)

    fig, ax = plt.subplots(figsize=(6,6))

    #ax.contour(X, Y, Z)
    ax.contourf(X, Y, Z)

    for i in range(particles_in.shape[0]):
        ax.scatter(particles_in[i,0],particles_in[i,1],  marker = '^', color='black',zorder=1)

    #fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title("Initial position of the particles")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    plt.savefig('initial_solution.png')
 
    fig, ax = plt.subplots(figsize=(6,6))

    #ax.contour(X, Y, Z)
    ax.contourf(X, Y, Z)

    for i in range(particles_out.shape[0]):
        ax.scatter(particles_out[i,0],particles_out[i,1], marker = '^', color='red',zorder=1)

    #fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title("Final position of the particles")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    plt.savefig('final_solution.png')
 

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf)
if __name__ == '__main__':
    plot_function(0)