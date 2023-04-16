import matplotlib.pyplot as plt
from typing import Dict, List
from utils.utils import *
from tqdm import tqdm
import solver_heuristic_layer
import numpy as np
import copy
import time







def prioritise_neighborhhod(e: EternityPuzzle, i, j, sigma, inverted, debug=False):
     #? Generate Gaussian filter 
    sigma=0.5
    muu=0

    i, j = 2*i, 2*j

    kernel_size = e.board_size * 2
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gaussian = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    if inverted:
        gaussian = (gaussian - 1) * -1

    #? filter on the i, j area
    half_board_range = e.board_size // 2

    i_lower = max(0, i-half_board_range)
    i_upper = i_lower + e.board_size
    if i_upper >= kernel_size:
        i_lower -= i_upper - kernel_size
        i_upper -= i_upper - kernel_size

    j_lower = max(0, j-half_board_range)
    j_upper = j_lower + e.board_size
    if j_upper >= kernel_size:
        j_lower -= j_upper - kernel_size
        j_upper -= j_upper - kernel_size

    neighborhood = gaussian[i_lower:i_upper, j_lower:j_upper]

    print(neighborhood)

    if debug:
        plt.imshow(neighborhood, cmap='binary')
        plt.colorbar()
        plt.savefig("selected_neighborhood")
        plt.close()

    return neighborhood