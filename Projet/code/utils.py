from eternity_puzzle import EternityPuzzle

# Function to flatten a grid into a list
def grid_to_list(grid):
    return [piece for row in grid for piece in row]


# Function to create a 2D grid from a list
def list_to_grid(liste, rows, cols):
    grid = []
    for i in range(rows):
        grid.append(liste[i * cols : i * cols + cols])
    return grid


def visualize(e: EternityPuzzle, solution):
    e.display_solution(solution, "visualization_solution.png")
