from eternity_puzzle import EternityPuzzle

# Function to flatten a grid into a list
def grid_to_list(grid):
    return [piece for row in grid for piece in row]


# Function to create a 2D grid from a list
def list_to_grid(e: EternityPuzzle, liste):
    grid = []
    for i in range(e.board_size):
        grid.append(liste[i * e.board_size : i * e.board_size + e.board_size])
    return grid


def visualize(e: EternityPuzzle, solution, name="visualisation"):
    e.display_solution(solution, name)
