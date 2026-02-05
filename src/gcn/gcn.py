"""
Grid-Cell Neighborhoods

This module provides a couple of solutions to the Grid-Cell Neighborhoods exercise,
the goal of which is to identify all cells on a two-dimensional array of numerical
values that are within a specified Manhattan distance of cells with positive values.

This problem is conceptually equivalent to finding the union of the volumes of all
of the "spheres" with radii equal to the Manhattan distance threshold and centered
on the cells with positive value.

A little terminology is useful here. We define the following terms:

- Grid: The two-dimensional array of numerical values is known as the "grid."

- Centers: Cells with positive values are referred to as "neighborhood centers"
  or simply as "centers."

- Neighborhood of a center: The collection of cells within the specified Manhattan
  distance of a given center X is known as the "neighborhood of center X."

- Neighborhood: The union of the neighborhoods of all centers is known simply as
  the "neighborhood."

- Boundary: The "boundary" of the neighborhood is defined as the collection of
  cells that are less than the specified Manhattan distance from all centers.

The default behavior for this module is to execute the iterative solution. For
performance comparison, a recursive, breadth-first search algorithm is also
provided. This is enabled through the use of the -R or --recurse flag.

This module can be invoked with the -b or --boundary flag to provide just the
boundary of the neighborhood.
"""

# TODO: Define a Grid class to represent the grid.
# TODO: Define a Cell class to represent a cell.
# TODO: Better encapsulate find_cells_on_boundary.

import argparse
import sys
import numpy as np
import random
import copy
import time

def get_example(example: int = 1) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], int]:
    """Gets the grid and Manhattan distance threshold used in the specified example.

    Args:
        example (int, optional): Number for example: {1, 2, 3, 4} (optional). Defaults to 1.

    Raises:
        ValueError: Raised if example is invalid

    Returns:
        tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], int]: Tuple of two-dimensional 
        numpy array representing the grid, Manhattan distance threshold
    """
    # Create an 11 x 11 array with all cells initialized to -1
    grid = np.array([-1] * 121, dtype=np.int_).reshape((11, 11))

    # Default the Manhattan distance threshold to 0
    N = 0

    # Set the cells where the centers reside to 1
    if example == 1:
        grid[5][5] = 1
        N = 3
    elif example == 2:
        grid[5][1] = 1
        N = 3
    elif example == 3:
        grid[3][7] = 1
        grid[7][3] = 1
        N = 2
    elif example == 4:
        grid[6][5] = 1
        grid[7][3] = 1
        N = 2
    else:
        raise ValueError("Invalid example number!")

    return grid, N

def generate_random_grid(H: int = None, W: int = None, sampling: float = 0.1) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Generates a grid of random floating point values with the specified height and width.

    If either the height or the width is not specified, then a random value
    between 1 and 10 is chosen for the missing dimension (either height or
    width or both).

    Args:
        H (int, optional): Height of grid. Defaults to None.
        W (int, optional): Width of grid. Defaults to None.
        sampling (float, optional): Sampling frequency. Defaults to 0.1.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.float32]]: Two-dimensional numpy array of
        floats representing the grid
    """
    # Randomly pick a height and width for the grid between 1 and 10
    H = random.randint(1, 10) if H is None else H
    W = random.randint(1, 10) if W is None else W
    
    # Create a one-dimensional list of size H * W with random values
    values_list = [(random.random() - 0.5 if random.random() < sampling else -1.0) for _ in range(H * W) ]

    # Return the grid as a two-dimensional numpy array of floats
    return np.array(values_list, dtype=np.float32).reshape((H, W))

def load_grid_from_file(filename: str) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Loads a grid from the specified file.

    The format of the grid file is simply lines of whitespace-separated
    numerical values, with each line representing a row in the grid and
    the number of values per row determining the number of columns in
    the grid.

    Args:
        filename (str): Name of (i.e., the path to) the grid file

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.float32]]: Two-dimensional 
        numpy array of floating point values representing the grid
    """
    return np.loadtxt(filename)

def print_grid(array: np.ndarray[tuple[int, int], np.dtype[np.any]], positive_char: chr = "x") -> None:
    """Prints out the provided grid.

    Args:
        array (np.ndarray[tuple[int, int], np.dtype[np.any]]]): Grid as a two-dimensional array
        positive_char (chr, optional): Character to use for positive values
    """
    # Get the height and width of the array
    H, W = array.shape
    
    # Build the array string
    array_str = ""
    for i in range(H):
        array_str += "["
        for j in range(W):
            array_str += positive_char if array[i][j] > 0 else "-"
        array_str += "]\n"

    # Print out the array string
    print(array_str)

def find_neighborhood_centers(grid: np.ndarray[tuple[int, int], np.dtype[np.any]]) -> list[tuple[int, int]]:
    """Finds the neighborhood centers.

    Neighborhood centers are those cells with positive values.

    Args:
        grid (np.ndarray[tuple[int, int], np.dtype[np.any]]): Two-dimensional numpy 
        array representing the grid

    Returns:
        list[tuple[int, int]]: Neighborhood centers as a list of grid coordinate tuples
    """
    return list(map(tuple, np.argwhere(grid > 0).tolist()))

def find_cells_in_neighborhood_iterative(grid: np.ndarray[tuple[int, int], np.dtype[np.any]], N: int) -> set[tuple[int, int]]:
    """Finds the cells in the neighborhood of the centers.

    This method iterates over the cells within the specified Manhattan distance threshold
    of the centers in order to determine the neighborhood.

    Args:
        grid (np.ndarray[tuple[int, int], np.dtype[np.any]]): Two-dimensional numpy array 
        representing the grid

        N (int): Manhattan distance threshold

    Returns:
        set[tuple[int, int]]: Set of cells in the neighborhood of the centers
    """
    # Initialize a set to track the cells
    cells = set()

    # Get the height and width of the grid
    H, W = grid.shape

    # Get the neighborhood centers
    centers = find_neighborhood_centers(grid)

    # Iterate over the centers
    for center in centers:
        Y0, X0 = center

        # Iterate over the grid to determine the cells which are in the
        # neighborhood of the center
        for i in range(max(-Y0, -N), min(H-Y0, N+1)):
            Y = Y0 + i
            
            # Set the column span for this row
            S = N - abs(i)

            for j in range(max(-X0, -S), min(W-X0, S+1)):
                X = X0 + j

                # Add the cell to the neighborhood set since, by construction,
                # it is within the Manhattan distance threshold of the center
                cells.add((Y, X))

    return cells, centers

def find_cells_in_neighborhood_recursive(grid: np.ndarray[tuple[int, int], np.dtype[np.any]], N: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Finds the cells in the neighborhood of the centers.

    This method performs a recursive breadth-first search to identify the cells within
    the specified Manhattan distance of each center and, thus, determine the neighborhood.

    Args:
        grid (np.ndarray[tuple[int, int], np.dtype[np.any]]): Two-dimensional numpy array
        representing the grid

        N (int): Manhattan distance threshold

    Returns:
        tuple[set[tuple[int, int]], list[tuple[int, int]]]: Tuple of cells in neighborhood, centers
    """

    # Define a function that will be called recursively in order to perform the
    # breadth-first search.
    def bfs(cell: tuple[int, int], level: int, visited: np.ndarray[tuple[int, int], np.dtype[np.bool]]) -> set[tuple[int, int]]:
        """Performs a bread-first search for the neighborhood cells.

        Args:
            cell (tuple[int, int]: Cell from which to search for neighborhood cells
            
            level (int): Recursion level (i.e., step from center)
            
            visited (np.ndarray[tuple[int, int], np.dtype[np.bool]]): Two-dimensional array of bools 
            to track visited cells

        Returns:
            set[tuple[int, int]]: Set of cells in the neighborhood
        """
        # Initialize a set to track the cells
        cells = set()

        # Get the height and width of the grid from the two-dimensional
        # visited array
        H, W = visited.shape

        # Get the coordinates of the cell
        Y, X = cell

        # Add the cell if it is on the grid and has not yet been visited
        if Y >= 0 and Y < H and X >= 0 and X < W:
            if not visited[Y][X]:
                # Add the cell to the neighborhood
                cells.add(cell)

                # Mark the cell as visited
                visited[Y][X] = True

        # Recurse each of the neighboring directions
        if level > 0:

            # Iterate over the hard-coded directions.
            # TODO: Avoid re-visiting the cell from which we came!
            for direction in [(0, -1), (1, 0), (0, 1), (-1, 0)]:

                # Recurse into the next level of cells.
                cells.update(bfs((Y + direction[0], X + direction[1]), level-1, visited))
    
        return cells

    # Initialize a set to track the cells
    cells = set()

    # Get the height and width of the grid
    H, W = grid.shape

    # Initialize a two-dimensional array to track the cells that
    # have been visited
    visited = np.array([False] * (H * W)).reshape(H, W)

    # Get the neighborhood centers
    centers = find_neighborhood_centers(grid)

    # Iterate over the centers
    for center in centers:

        # Update the neighborhood with the cells that are in the
        # neighborhood of this center
        cells.update(bfs(center, N, visited))

    return cells, centers

def find_cells_on_boundary(cells: set[tuple[int, int]], centers: list[tuple[int, int]], H: int, W: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Finds cells on the boundary.

    Args:
        cells (set[tuple[int, int]]): Set of cells in the neighborhood
        centers (list[tuple[int, int]]): List of centers
        H (int): Height of the grid
        W (int): Width of the grid

    Returns:
        set[tuple[int, int]]: Tuple of boundary cells, centers
    """
    # Make a copy of the neighborhood set from which to remove cells
    # in the neighborhood that are not on the boundary. We need to use 
    # a copy here because we cannot remove items from a set while
    # iterating over it.
    cells_copy = copy.deepcopy(cells)

    # Remove cells from the neighborhood tha t are not on the boundary,
    # as long as they are not on the edge of the grid
    for cell in cells:
        Y, X = cell

        # Keep the cell if it's on the edge of the grid
        if not (Y == 0 or Y == H-1 or X == 0 or X == W-1):

            # Remove cells that are within the Manhattan distance
            # threshold of any of the centers
            for center in centers:
                Y0, X0 = center

                if abs(Y-Y0) + abs(X-X0) < N:
                    cells_copy.remove(cell)
                    break
    
    # Return the cells in the neighborhood that are on the boundary
    # of the neighborhood.
    return cells_copy
     
def main(grid: np.ndarray[tuple[int, int], np.dtype[np.any]], N: int, bRecurse: bool = False, bBoundary: bool = False) -> None:
    """Main function

    Args:
        grid (np.ndarray[tuple[int, int], np.dtype[np.any]]): Two-dimensional numpy array 
        representing the grid
        
        N (int): Manhattan distance threshold
        
        bRecurse (bool, optional): Use recusive method. Defaults to False.
       
        bboundary (bool, optional): Determine boundary. Defaults to False.
    """
    # Validate the grid size and Manhattan distance threshold
    assert grid.size > 0, "The grid cannot be empty!"
    assert N >= 0, "The Manhattan distance threshold cannot be negative!"

    # Print out the two dimensional array showing just the signs
    # of the values in the grid. We'll call this the "signed grid."
    print("\nThe signed grid is:\n")
    print_grid(grid, positive_char = "+")

    # Find the cells that make up the neighborhood
    start_time = time.perf_counter_ns()
    cells, centers = find_cells_in_neighborhood_iterative(grid, N) if not bRecurse else find_cells_in_neighborhood_recursive(grid, N)
    stop_time = time.perf_counter_ns()

    print(f"The execution time was {(stop_time - start_time) / 1000.0:.2f} microseconds")

    # Find the cells on the boundary, if requested
    if bBoundary:
        cells = find_cells_on_boundary(cells, centers, len(grid), len(grid[0]))

    # Contruct a grid showing the neighborhood or boundary
    H, W = grid.shape
    grid_result = np.array([-1] * (H * W), dtype=np.int_).reshape((H, W))
    for Y, X in cells:
        grid_result[Y][X] = 1

    print("\nThe result grid is:\n")
    print_grid(grid_result)      

    # The number of cells in the neighborhood or on the boundary 
    # is just the number of elements in the cells set
    print(f"The number of cells for the neighborhood or boundary is {len(cells)}")

if __name__ == "__main__":

    # Define the allowed command-line arguments
    parser = argparse.ArgumentParser("A script for the Grid-Cell Neighborhoods exercise")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--example', type=int, choices=[1, 2, 3, 4], help="The example to run")
    group.add_argument('-r', '--random', action='store_true', help="Generate a random grid")
    group.add_argument('-f', '--filename', type=str, help="Load the grid from a file")
    parser.add_argument('-N', type=int, default=0, help="Manhattan distance threshold (ignored if -e is specified)")
    parser.add_argument('-H', type=int, default=None, help="Grid height (ignored if either -e or -f is specified)")
    parser.add_argument('-W', type=int, default=None, help="Grid width (ignored if either -e or -f is specified)")
    parser.add_argument('-S', type=float, default=0.1, help="Sampling")
    parser.add_argument('-R', "--recurse", action='store_true', help="Use recursion")
    parser.add_argument('-b', "--boundary", action='store_true', help="Find the boundary")

    # Get the user-specified arguments
    args = parser.parse_args()

    # Initialize the parameters
    grid = np.array([])
    N = args.N

    # Get the grid, either using the example, generating a random
    # one, or reading it from a file. Bail if there is no grid.
    if args.example:
        grid, N = get_example(args.example)

    elif args.random:
        grid = generate_random_grid(args.H, args.W, args.S)

    elif args.filename:
        grid = load_grid_from_file(args.filename)

    # Call the main function
    main(grid, N, args.recurse, args.boundary)
