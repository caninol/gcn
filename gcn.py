
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
# TODO: Use numpy arrays instead of lists of lists.
# TODO: Better encapsulate find_cells_on_boundary.

import argparse
import random
import copy
import time

def get_example(example: int = 1) -> tuple[list[list[int | float]], int]:
    """Gets the grid and Manhattan distance threshold used in the examples.

    Args:
        example (int, optional): Number for example (optional). Defaults to 1.

    Returns:
        tuple[list[list[int | float]], int]: Tuple of two-dimensional array
        representing the grid, Manhattan distance threshold
    """
    grid = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]

    N = 1

    # Set the neighborhood centers
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

def generate_random_grid(H: int = None, W: int = None, sampling: float = 0.1) -> list[list[float]]:
    """Generates a grid of random floating point values with the specified height and width.

    If either the height or the width is not specified, then a random value
    between 1 and 10 is chosen for the missing dimension (either height or
    width or both).

    Args:
        H (int, optional): Height of grid. Defaults to None.
        W (int, optional): Width of grid. Defaults to None.
        sampling (float, optional): Sampling frequency. Defaults to 0.1.

    Returns:
        list[list[float]]: Two-dimensional array of floats representing the grid
    """
    # Randomly pick a height and width for the grid between 1 and 10
    H = random.randint(1, 10) if H is None else H
    W = random.randint(1, 10) if W is None else W

    # Make sure the grid has is at least 1x1
    assert H > 0 and W > 0

    # Randomly assign each sampled grid cell with a value between -0.5 
    # and 0.5. Set the unsampled grid cells to -1.0.
    return [[(random.random() - 0.5 if random.random() < sampling else -1.0) 
             for _ in range(W)  ] for _ in range(H)]

def load_grid_from_file(filename: str) -> list[list[int | float]]:
    """Loads a grid from the specified file.

    The format of the grid file is simply lines of whitespace-separated
    numerical values, with each line representing a row in the grid and
    the number of values per row determining the number of columns in
    the grid.

    Args:
        filename (str): Name of (i.e., the path to) the grid file

    Returns:
        list[list[float]]: Two-dimensional array of floating point
        values representing the grid
    """
    grid = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            # Yuck! Need to serialize/deserialize the Grid object,
            # but, for expediency, we'll do this for now...
            grid.append([float(v) for v in line.split()])

    # Assert that grid is a two-dimensional array of fixed width
    H = len(grid)
    assert H > 0
    
    W = len(grid[0])
    assert W > 0

    for row in grid:
        assert len(row) == W

    return grid

def print_two_dimensional_array(array: list[list[int | float | str]]) -> None:
    """Prints out the provided two-dimensional array.

    Args:
        array (list[list[int  |  float  |  str]]): Two-dimensional array
    """
    array_str = ""

    # Build the output string while validating that the input is a
    # two-dimensional array
    assert isinstance(array, list)
    for row in array:
        assert isinstance(row, list)
        array_str += str(row).replace(",", "").replace("'", "") + '\n'

    # Print out the array, represented as a string
    print(array_str)

def find_neighborhood_centers(grid: list[list[int | float]]) -> list[tuple[int, int]]:
    """Finds the neighborhood centers.

    Neighborhood centers are those cells with positive values.

    Args:
        grid (list[list[int  |  float]]): Two-dimensional array representing the grid

    Returns:
        list[tuple[int, int]]: Neighborhood centers as a list of grid coordinate tuples
    """
    centers = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] > 0:
                centers.append((i, j))

    return centers

def find_cells_in_neighborhood_iterative(grid: list[list[int | float]], N: int) -> set[tuple[int, int]]:
    """Finds the cells in the neighborhood of the centers.

    This method iterates over the cells within the specified Manhattan distance threshold
    of the centers in order to determine the neighborhood.

    Args:
        grid (list[list[int  |  float]]): Two-dimensional array representing the grid
        N (int): Manhattan distance threshold

    Returns:
        set[tuple[int, int]]: Set of cells in the neighborhood of the centers
    """
    # Initialize a set to track the cells
    cells = set()

    # Get the height and width of the grid
    H = len(grid)
    W = len(grid[0])

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

def find_cells_in_neighborhood_recursive(grid: list[list[int | float]], N: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Finds the cells in the neighborhood of the centers.

    This method performs a recursive breadth-first search to identify the cells within
    the specified Manhattan distance of each center and, thus, determine the neighborhood.

    Args:
        grid (list[list[int  |  float]]): Two-dimensional array representing the grid
        N (int): Manhattan distance threshold

    Returns:
        tuple[set[tuple[int, int]], list[tuple[int, int]]]: Tuple of cells in neighborhood, centers
    """

    # Define a function that will be called recursively in order to perform the
    # breadth-first search.
    def bfs(cell: tuple[int, int], level: int, visited: list[list[bool]]) -> set[tuple[int, int]]:
        """Performs a bread-first search for the neighborhood cells.

        Args:
            cell (tuple[int, int]: Cell from which to search for neighborhood cells
            level (int): Recursion level (i.e., step from center)
            visited (list[list[bool]]): Two-dimensional array to track visited cells

        Returns:
            set[tuple[int, int]]: Set of cells in the neighborhood
        """
        # Initialize a set to track the cells
        cells = set()

        # Get the height and width of the grid from the two-dimensional
        # visited array
        H = len(visited)
        W = len(visited[0])

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

            # Hard-code the directions to search
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

            # Iterate over the directions.
            # TODO: Avoid re-visiting the cell from which we came!
            for direction in directions:

                # Recurse into the next level of cells.
                cells.update(bfs((Y + direction[0], X + direction[1]), level-1, visited))
    
        return cells

    # Initialize a set to track the cells
    cells = set()

    # Get the height and width of the grid
    H = len(grid)
    W = len(grid[0])

    # Initialize a two-dimensional array to track the cells that
    # have been visited
    visited = [[False] * W for _ in range(H)]

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
     
def main(grid: list[list[int | float]], N: int, bRecurse: False, bBoundary: False) -> None:
    """Main function

    Args:
        grid (list[list[int  |  float]]): Two-dimensional array representing the grid
        N (int): Manhattan distance threshold
        bRecurse (bool): Use recusive method
        bboundary (bool): Determine boundary
    """
    # Validate the grid and Manhattan distance threshold
    assert len(grid) > 0 and len(grid[0]) > 0
    assert N >= 0

    # Print out the initial grid
    print("\nThe initial grid is:")
    print_two_dimensional_array(grid)

    # Print out the two dimensional array showing just the signs
    # of the values in the grid. We'll call this the "signed grid."
    print("\nThe signed grid is:")
    grid_signed = [['+' if v > 0 else '-' for v in grid[row]] for row in range(len(grid))]
    print_two_dimensional_array(grid_signed)

    # Find the cells that make up the neighborhood
    start_time = time.perf_counter_ns()
    cells, centers = find_cells_in_neighborhood_iterative(grid, N) if bRecurse else find_cells_in_neighborhood_recursive(grid, N)
    stop_time = time.perf_counter_ns()

    print(f"The execution time was {(stop_time - start_time) / 1000.0:.2f} microseconds")

    # Find the cells on the boundary, if requested
    if bBoundary:
        cells = find_cells_on_boundary(cells, centers, len(grid), len(grid[0]))

    # Contruct a grid showing the neighborhood or boundary
    grid_result = [['-' for _ in range(len(grid[0]))] for _ in range(len(grid))]
    for Y, X in cells:
        grid_result[Y][X] = 'x'

    print("\nThe resulting grid is:")
    print_two_dimensional_array(grid_result)      

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

    # Initialize parameters
    grid = []
    N = args.N

    # Get the grid, either using the example, generating a random
    # one, or reading it from a file
    if args.example:
        grid, N = get_example(args.example)

    elif args.random:
        grid = generate_random_grid(args.H, args.W, args.S)

    elif args.filename:
        grid = load_grid_from_file(args.filename)

    # Call the main function
    main(grid, N, args.recurse, args.boundary)