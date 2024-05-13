"""
Chessboard Movement Program

This Python program takes two positions on a chessboard as input and returns the set of all
minimum-length sequences to move the piece from the initial cell to the final cell.
The program employs a breadth-first search algorithm to find these minimum-length sequences.

"""

import graphviz


class KnightProblem:
    """
    Class representing the knight's problem
    """

    def __init__(self, x=0, y=0, dist=0, initial_path=None):
        """
        Initialize the KnightProblem instance

        :param x: x-coordinate
        :param y: y-coordinate
        :param dist: distance
        :param initial_path: initial_path
        """
        self.x = x
        self.y = y
        self.dist = dist
        self.initial_path = initial_path or []


def is_inside(x, y, boardsize):
    """
    Check whether the given position is inside the board

    :param x: x-coordinate
    :param y: y-coordinate
    :param boardsize: size of the chessboard
    :return: True if inside the board, False otherwise
    """
    return 1 <= x <= boardsize and 1 <= y <= boardsize


def min_step_to_reach_target(startpos, endpos, size):
    """
    Method returns minimum step to reach target position

    :param knightpos: starting position of the knight
    :param targetpos: target position
    :param size: size of the chessboard
    :return: list of all minimum length paths
    """

    # all possible movements for the knight
    dx = [2, 2, -2, -2, 1, 1, -1, -1]
    dy = [1, -1, 1, -1, 2, -2, 2, -2]

    queue = []

    # push starting position of knight with 0 distance
    queue.append(KnightProblem(startpos[0], startpos[1], 0, [(startpos[0], startpos[1])]))

    # make all cell unvisited
    visited = [[False for _ in range(size + 1)] for _ in range(size + 1)]

    # visit starting state
    visited[startpos[0]][startpos[1]] = True

    # list to store all minimum length paths
    all_paths = []

    # loop until we have one element in queue
    while queue:
        t = queue[0]
        queue.pop(0)

        # if current cell is equal to target cell
        if t.x == endpos[0] and t.y == endpos[1]:
            # save the current path
            all_paths.append(t.initial_path)

        # iterate for all reachable states
        for i in range(8):
            x = t.x + dx[i]
            y = t.y + dy[i]

            if is_inside(x, y, size) and not visited[x][y]:
                visited[x][y] = True
                # update the path with the current position
                new_path = t.initial_path + [(x, y)]
                queue.append(KnightProblem(x, y, t.dist + 1, new_path))

    return all_paths


def visualize(boardsize, paths):
    """
    Function to create a Graphviz/DOT file for the chessboard

    :param boardsize: size of the chessboard
    :param paths: list of paths
    """
    dot = graphviz.Digraph(comment='Chessboard')

    # Create nodes for all cells in the chessboard
    for i in range(1, boardsize + 1):
        for j in range(1, boardsize + 1):
            node_label = f"({i}, {j})"
            dot.node(f"{i}_{j}", label=node_label)

    # Connect adjacent cells
    for i in range(1, boardsize + 1):
        for j in range(1, boardsize + 1):
            # Connect horizontally and vertically
            if i < boardsize:
                dot.edge(f"{i}_{j}", f"{i + 1}_{j}")
            if j < boardsize:
                dot.edge(f"{i}_{j}", f"{i}_{j + 1}")

    # Highlight the shortest paths
    for initial_path in paths:
        for i in range(len(initial_path) - 1):
            start_node = initial_path[i]
            end_node = initial_path[i + 1]
            dot.edge(f"{start_node[0]}_{start_node[1]}",
                      f"{end_node[0]}_{end_node[1]}", color='red', style='bold')

    dot.render('chessboard', format='jpg', cleanup=True)


if __name__ == '__main__':
    n = int(input("Enter the size of the chessboard (default is 8): ") or 8)
    knightpos_x = int(input("x-coordinate of knight's start position (default is 2): ") or 2)
    knightpos_y = int(input("y-coordinate of knight's start position (default is 2): ") or 2)
    targetpos_x = int(input("x-coordinate of target position (default is 4): ") or 4)
    targetpos_y = int(input("y-coordinate of target position (default is 3): ") or 3)

    knightpos = [knightpos_x, knightpos_y]
    targetpos = [targetpos_x, targetpos_y]

    # Function call
    calculated_paths = min_step_to_reach_target(knightpos, targetpos, n)

    # Display all minimum length paths
    for path in calculated_paths:
        print("Path:", path)
        print("Length:", len(path) - 1)  # Length of the path excluding the starting position
        print()

    # Create Graphviz/DOT file for the chessboard with highlighted shortest paths
    visualize(n, calculated_paths)
