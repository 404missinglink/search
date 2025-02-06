from heapq import heappush, heappop
from math import sqrt

class AStarSearchProblem:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Find start and goal positions
        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i][j] == 'S':
                    self.start = (i, j)
                elif grid[i][j] == 'G':
                    self.goal = (i, j)

    def get_successors(self, state):
        x, y = state
        possible_moves = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)  # Diagonal moves
        ]
        
        return [(next_x, next_y) for next_x, next_y in possible_moves 
                if (0 <= next_x < self.rows and 
                    0 <= next_y < self.cols and 
                    self.grid[next_x][next_y] != 1)]

    def heuristic(self, state):
        """Euclidean distance heuristic"""
        x1, y1 = state
        x2, y2 = self.goal
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def cost(self, current, next_state):
        """Return cost of movement - diagonal moves cost more"""
        x1, y1 = current
        x2, y2 = next_state
        return sqrt(2) if abs(x1 - x2) + abs(y1 - y2) == 2 else 1

def astar_search(problem):
    """
    Perform A* search and return path from start to goal
    Also returns exploration history for visualization
    """
    start = problem.start
    frontier = [(0, start, [start])]  # (priority, state, path)
    explored = {start: 0}  # state -> cost to reach
    exploration_history = []  # Track explored states for visualization
    
    while frontier:
        f_cost, current, path = heappop(frontier)
        exploration_history.append(set(path))
        
        if current == problem.goal:
            return path, exploration_history
            
        for next_state in problem.get_successors(current):
            new_cost = explored[current] + problem.cost(current, next_state)
            
            if next_state not in explored or new_cost < explored[next_state]:
                explored[next_state] = new_cost
                priority = new_cost + problem.heuristic(next_state)
                heappush(frontier, (priority, next_state, path + [next_state]))
    
    return None, exploration_history

def print_solution(grid, path, explored=None):
    """
    Print the grid with:
    '*' for solution path
    '.' for explored cells
    'S' for start
    'G' for goal
    '#' for walls
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] in ['S', 'G']:
                print(grid[i][j], end=' ')
            elif (i, j) in path:
                print('*', end=' ')
            elif explored and (i, j) in explored:
                print('.', end=' ')
            else:
                print('#' if grid[i][j] == 1 else ' ', end=' ')
        print()

def visualize_search_process(grid, exploration_history, final_path=None):
    """Visualize the search process step by step"""
    import time
    import os
    
    for step, explored in enumerate(exploration_history):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"Step {step + 1}:")
        print_solution(grid, explored)
        time.sleep(0.5)
    
    if final_path:
        print("\nFinal solution path:")
        print_solution(grid, final_path)

if __name__ == "__main__":
    # Create a simple maze
    maze = [
        [0, 0, 0, 1, 0],
        ['S', 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 'G'],
        [0, 1, 0, 1, 0]
    ]
    
    problem = AStarSearchProblem(maze)
    solution, history = astar_search(problem)
    
    if solution:
        print("Visualizing search process...")
        visualize_search_process(maze, history, solution)
    else:
        print("No solution found!") 