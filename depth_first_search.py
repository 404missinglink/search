class DepthFirstSearchProblem:
    def __init__(self, grid):
        """
        Initialize search problem with a grid where:
        0 = empty space
        1 = wall
        'S' = start
        'G' = goal
        """
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
        """Returns possible next states from current state"""
        x, y = state
        possible_moves = [
            (x+1, y),  # down
            (x, y+1),  # right
            (x-1, y),  # up
            (x, y-1)   # left
        ]
        
        return [(next_x, next_y) for next_x, next_y in possible_moves 
                if (0 <= next_x < self.rows and 
                    0 <= next_y < self.cols and 
                    self.grid[next_x][next_y] != 1)]

    def is_goal(self, state):
        """Check if current state is goal state"""
        return state == self.goal

def depth_first_search(problem):
    """
    Perform depth-first search and return path from start to goal
    Also returns exploration history for visualization
    """
    stack = [(problem.start, [problem.start])]
    visited = {problem.start}
    exploration_history = []
    
    while stack:
        state, path = stack.pop()
        exploration_history.append(set(path))
        
        if problem.is_goal(state):
            return path, exploration_history
            
        for next_state in reversed(problem.get_successors(state)):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [next_state]))
    
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
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear console
        print(f"Step {step + 1}:")
        print_solution(grid, explored)
        time.sleep(0.5)  # Pause to show each step
    
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
    
    problem = DepthFirstSearchProblem(maze)
    solution, history = depth_first_search(problem)
    
    if solution:
        print("Visualizing search process...")
        visualize_search_process(maze, history, solution)
    else:
        print("No solution found!") 