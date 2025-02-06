class SimpleSearchProblem:
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
        
        # Find start position
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
            (x+1, y), # down
            (x-1, y), # up
            (x, y+1), # right
            (x, y-1)  # left
        ]
        
        valid_moves = []
        for next_x, next_y in possible_moves:
            # Check if move is within grid and not a wall
            if (0 <= next_x < self.rows and 
                0 <= next_y < self.cols and 
                self.grid[next_x][next_y] != 1):
                valid_moves.append((next_x, next_y))
        
        return valid_moves

    def is_goal(self, state):
        """Check if current state is goal state"""
        return state == self.goal

def breadth_first_search(problem):
    """
    Perform breadth-first search and return path from start to goal
    Also returns exploration history for visualization
    """
    # Queue of (state, path) pairs
    queue = [(problem.start, [problem.start])]
    # Set of visited states
    visited = {problem.start}
    # Keep track of exploration history
    exploration_history = []
    
    while queue:
        state, path = queue.pop(0)
        exploration_history.append(set(path))  # Add current path to history
        
        if problem.is_goal(state):
            return path, exploration_history
            
        for next_state in problem.get_successors(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    
    return None, exploration_history  # No path found

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

# Update the main block
if __name__ == "__main__":
    # Create a simple maze
    # 0 = empty space, 1 = wall, 'S' = start, 'G' = goal
    maze = [
        [0, 0, 0, 1, 0],
        ['S', 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 'G'],
        [0, 1, 0, 1, 0]
    ]
    
    # Create problem instance
    problem = SimpleSearchProblem(maze)
    
    # Find solution
    solution, exploration_history = breadth_first_search(problem)
    
    if solution:
        print("Visualizing search process...")
        visualize_search_process(maze, exploration_history, solution)
    else:
        print("No solution found!")
