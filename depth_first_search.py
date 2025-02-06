class DepthFirstSearchProblem:
    def __init__(self, grid):
        """
        Initialize search problem with a grid where:
        0 = empty space
        1 = wall
        'S' = start
        'G' = goal
        
        This is the constructor method that runs when we create a new instance of this class.
        It sets up the initial state of our maze-solving problem.
        """
        # Store the 2D grid/maze as instance variable
        # A 2D grid is a list of lists, where each inner list represents a row
        self.grid = grid
        
        # Calculate dimensions of the grid
        # len(grid) gives number of rows (outer list length)
        # len(grid[0]) gives number of columns (length of first row)
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Iterate through the grid to find start ('S') and goal ('G') positions
        # We use nested loops to check every cell in the 2D grid
        # Positions are stored as tuples of (row, column)
        for i in range(self.rows):  # Loop through each row
            for j in range(self.cols):  # Loop through each column in current row
                if grid[i][j] == 'S':
                    self.start = (i, j)  # Store start position as tuple
                elif grid[i][j] == 'G':
                    self.goal = (i, j)   # Store goal position as tuple

    def get_successors(self, state):
        """
        Returns possible next states (positions) from current state.
        A successor is a valid position that can be moved to from the current position.
        
        Parameters:
            state: tuple (x, y) representing current position in grid
        Returns:
            list of tuples representing valid positions we can move to
        """
        # Unpack the current position coordinates from the state tuple
        x, y = state
        
        # Define all possible moves in 4 directions
        # Each move is represented as a tuple (new_x, new_y)
        possible_moves = [
            (x+1, y),  # down  - increase row by 1
            (x, y+1),  # right - increase column by 1
            (x-1, y),  # up    - decrease row by 1
            (x, y-1)   # left  - decrease column by 1
        ]
        
        # List comprehension to filter only valid moves
        # A move is valid if:
        # 1. It's within the grid boundaries
        # 2. It's not a wall (value != 1)
        return [(next_x, next_y) for next_x, next_y in possible_moves 
                if (0 <= next_x < self.rows and    # Check if within row bounds
                    0 <= next_y < self.cols and    # Check if within column bounds
                    self.grid[next_x][next_y] != 1)]  # Check if not a wall

    def is_goal(self, state):
        """
        Check if current state (position) is the goal state
        
        Parameters:
            state: tuple (x, y) representing current position
        Returns:
            boolean: True if current position is goal, False otherwise
        """
        return state == self.goal

def depth_first_search(problem):
    """
    Perform depth-first search and return path from start to goal.
    Also returns exploration history for visualization.
    Uses a stack (LIFO - Last In, First Out) for frontier management.
    
    DFS explores as far as possible along each branch before backtracking.
    Think of it like exploring a maze by always following the right wall.
    
    Parameters:
        problem: DepthFirstSearchProblem instance containing the maze
    Returns:
        tuple: (solution_path, exploration_history) if solution found
               (None, exploration_history) if no solution exists
    """
    # Initialize stack with start state and its path
    # Python list can be used as a stack: append() to push, pop() to pop
    # Each stack element is a tuple: (current_position, path_to_current_position)
    stack = [(problem.start, [problem.start])]
    
    # Keep track of visited positions using a set for O(1) lookup time
    # Sets in Python are like lists but with no duplicates allowed
    visited = {problem.start}
    
    # Store history of exploration for visualization
    # Each element represents the set of cells explored at that step
    exploration_history = []
    
    # Continue while there are positions to explore
    while stack:  
        # Pop the last element from stack (LIFO principle)
        # This removes and returns the most recently added element
        state, path = stack.pop()
        
        # Add current path to exploration history for visualization
        exploration_history.append(set(path))
        
        # Check if we've reached the goal
        if problem.is_goal(state):
            return path, exploration_history
        
        # Get all possible next positions
        # Reverse them to explore right-to-left, bottom-to-top
        # This affects the search pattern but not the correctness
        for next_state in reversed(problem.get_successors(state)):
            # Only explore unvisited positions to avoid cycles
            if next_state not in visited:
                visited.add(next_state)
                # Add new position and its path to stack
                # path + [next_state] creates a new list with the next state appended
                stack.append((next_state, path + [next_state]))
    
    # Return None if no path to goal is found
    return None, exploration_history

def print_solution(grid, path, explored=None):
    """
    Print the grid with:
    '*' for solution path
    '.' for explored cells (optional)
    'S' for start position
    'G' for goal position
    '#' for walls
    ' ' for empty spaces
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # Keep start and goal markers as is
            if grid[i][j] in ['S', 'G']:
                print(grid[i][j], end=' ')
            # Mark solution path with asterisks
            elif (i, j) in path:
                print('*', end=' ')
            # Mark explored cells with dots (if provided)
            elif explored and (i, j) in explored:
                print('.', end=' ')
            # Show walls as # and empty spaces as spaces
            else:
                print('#' if grid[i][j] == 1 else ' ', end=' ')
        print()  # New line after each row

def visualize_search_process(grid, exploration_history, final_path=None):
    """
    Visualize the search process step by step with time delays
    Shows how the algorithm explores the maze
    """
    import time
    import os
    
    for step, explored in enumerate(exploration_history):
        # Clear console screen (works on both Unix and Windows)
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"Step {step + 1}:")
        print_solution(grid, explored)
        time.sleep(0.5)  # Pause for 0.5 seconds between steps
    
    # Show final solution path if one was found
    if final_path:
        print("\nFinal solution path:")
        print_solution(grid, final_path)

if __name__ == "__main__":
    # Create a sample maze for testing
    # 0 = empty space, 1 = wall, 'S' = start, 'G' = goal
    maze = [
        [0, 0, 0, 1, 0],
        ['S', 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 'G'],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    
    # Create problem instance and solve it
    problem = DepthFirstSearchProblem(maze)
    solution, history = depth_first_search(problem)
    
    # Display results
    if solution:
        print("Visualizing search process...")
        visualize_search_process(maze, history, solution)
    else:
        print("No solution found!") 