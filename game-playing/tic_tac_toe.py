"""
Tic Tac Toe with Minimax Algorithm
---------------------------------
This is a simple implementation of the minimax algorithm for Tic Tac Toe.
The board is represented as a 3x3 grid where:
- 'X' is the maximizing player (trying to get highest score)
- 'O' is the minimizing player (trying to get lowest score)
- ' ' (space) represents empty squares

The scoring system is simple:
- +10 points if X wins
- -10 points if O wins
- 0 points for a draw
"""

def evaluate_board(board):
    """
    Looks at the board and returns a score based on who is winning:
    - Returns +10 if X wins (good for X)
    - Returns -10 if O wins (good for O)
    - Returns 0 if nobody has won yet
    """
    # Check all rows (→) for a win
    for row in board:
        if row.count('X') == 3:  # If row has three X's
            return 10
        if row.count('O') == 3:  # If row has three O's
            return -10
    
    # Check all columns (↓) for a win
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == 'X':  # If column has three X's
            return 10
        if board[0][col] == board[1][col] == board[2][col] == 'O':  # If column has three O's
            return -10
    
    # Check diagonal (↘) and other diagonal (↙) for a win
    if board[0][0] == board[1][1] == board[2][2] == 'X' or \
       board[0][2] == board[1][1] == board[2][0] == 'X':  # If either diagonal has three X's
        return 10
    if board[0][0] == board[1][1] == board[2][2] == 'O' or \
       board[0][2] == board[1][1] == board[2][0] == 'O':  # If either diagonal has three O's
        return -10
    
    return 0  # Nobody has won yet

def is_moves_left(board):
    """
    Checks if there are any empty spaces left on the board.
    Returns True if there are empty spaces, False if the board is full.
    """
    # Look through each row for any empty space
    for row in board:
        if ' ' in row:  # If we find an empty space
            return True
    return False  # No empty spaces found

def minimax(board, depth, is_max):
    """
    The minimax algorithm: thinks ahead about all possible moves!
    
    How it works:
    1. If X wins: return +10 (good for X)
    2. If O wins: return -10 (good for O)
    3. If nobody can move: return 0 (it's a draw)
    4. Otherwise:
       - If it's X's turn: try all moves and pick the highest score
       - If it's O's turn: try all moves and pick the lowest score
    
    Parameters:
    - board: The current game board
    - depth: How many moves we've looked ahead (used to prefer winning sooner)
    - is_max: True if it's X's turn, False if it's O's turn
    """
    # First check if someone has already won
    score = evaluate_board(board)
    
    # X has won! The sooner we win, the better (that's why we subtract depth)
    if score == 10:
        return score - depth
    
    # O has won! The later we lose, the better (that's why we add depth)
    if score == -10:
        return score + depth
    
    # It's a draw - no more moves and nobody has won
    if not is_moves_left(board):
        return 0
    
    # It's X's turn (maximizing player)
    if is_max:
        best_score = -1000  # Start with a very low score
        # Try every possible move
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':  # Found an empty spot
                    # Try this move
                    board[i][j] = 'X'
                    # See what would happen if we make this move
                    score = minimax(board, depth + 1, False)
                    # Undo the move
                    board[i][j] = ' '
                    # Keep track of the best score we've seen
                    best_score = max(score, best_score)
        return best_score
    
    # It's O's turn (minimizing player)
    else:
        best_score = 1000  # Start with a very high score
        # Try every possible move
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':  # Found an empty spot
                    # Try this move
                    board[i][j] = 'O'
                    # See what would happen if we make this move
                    score = minimax(board, depth + 1, True)
                    # Undo the move
                    board[i][j] = ' '
                    # Keep track of the best score we've seen
                    best_score = min(score, best_score)
        return best_score

def find_best_move(board):
    """
    Finds the best possible move for X by:
    1. Trying every empty spot on the board
    2. Using minimax to see how good each move is
    3. Picking the move with the highest score
    
    Returns: (row, col) of the best move
    """
    best_score = -1000  # Start with a very low score
    best_move = (-1, -1)  # Keep track of the best move's position
    
    # Try every spot on the board
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':  # Found an empty spot
                # Try making a move here
                board[i][j] = 'X'
                # See how good this move is
                move_score = minimax(board, 0, False)
                # Undo the move
                board[i][j] = ' '
                
                # If this move is better than our best so far, remember it
                if move_score > best_score:
                    best_move = (i, j)
                    best_score = move_score
    
    return best_move

def print_board(board):
    """
    Prints the board in a nice format:
    X | O | X
    ---------
    O | X | O
    ---------
    X |   |  
    """
    for i, row in enumerate(board):
        print(' | '.join(cell if cell != ' ' else ' ' for cell in row))
        if i < 2:  # Don't print the line after the last row
            print('---------')

# Example of how to use the code
if __name__ == "__main__":
    # Start with a board where X needs to make a move
    example_board = [
        ['X', 'O', ' '],
        [' ', 'X', ' '],
        [' ', 'O', ' ']
    ]
    
    print("Starting board:")
    print_board(example_board)
    print("\nThinking about the best move...")
    
    # Find and make the best move
    row, col = find_best_move(example_board)
    example_board[row][col] = 'X'
    
    print(f"\nBest move is: row {row}, column {col}")
    print("\nResulting board:")
    print_board(example_board)
