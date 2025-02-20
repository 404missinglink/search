def evaluate_board(board):
    """
    Evaluates a tic tac toe board and returns a score.
    Returns: 10 for X win, -10 for O win, 0 for draw/ongoing
    
    Example board:
    board = [
        ['X', 'O', 'X'],
        ['O', 'X', 'O'],
        ['X', ' ', ' ']
    ]
    """
    # Check rows for a win
    for row in board:
        if row.count('X') == 3:
            return 10
        if row.count('O') == 3:
            return -10
    
    # Check columns for a win
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == 'X':
            return 10
        if board[0][col] == board[1][col] == board[2][col] == 'O':
            return -10
    
    # Check diagonals for a win
    if board[0][0] == board[1][1] == board[2][2] == 'X' or \
       board[0][2] == board[1][1] == board[2][0] == 'X':
        return 10
    if board[0][0] == board[1][1] == board[2][2] == 'O' or \
       board[0][2] == board[1][1] == board[2][0] == 'O':
        return -10
    
    # If no winner, return 0
    return 0

def is_moves_left(board):
    """Check if there are empty spaces on the board"""
    return any(' ' in row for row in board)

def minimax(board, depth, is_max):
    """
    The minimax function to find the best move.
    Parameters:
        board: The current game board
        depth: How deep in the game tree we are
        is_max: True if it's maximizing player's turn (X), False for minimizing player (O)
    Returns:
        The best score possible from this position
    """
    # First, check if someone has won or if it's a draw
    score = evaluate_board(board)
    
    # If Maximizer (X) has won
    if score == 10:
        return score - depth  # Prefer winning sooner rather than later
    
    # If Minimizer (O) has won
    if score == -10:
        return score + depth  # Prefer losing later rather than sooner
    
    # If no moves left, it's a draw
    if not is_moves_left(board):
        return 0
    
    # If it's Maximizer's turn
    if is_max:
        best = -1000
        # Try all empty cells
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    # Make the move
                    board[i][j] = 'X'
                    # Recursively compute the best score
                    best = max(best, minimax(board, depth + 1, False))
                    # Undo the move
                    board[i][j] = ' '
        return best
    
    # If it's Minimizer's turn
    else:
        best = 1000
        # Try all empty cells
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    # Make the move
                    board[i][j] = 'O'
                    # Recursively compute the best score
                    best = min(best, minimax(board, depth + 1, True))
                    # Undo the move
                    board[i][j] = ' '
        return best

def find_best_move(board):
    """
    Find the best move for X (maximizing player)
    Returns: (row, col) of the best move
    """
    best_score = -1000
    best_move = (-1, -1)
    
    # Try all possible moves
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                # Make the move
                board[i][j] = 'X'
                # Compute score for this move
                move_score = minimax(board, 0, False)
                # Undo the move
                board[i][j] = ' '
                
                # Update best_move if this move is better
                if move_score > best_score:
                    best_move = (i, j)
                    best_score = move_score
    
    return best_move

# Example usage
if __name__ == "__main__":
    # Example board where X needs to make a move
    example_board = [
        ['X', 'O', ' '],
        [' ', 'X', ' '],
        [' ', 'O', ' ']
    ]
    
    # Find and make the best move
    row, col = find_best_move(example_board)
    print(f"Best move is: row {row}, col {col}")
    example_board[row][col] = 'X'
    
    # Print the resulting board
    for row in example_board:
        print(row)
