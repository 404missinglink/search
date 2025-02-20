"""
Tic Tac Toe with Minimax Algorithm and Alpha-Beta Pruning
-------------------------------------------------------
A beginner-friendly implementation of the minimax algorithm with alpha-beta pruning.
This program shows how AI can play perfect Tic Tac Toe by thinking ahead!

What is Alpha-Beta Pruning?
--------------------------
Alpha-beta pruning is like having a smart student doing a test:
- If they know a choice will be worse than one they already found,
  they skip checking that choice completely!

Alpha (α): The best score the maximizing player (X) is guaranteed
Beta (β):  The best score the minimizing player (O) is guaranteed

Example of pruning:
If X is looking at moves and already found a move with score 5,
and then sees that O's next move would give a score of 3 or less,
X can skip checking that path because it's already worse than what we have!

Board representation:
┌───┬───┬───┐
│ X │ O │   │ --> represented as [['X', 'O', ' '],
├───┼───┼───┤     ['O', 'X', ' '],
│ O │ X │   │      ['X', ' ', ' ']]
├───┼───┼───┤
│ X │   │   │
└───┴───┴───┘

Scoring system:
- If X wins: +10 points
- If O wins: -10 points
- If draw:    0 points

The AI (X) tries to get the highest score possible.
The opponent (O) tries to get the lowest score possible.
"""

def create_empty_board():
    """Creates a new empty 3x3 board"""
    return [[' ' for _ in range(3)] for _ in range(3)]

def is_valid_move(board, row, col):
    """Checks if a move is valid (space is empty and within bounds)"""
    if 0 <= row <= 2 and 0 <= col <= 2:
        return board[row][col] == ' '
    return False

def evaluate_board(board):
    """
    Looks at the board and returns a score based on who is winning.
    Think of this as the 'judge' that decides how good a position is.
    
    Returns:
    - +10 if X wins (good for X)
    - -10 if O wins (good for O)
    -   0 if nobody has won yet
    """
    # Check all rows (→) for a win
    for row in board:
        if row == ['X', 'X', 'X']:  # A row of X's
            return 10
        if row == ['O', 'O', 'O']:  # A row of O's
            return -10
    
    # Check all columns (↓) for a win
    for col in range(3):
        if (board[0][col] == board[1][col] == board[2][col] == 'X'):  # A column of X's
            return 10
        if (board[0][col] == board[1][col] == board[2][col] == 'O'):  # A column of O's
            return -10
    
    # Check main diagonal (↘)
    if (board[0][0] == board[1][1] == board[2][2] == 'X' or  # Main diagonal of X's
        board[0][2] == board[1][1] == board[2][0] == 'X'):   # Other diagonal of X's
        return 10
    
    # Check other diagonal (↙)
    if (board[0][0] == board[1][1] == board[2][2] == 'O' or  # Main diagonal of O's
        board[0][2] == board[1][1] == board[2][0] == 'O'):   # Other diagonal of O's
        return -10
    
    return 0  # Nobody has won yet

def is_moves_left(board):
    """
    Checks if there are any empty spaces left on the board.
    This is like checking if the game should continue or end in a draw.
    """
    return any(' ' in row for row in board)

def minimax(board, depth, is_max, alpha=float('-inf'), beta=float('inf'), show_thinking=False):
    """
    The minimax algorithm with alpha-beta pruning: the smarter brain of our AI!
    
    How it thinks:
    1. First, check if someone has won or if it's a draw
    2. If not, try every possible move (but skip bad ones using alpha-beta pruning!)
    3. For each move, think ahead about what the opponent would do
    4. Keep track of the best outcome we can achieve
    
    Alpha-Beta Pruning:
    - Alpha (α): Best score X can guarantee (starts at -∞)
    - Beta (β):  Best score O can guarantee (starts at +∞)
    - If α ≥ β, we can stop looking (one player found a better move elsewhere)
    
    Parameters:
    - board: Current game state
    - depth: How many moves we've looked ahead
    - is_max: True if it's X's turn (AI), False if it's O's turn
    - alpha: Best score X can guarantee so far
    - beta: Best score O can guarantee so far
    - show_thinking: If True, prints out the AI's thinking process
    """
    # First check if someone has already won
    score = evaluate_board(board)
    
    # If X has won, return score (adjusted for how quickly we won)
    if score == 10:
        return score - depth  # Winning sooner is better than later
    
    # If O has won, return score (adjusted for how quickly we lost)
    if score == -10:
        return score + depth  # Losing later is better than sooner
    
    # If no moves left, it's a draw
    if not is_moves_left(board):
        return 0
    
    if show_thinking:
        print_board(board, depth=depth)
        print(f"{'  ' * depth}α = {alpha}, β = {beta}")
    
    # X's turn (AI) - try to maximize score
    if is_max:
        best_score = float('-inf')  # Start with worst possible score
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    # Try the move
                    board[i][j] = 'X'
                    if show_thinking:
                        print(f"{'  ' * depth}AI trying move ({i}, {j})")
                    
                    # Look ahead and see what opponent would do
                    score = minimax(board, depth + 1, False, alpha, beta, show_thinking)
                    
                    # Undo the move
                    board[i][j] = ' '
                    
                    # Update best score and alpha
                    best_score = max(score, best_score)
                    alpha = max(alpha, best_score)
                    
                    if show_thinking:
                        print(f"{'  ' * depth}Move ({i}, {j}) got score {score}")
                        if beta <= alpha:
                            print(f"{'  ' * depth}Pruning! Found better move elsewhere (α={alpha} ≥ β={beta})")
                    
                    # Pruning: If we found a move that's better than what O allows
                    if beta <= alpha:
                        return best_score
        
        return best_score
    
    # O's turn (opponent) - try to minimize score
    else:
        best_score = float('inf')  # Start with worst possible score
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    # Try the move
                    board[i][j] = 'O'
                    if show_thinking:
                        print(f"{'  ' * depth}Opponent trying move ({i}, {j})")
                    
                    # Look ahead and see what AI would do
                    score = minimax(board, depth + 1, True, alpha, beta, show_thinking)
                    
                    # Undo the move
                    board[i][j] = ' '
                    
                    # Update best score and beta
                    best_score = min(score, best_score)
                    beta = min(beta, best_score)
                    
                    if show_thinking:
                        print(f"{'  ' * depth}Move ({i}, {j}) got score {score}")
                        if beta <= alpha:
                            print(f"{'  ' * depth}Pruning! Found better move elsewhere (α={alpha} ≥ β={beta})")
                    
                    # Pruning: If we found a move that's better than what X allows
                    if beta <= alpha:
                        return best_score
        
        return best_score

def find_best_move(board, show_thinking=False):
    """
    Finds the best move for X (the AI) by:
    1. Trying every possible move
    2. Using minimax with alpha-beta pruning to see how good each move is
    3. Picking the move with the best outcome
    
    Returns: (row, col) of the best move
    """
    best_score = float('-inf')
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    print("\nAI is thinking about all possible moves:")
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                # Try this move
                board[i][j] = 'X'
                # See how good this move is
                score = minimax(board, 0, False, alpha, beta, show_thinking)
                # Undo the move
                board[i][j] = ' '
                
                print(f"If AI moves at ({i}, {j}), best outcome would be: {score}")
                
                # Update best move if this is better
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
                    alpha = max(alpha, best_score)
    
    return best_move

def print_board(board, depth=0):
    """
    Prints the board in a nice format with optional indentation
    to show the AI's thinking process.
    """
    indent = '  ' * depth
    print(f"\n{indent}┌───┬───┬───┐")
    for i, row in enumerate(board):
        print(f"{indent}│ {' │ '.join(cell if cell != ' ' else ' ' for cell in row)} │")
        if i < 2:
            print(f"{indent}├───┼───┼───┤")
    print(f"{indent}└───┴───┴───┘")

def play_game():
    """
    Play a game of Tic Tac Toe against the AI!
    You are O, the AI is X.
    """
    board = create_empty_board()
    print("\nWelcome to Tic Tac Toe!")
    print("You are O, the AI is X")
    print("Enter your move as: row column (0-2)")
    
    # AI goes first
    print("\nAI's turn:")
    row, col = find_best_move(board, show_thinking=True)
    board[row][col] = 'X'
    print_board(board)
    
    while is_moves_left(board):
        # Player's turn
        while True:
            try:
                row, col = map(int, input("\nYour turn (row col): ").split())
                if is_valid_move(board, row, col):
                    break
                print("Invalid move! Try again.")
            except (ValueError, IndexError):
                print("Invalid input! Use format: row col (0-2)")
        
        board[row][col] = 'O'
        print_board(board)
        
        # Check if player won
        if evaluate_board(board) == -10:
            print("\nCongratulations! You won!")
            return
        
        # Check if it's a draw
        if not is_moves_left(board):
            print("\nIt's a draw!")
            return
        
        # AI's turn
        print("\nAI's turn:")
        row, col = find_best_move(board, show_thinking=True)
        board[row][col] = 'X'
        print_board(board)
        
        # Check if AI won
        if evaluate_board(board) == 10:
            print("\nAI wins!")
            return
        
        # Check if it's a draw
        if not is_moves_left(board):
            print("\nIt's a draw!")
            return

if __name__ == "__main__":
    play_game()
