import random as r
import numpy as np
# Selects random valid column
def agent_random(obs, config):
    valid_moves = [col for col in range(config['columns']) if obs.board[col] == 0]
    return r.choice(valid_moves)

# Selects middle column
def agent_middle(obs, config):
    return config['columns']//2

# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config['columns']) if obs.board[col] == 0]
    return valid_moves[0]


############################


# agent Game Tree (create different tree )

# game board methods
config = {'rows': 6, 'columns': 7, 'winning_count': 4}

# drop a piece to a specific column
def drop_piece(grid, drop_col, piece):
    for row in range(config['rows']-1, -1, -1):
        if grid[row][drop_col] == 0:
            grid[row][drop_col] = piece
            break
    return grid

# Returns True if dropping piece in column results in game win
def check_pieces_in_window(window, check_num, piece):
    # check if the window has enough piece and space
    return (window.count(piece) == check_num and window.count(0) == config['winning_count']-check_num)
    
# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, check_num, piece):
    num_windows = 0
    # horizontal
    for row in range(config['rows']):
        for col in range(config['columns']-(config['winning_count']-1)):
            window = list(grid[row, col:col+config['winning_count']])
            if check_pieces_in_window(window, check_num, piece):
                num_windows += 1
    # vertical
    for row in range(config['rows']-(config['winning_count']-1)):
        for col in range(config['columns']):
            window = list(grid[row:row+config['winning_count'], col])
            if check_pieces_in_window(window, check_num, piece):
                num_windows += 1
    # positive diagonal
    for row in range(config['rows']-(config['winning_count']-1)):
        for col in range(config['columns']-(config['winning_count']-1)):
            window = list(grid[range(row, row+config['winning_count']), range(col, col+config['winning_count'])])
            if check_pieces_in_window(window, check_num, piece):
                num_windows += 1
    # negative diagonal
    for row in range(config['winning_count']-1, config['rows']):
        for col in range(config['columns']-(config['winning_count']-1)):
            window = list(grid[range(row, row-config['winning_count'], -1), range(col, col+config['winning_count'])])
            if check_pieces_in_window(window, check_num, piece):
                num_windows += 1
    return num_windows

def calculate_grid_score(grid, piece):
    num_win_windows = count_windows(grid, config['winning_count'], piece)
    num_almost_win_windows = count_windows(grid, config['winning_count']-1, piece)
    num_win2_windows = count_windows(grid, config['winning_count']-2, piece)
    
    num_almost_lose_windows = count_windows(grid, config['winning_count']-1, piece%2+1)
    num_lose2_windows = count_windows(grid, config['winning_count']-2, piece%2+1)
    num_lose_windows = count_windows(grid, config['winning_count'], piece%2+1)
    score = 200000*num_win_windows + 900*num_almost_win_windows \
            +50*num_win2_windows                               \
            -2000*num_almost_lose_windows-60*num_lose2_windows \
            -150000*num_lose_windows
    return score
    
def find_best_action_rec(grid, drop_col, piece, depth, max_depth):
    new_grid = grid.copy()
    new_grid = drop_piece(new_grid, drop_col, piece)

    opponent_piece = piece%2+1
    is_at_enemy_layer = depth%2==1
    # if we have arrived max depth, return the score
    if depth >= max_depth-1:
        # always calculate score of this agent(piece will filp every time)
        grid_score = calculate_grid_score(new_grid, opponent_piece if is_at_enemy_layer else piece)
        return grid_score
    
    # if not arrived max depth, go deeper
    next_depth = depth + 1
    is_next_enemy_layer = next_depth %2 == 1
    action_scores = [10000000 if is_next_enemy_layer else -10000000]
    # action_scores = []
    for drop_col in range(config['columns']):
        if grid[0][drop_col] != 0: continue # select valid action
            
        # calculate score of this action, and store it
        score = find_best_action_rec(new_grid, drop_col, opponent_piece, next_depth, max_depth)
        action_scores.append(score)

    # print(action_scores)
    return min(action_scores) if is_next_enemy_layer else max(action_scores)
    

def find_best_action(grid, piece, max_depth=2):
    scores = {} # calculate every score of actions
    for drop_col in range(config['columns']):
        if grid[0][drop_col] != 0: continue # if the column if full then see next column
            
        # calculate scores of this action
        scores[drop_col] = find_best_action_rec(grid, drop_col, piece, 0, max_depth)

    # find actions that result in max scores
    max_score = max(scores.values())
    best_moves = [key for key in scores.keys() if scores[key] == max_score]
    
    # Select at random from the maximizing actions
    return r.choice(best_moves)
    
############################




def MinimaxAgent(max_depth): # 展開所有動作的Agent，找出分數最大的可能
    
    def MyAgent(obs, config):
        game_board_grid = np.array(obs.board).reshape(config['rows'], config['columns'])
        return find_best_action(game_board_grid, obs.mark, max_depth)
    
    return MyAgent


from tf_agents.agents import DdpgAgent

def DqnAgent():
    pass

def PPOAgent():
    pass