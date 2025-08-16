import time
import random

random.seed(12345)
zobrist = [[random.getrandbits(64) for _ in range(64)] for _ in range(2)]
zobrist_side = random.getrandbits(64)

tt = {}
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

above_rank_3_white = 0xFFFFFF0000000000
above_rank_3_black = 0x0000000000FFFFFF
white_board = 0x000000000000FFFF
#white_board = 0x0000000816588000
#black_board = 0x002E4C1000000000
black_board = 0xFFFF000000000000
right_edge = 0x0101010101010101
left_edge = 0x8080808080808080
white_goal_rank7 = 0xFF00000000000000
black_goal_rank0 = 0x00000000000000FF
ring_mask_1 = 0x0000001818000000
ring_mask_2 = 0x00003C24243C0000
ring_mask_3 = 0x007E424242427E00
ring_mask_4 = 0xFF818181818181FF
MASK64 = 0xFFFFFFFFFFFFFFFF  # 64-bit mask
DEPTH = 14
evalTime = 0
globalCounter = 0

advancement = {7: 10000000, 6: 10000 , 5:200, 0:10000000, 1:10000, 2:200}

def print_board(white, black):
    print("  +-----------------+")
    for row in range(7, -1, -1):  # top row first
        line = str(row + 1) + " | "
        for col in range(8):
            # flip column because 0 is bottom-right
            sq = row * 8 + (7 - col)
            if (white >> sq) & 1:
                line += "W "
            elif (black >> sq) & 1:
                line += "B "
            else:
                line += ". "
        line += "|"
        print(line)
    print("  +-----------------+")
    print("    A B C D E F G H")  # column indices

def bit_scan(bb):
    while bb:
        lsb = bb & -bb                 # isolate the least significant 1
        sq = lsb.bit_length() - 1      # get its index
        yield sq
        bb &= bb - 1 

def calc_mobility_white(white,black):
    forward_moves = (white << 8) & ~(white | black) & MASK64
    forward_left_moves = (white << 9) & ~(white | right_edge) & MASK64
    forward_right_moves = (white << 7) & ~(white | left_edge) & MASK64
    return forward_moves | forward_left_moves | forward_right_moves

def calc_mobility_black(black,white):
    down_moves = (black >> 8) & ~(white | black) & MASK64
    down_left_moves = (black >> 7) & ~(black | right_edge) & MASK64
    down_right_moves = (black >> 9) & ~(black | left_edge) & MASK64
    return down_moves | down_left_moves | down_right_moves

def generate_all_white_moves(white,black):
    forward_moves = (white << 8) & ~(white | black) & MASK64
    forward_left_moves = (white << 9) & ~(white | right_edge) & MASK64
    forward_right_moves = (white << 7) & ~(white | left_edge) & MASK64
    all_white_moves = forward_moves | forward_left_moves | forward_right_moves
    all_moves = []

    for to_sq in bit_scan(all_white_moves):
        from_candidates = []
        if to_sq > 7 and ((1 << (to_sq - 8)) & white) and not ((1 << to_sq) & (white | black)):
            all_moves.append((to_sq - 8, to_sq))
        if to_sq > 8 and ((1 << (to_sq - 9)) & white & ~left_edge):
            all_moves.append((to_sq - 9, to_sq))
        if to_sq > 6 and ((1 << (to_sq - 7)) & white & ~right_edge):
            all_moves.append((to_sq - 7, to_sq))
        for from_sq in from_candidates:
            all_moves.append((from_sq, to_sq))

    return all_moves

def generate_all_black_moves(white,black):
    down_moves = (black >> 8) & ~(white | black) & MASK64
    down_left_moves = (black >> 7) & ~(black | right_edge) & MASK64
    down_right_moves = (black >> 9) & ~(black | left_edge) & MASK64
    all_black_moves = down_moves | down_left_moves | down_right_moves
    all_moves = []

    for to_sq in bit_scan(all_black_moves):
        from_candidates = []
        if to_sq < 56 and ((1 << (to_sq + 8)) & black) and not ((1 << to_sq) & (white | black)):
            all_moves.append((to_sq + 8, to_sq))
        if to_sq < 55 and ((1 << (to_sq + 9)) & black & ~right_edge):
            all_moves.append((to_sq + 9, to_sq))
        if to_sq < 57 and ((1 << (to_sq + 7)) & black & ~left_edge):
            all_moves.append((to_sq + 7, to_sq))
        for from_sq in from_candidates:
            all_moves.append((from_sq, to_sq))

    return all_moves

def order_moves(moves, white, black, whiteToMove, pv_move=None, tt_move = None):
    def move_score(move):
        from_sq, to_sq = move
        score = 0

        # 1. Captures get high priority
        if (whiteToMove and (1 << to_sq) & black) or (not whiteToMove and (1 << to_sq) & white):
            score += 100

        # 2. Advance more = better
        if whiteToMove:
            score += (to_sq // 8) * 5  # deeper ranks = more points
            if to_sq // 8 > 4:
                score += advancement[to_sq // 8]  # bonus for advancing to rank 5 or higher
        else:
            score += ((7 - to_sq // 8) * 5)  # deeper ranks = more points
            if to_sq // 8 < 3:
                score += advancement[to_sq // 8]

        # 3. Central squares = better
        center_bonus = [27, 28, 35, 36]  # or bigger mask
        if to_sq in center_bonus:
            score += 10
        
        if pv_move and move == pv_move:
            score += 1000

        if tt_move and move == tt_move:
            score += 500

        return score

    return sorted(moves, key=move_score, reverse=True)

def evaluate_board(white,black):
    global globalCounter
    global evalTime
    start = time.perf_counter()

    globalCounter += 1

    W_MATERIAL = 100
    W_ADVANCE  = 20
    W_MOB      = 5
    W_CAPTURE  = 50
    #W_PASSED   = 60
    #W_PASSED_DIST = 12
    W_SUPPORT  = 8
    W_CONNECTED = 8
    W_BLOCKED  = -8
    W_CENTER   = 3
    INF = 10**9

    if white & white_goal_rank7:
        return INF

    if black & black_goal_rank0:
        return -INF

    #material

    white_count = white.bit_count()
    black_count = black.bit_count()
    material = (white_count - black_count) * W_MATERIAL

    #advancement
    white_advancement = 0
    for index in bit_scan(white):
        white_advancement += (index // 8)
        if index // 8 > 4:
            white_advancement += (4*(index//8 - 4))**2
    for index in bit_scan(black):
        white_advancement -= (7 - index // 8)
        if index // 8 < 3:
            white_advancement -= (4*(3 - index // 8))**2

    advancement = white_advancement * W_ADVANCE
    
    #mobility
    white_mobility = calc_mobility_white(white,black).bit_count()
    black_mobility = calc_mobility_black(black,white).bit_count()
    mobility = (white_mobility - black_mobility) * W_MOB

    #captures
    w_attacks = (((white << 9) & ~right_edge) | ((white << 7) & ~left_edge))
    b_attacks = (((black >> 9) & ~left_edge)  | ((black >> 7) & ~right_edge))
    w_captures = (w_attacks & black).bit_count()
    b_captures = (b_attacks & white).bit_count()
    capture_score = (w_captures - b_captures) * W_CAPTURE

    #blocked pawns
    all_moves_white = calc_mobility_white(white, black)
    pawns_with_moves = ((all_moves_white >> 8) | (all_moves_white >> 9) | (all_moves_white >> 7)) & white
    blocked_pawns = white & ~pawns_with_moves
    num_blocked_white = blocked_pawns.bit_count()

    all_moves_black = calc_mobility_black(black, white)
    pawns_with_moves_black = ((all_moves_black << 8) | (all_moves_black << 9) | (all_moves_black << 7)) & black
    blocked_pawns_black = black & ~pawns_with_moves_black
    num_blocked_black = blocked_pawns_black.bit_count()

    blocked_pawns = (num_blocked_white - num_blocked_black) * W_BLOCKED

    #center
    center = ((ring_mask_1 & white).bit_count() - (ring_mask_1 & black).bit_count()) * W_CENTER * 3
    center += ((ring_mask_2 & white).bit_count() - (ring_mask_2 & black).bit_count()) * W_CENTER * 2
    center += ((ring_mask_3 & white).bit_count() - (ring_mask_3 & black).bit_count()) * W_CENTER * 1


    end = time.perf_counter()
    evalTime += end - start
    return material + advancement + mobility + capture_score + blocked_pawns + center


def make_move(white, black, move, is_white):
    from_sq, to_sq = move
    from_mask = 1 << from_sq
    to_mask   = 1 << to_sq

    if is_white:
        # Remove pawn from original square
        new_white = white & ~from_mask
        # Move to destination (captures automatically)
        new_white |= to_mask
        # Remove any black piece on the destination
        new_black = black & ~to_mask
    else:
        new_black = black & ~from_mask
        new_black |= to_mask
        new_white = white & ~to_mask

    return new_white, new_black

def game_over(white,black):
    return white & white_goal_rank7 or black & black_goal_rank0

def compute_hash(white_bb, black_bb, whiteToPlay):
    h = 0
    # White pieces
    for square in range(64):
        if (white_bb >> square) & 1:
            h ^= zobrist[0][square]
    # Black pieces
    for square in range(64):
        if (black_bb >> square) & 1:
            h ^= zobrist[1][square]
    # Side to move
    if not whiteToPlay:
        h ^= zobrist_side
    return h

def update_hash(h, piece, sq_from, sq_to, captured_piece=None, side_to_move_changed=True):
    # Remove moving piece from old square
    h ^= zobrist[piece][sq_from]
    
    # If a piece was captured, remove it from hash
    if captured_piece is not None:
        h ^= zobrist[captured_piece][sq_to]
    
    # Add moving piece at new square
    h ^= zobrist[piece][sq_to]
    
    # Toggle side-to-move
    if side_to_move_changed:
        h ^= zobrist_side
    
    return h

def is_capture(black, white, move, whiteToMove):
    from_sq, to_sq = move
    if whiteToMove:
        return (black & (1 << to_sq)) != 0
    else:
        return (white & (1 << to_sq)) != 0

def generate_dangerous_moves(white, black, maximizingPlayer):
    moves = []

    if maximizingPlayer:  # White to move
        # left-diagonal captures
        left_moves = (white << 9) & ~right_edge
        right_moves = (white << 7) & ~left_edge
        forward_moves = (white << 8)
        left_targets = left_moves & black
        # right-diagonal captures
        right_targets = right_moves & black
        for to_sq in bit_scan(left_targets):
            from_sq = to_sq - 9
            moves.append((from_sq, to_sq))

        for to_sq in bit_scan(right_targets):
            from_sq = to_sq - 7
            moves.append((from_sq, to_sq))
        for to_sq in bit_scan(forward_moves & ~white & ~black & above_rank_3_white):
            from_sq = to_sq - 8
            moves.append((from_sq, to_sq))
        for to_sq in bit_scan(left_moves & ~white & ~black & above_rank_3_white):
            from_sq = to_sq - 9
            moves.append((from_sq, to_sq))
        for to_sq in bit_scan(right_moves & ~white & ~black & above_rank_3_white):
            from_sq = to_sq - 7
            moves.append((from_sq, to_sq))
    else:  # Black to move
        # left-diagonal captures
        left_moves = (black >> 7) & ~right_edge
        right_moves = (black >> 9) & ~left_edge
        forward_moves = (black >> 8)
        left_targets = left_moves & white
        # right-diagonal captures
        right_targets = right_moves & white
        for to_sq in bit_scan(left_targets):
            from_sq = to_sq + 7
            moves.append((from_sq, to_sq))

        for to_sq in bit_scan(right_targets):
            from_sq = to_sq + 9
            moves.append((from_sq, to_sq))

        for to_sq in bit_scan(forward_moves & ~white & ~black & above_rank_3_black):
            from_sq = to_sq + 8
            moves.append((from_sq, to_sq))
        
        for to_sq in bit_scan(left_moves & ~white & ~black & above_rank_3_black):
            from_sq = to_sq + 7
            moves.append((from_sq, to_sq))
        
        for to_sq in bit_scan(right_moves & ~white & ~black & above_rank_3_black):
            from_sq = to_sq + 9
            moves.append((from_sq, to_sq))

    return moves

def quiescence(white, black, alpha, beta, maximizingPlayer, hash_key):
    stand_pat = evaluate_board(white, black)
    
    if maximizingPlayer:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    # Generate only capture moves
    moves = generate_dangerous_moves(white, black, maximizingPlayer)

    if not moves:
        return stand_pat  # no more captures

    if maximizingPlayer:
        max_eval = alpha
        for move in moves:
            new_white, new_black = make_move(white, black, move, is_white=True)
            new_hash = update_hash(hash_key, piece=0, sq_from=move[0], sq_to=move[1],
                                   captured_piece=1 if is_capture(black, white, move, True) else None,
                                   side_to_move_changed=True)
            eval = quiescence(new_white, new_black, alpha, beta, False, new_hash)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = beta
        for move in moves:
            new_white, new_black = make_move(white, black, move, is_white=False)
            new_hash = update_hash(hash_key, piece=1, sq_from=move[0], sq_to=move[1],
                                   captured_piece=0 if is_capture(black, white, move, False) else None,
                                   side_to_move_changed=True)
            eval = quiescence(new_white, new_black, alpha, beta, True, new_hash)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def minimax(white, black, depth, alpha, beta, maximizingPlayer, pv_move=None, hash_key=None):
    
    if hash_key is None:
        hash_key = compute_hash(white, black, maximizingPlayer)

    if hash_key in tt:
        entry = tt[hash_key]
        if entry["depth"] >= depth:  # only trust if searched deep enough
            if entry["flag"] == EXACT:
                return entry["score"], entry["best_move"]
            elif entry["flag"] == LOWERBOUND and entry["score"] >= beta:
                return entry["score"], entry["best_move"]
            elif entry["flag"] == UPPERBOUND and entry["score"] <= alpha:
                return entry["score"], entry["best_move"]
    
    if depth == 0 or game_over(white, black):
        score = quiescence(white, black, alpha, beta, maximizingPlayer, hash_key)
        return score, None

    best_move = None

    if maximizingPlayer:  # White to move
        max_eval = -float('inf')
        moves = generate_all_white_moves(white, black)

        tt_move = tt[hash_key]["best_move"] if hash_key in tt else None
        moves = order_moves(moves, white, black, True, pv_move, tt_move)
        for move in moves:
            new_white, new_black = make_move(white, black, move, is_white=True)
            new_hash = update_hash(hash_key, piece=0, sq_from=move[0], sq_to=move[1],
                                   captured_piece=1 if is_capture(black, white, move, True) else None,
                                   side_to_move_changed=True)
            eval, _ = minimax(new_white, new_black, depth-1, alpha, beta, False, pv_move, new_hash)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # beta cutoff
        score = max_eval
    else:  # Black to move
        min_eval = float('inf')
        moves = generate_all_black_moves(white, black)

        tt_move = tt[hash_key]["best_move"] if hash_key in tt else None
        moves = order_moves(moves, white, black, False, pv_move, tt_move)
        for move in moves:
            new_white, new_black = make_move(white, black, move, is_white=False)
            new_hash = update_hash(hash_key, piece=1, sq_from=move[0], sq_to=move[1],
                                   captured_piece=0 if is_capture(black, white, move, False) else None,
                                   side_to_move_changed=True)
            eval, _ = minimax(new_white, new_black, depth-1, alpha, beta, True, pv_move, new_hash)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # alpha cutoff
        score = min_eval
    
    if score <= alpha: 
        flag = UPPERBOUND
    elif score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt[hash_key] = {
        "depth": depth,
        "score": score,
        "flag": flag,
        "best_move": best_move
    }

    return score, best_move
    
def iterative_deepening(white, black, max_depth, time_limit=None):
    start_time = time.time()
    best_move = None

    for depth in range(1, max_depth + 1):
        # Call minimax with alpha-beta pruning
        eval, move = minimax(white, black, depth, -float('inf'), float('inf'), False, pv_move=best_move)  # Assume black to move first
        if move is not None:
            best_move = move

        # Time cutoff check
        if time_limit and (time.time() - start_time) >= time_limit:
            print(f"Time limit reached at depth {depth}. Time {time.time() - start_time:.2f} seconds")
            break
        if eval > 1000000:
            print(f"Early exit at depth {depth} with evaluation {eval}, time {time.time() - start_time:.2f} seconds")
            break

    return eval, best_move


while True:
    print_board(white_board, black_board)

    # Get player input
    move_input = input("Enter your move (from_col from_row to_col to_row): ")
    try:
        move_from = move_input.strip().split()[0]
        move_to = move_input.strip().split()[1]
        fc = ord(move_from[0]) - ord('A')
        fr = int(move_from[1]) - 1
        tc = ord(move_to[0]) - ord('A')
        tr = int(move_to[1]) - 1
    except:
        print("Invalid input format!")
        continue

    from_sq = fr * 8 + (7 - fc)  # adjust for bitboard layout
    to_sq = tr * 8 + (7 - tc)

    move = (from_sq, to_sq)

    # Check if move is legal
    legal_moves = generate_all_white_moves(white_board, black_board)
    if move not in legal_moves:
        print("Illegal move! Try again.")
        continue

    # Make the move
    white_board, black_board = make_move(white_board, black_board, move, is_white=True)
    print("Move played!")
    if game_over(white_board, black_board):
        if white_board & white_goal_rank7:
            print("White wins!")
        elif black_board & black_goal_rank0:
            print("Black wins!")
        break
    start = time.perf_counter()
    board_eval, engine_move = iterative_deepening(white_board, black_board, DEPTH, time_limit=1)
    end = time.perf_counter()
    print(end - start, "seconds for engine move calculation")
    white_board, black_board = make_move(white_board, black_board, engine_move, is_white=False)
    print(board_eval, "Engine move:", engine_move)
    print(globalCounter, evalTime)
    evalTime = globalCounter = 0
    if game_over(white_board, black_board):
        if white_board & white_goal_rank7:
            print("White wins!")
        elif black_board & black_goal_rank0:
            print("Black wins!")
        break


    
