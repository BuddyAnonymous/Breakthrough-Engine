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
center_table = [0]*64
# assume ring_mask_1..3 are defined bitboards
for sq in range(64):
    mask = 1 << sq
    if mask & ring_mask_1:
        center_table[sq] = W_CENTER * 3
    elif mask & ring_mask_2:
        center_table[sq] = W_CENTER * 2
    elif mask & ring_mask_3:
        center_table[sq] = W_CENTER * 1
    else:
        center_table[sq] = 0

adv_table_white = [0]*64
adv_table_black = [0]*64
for sq in range(64):
    rank = sq // 8
    # original had: add rank, and if rank > 4 some quadratic bonus
    val = rank
    if rank > 4:
        val += (4*(rank - 4))**2
    adv_table_white[sq] = val

    # for black your code subtracts (7 - rank) and if rank < 3 adds penalty
    valb = (7 - rank)
    if rank < 3:
        valb += (4*(3 - rank))**2
    adv_table_black[sq] = valb

def _sum_table_for_bitboard(bb, table):
    s = 0
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        s += table[idx]
        bb ^= lsb
    return s

advancement = {7: 10000000, 6: 10000 , 5:200, 0:10000000, 1:10000, 2:200}

_eval_cache = {}

def evaluate_board_cached(white, black, hash_key=None):
    if hash_key is not None:
        v = _eval_cache.get(hash_key)
        if v is not None:
            return v
    score = evaluate_board(white, black)   # your existing, pure evaluator
    if hash_key is not None:
        _eval_cache[hash_key] = score
    return score

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

def generate_all_white_moves(white, black):
    moves = []
    full = white | black
    for from_sq in bit_scan(white):
        # forward
        to = from_sq + 8
        if to < 64 and not ((full >> to) & 1):
            moves.append((from_sq, to))
        # diag-left
        to = from_sq + 9
        if to < 64 and not ((1 << from_sq) & left_edge) and not ((white >> to) & 1):   # not on right edge
            # allow diagonal move to empty or capture
            moves.append((from_sq, to))
        # diag-right
        to = from_sq + 7
        if to < 64 and not ((1 << from_sq) & right_edge) and not ((white >> to) & 1):    # not on left edge
            moves.append((from_sq, to))
    return moves

def generate_all_black_moves(white,black):
    moves = []
    full = white | black

    for from_sq in bit_scan(black):
        to = from_sq - 8
        if to >= 0 and not ((full >> to) & 1):
            moves.append((from_sq, to))
        to = from_sq - 9
        if to >= 0 and not ((1 << from_sq) & right_edge) and not ((black >> to) & 1):  # not on left edge
            moves.append((from_sq, to))
        to = from_sq - 7
        if to >= 0 and not ((1 << from_sq) & left_edge) and not ((black >> to) & 1):  # not on right edge
            moves.append((from_sq, to))
    return moves

def order_moves(moves, white, black, whiteToMove, pv_move=None, tt_move=None):
    # precompute frequently used lookups
    opp = black if whiteToMove else white
    center_bonus = {27, 28, 35, 36}  # set lookup = O(1)

    def key(move):
        from_sq, to_sq = move
        score = 0

        # 1. Captures
        if (opp >> to_sq) & 1:
            score += 100

        # 2. Advancement
        rank = to_sq // 8
        if whiteToMove:
            score += rank * 5
            if rank > 4:
                score += advancement[rank]
        else:
            score += (7 - rank) * 5
            if rank < 3:
                score += advancement[rank]

        # 3. Center bonus
        if to_sq in center_bonus:
            score += 10

        # 4. PV/TT
        if move == pv_move:
            score += 1000
        elif move == tt_move:
            score += 500

        return score

    return sorted(moves, key=key, reverse=True)

def evaluate_board(white,black):
    #global globalCounter

    #globalCounter += 1
    W_MATERIAL_l = W_MATERIAL
    W_ADVANCE_l = W_ADVANCE
    W_MOB_l = W_MOB
    W_CAPTURE_l = W_CAPTURE
    W_BLOCKED_l = W_BLOCKED
    INF_l = INF
    re = right_edge
    le = left_edge
    adv_w = adv_table_white
    adv_b = adv_table_black
    center_tbl = center_table
    calc_mob_w = calc_mobility_white
    calc_mob_b = calc_mobility_black
    sum_tbl = _sum_table_for_bitboard
    white_goal_rank = white_goal_rank7
    black_goal_rank = black_goal_rank0

    if white & white_goal_rank:
        return INF_l

    if black & black_goal_rank:
        return -INF_l

    #material

    white_count = white.bit_count()
    black_count = black.bit_count()
    material = (white_count - black_count) * W_MATERIAL_l

    #advancement
    white_adv_sum = sum_tbl(white, adv_w)
    black_adv_sum = sum_tbl(black, adv_b)
    advancement = (white_adv_sum - black_adv_sum) * W_ADVANCE_l
    
    #mobility
    white_mobility = calc_mob_w(white,black)
    black_mobility = calc_mob_b(black,white)
    mobility = (white_mobility.bit_count() - black_mobility.bit_count()) * W_MOB_l

    #captures
    w_attacks = (((white << 9) & ~re) | ((white << 7) & ~le))
    b_attacks = (((black >> 9) & ~le)  | ((black >> 7) & ~re))
    capture_score = ( (w_attacks & black).bit_count() - (b_attacks & white).bit_count() ) * W_CAPTURE_l

    #blocked pawns
    pawns_with_moves = ((white_mobility >> 8) | (white_mobility >> 9) | (white_mobility >> 7)) & white
    blocked_pawns = white & ~pawns_with_moves
    num_blocked_white = blocked_pawns.bit_count()

    pawns_with_moves_black = ((black_mobility << 8) | (black_mobility << 9) | (black_mobility << 7)) & black
    blocked_pawns_black = black & ~pawns_with_moves_black
    num_blocked_black = blocked_pawns_black.bit_count()

    blocked_pawns = (num_blocked_white - num_blocked_black) * W_BLOCKED_l

    #center
    center_score = _sum_table_for_bitboard(white, center_tbl) - _sum_table_for_bitboard(black, center_tbl)

    return material + advancement + mobility + capture_score + blocked_pawns + center_score


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

def compute_hash(white_bb, black_bb, whiteToMove):
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
    if not whiteToMove:
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

    eval_fn = evaluate_board_cached   # or evaluate_board
    make_move_local = make_move
    update_hash_local = update_hash
    is_capture_local = is_capture

    stand_pat = eval_fn(white, black)
    
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
            new_white, new_black = make_move_local(white, black, move, is_white=True)
            new_hash = update_hash_local(hash_key, piece=0, sq_from=move[0], sq_to=move[1],
                                   captured_piece=1 if is_capture_local(black, white, move, True) else None,
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
            new_white, new_black = make_move_local(white, black, move, is_white=False)
            new_hash = update_hash_local(hash_key, piece=1, sq_from=move[0], sq_to=move[1],
                                   captured_piece=0 if is_capture_local(black, white, move, False) else None,
                                   side_to_move_changed=True)
            eval = quiescence(new_white, new_black, alpha, beta, True, new_hash)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def minimax(white, black, depth, alpha, beta, maximizingPlayer, pv_move=None, hash_key=None):
    
    tt_local = tt
    EXACT_l, LOWER_l, UPPER_l = EXACT, LOWERBOUND, UPPERBOUND
    make_move_local = make_move
    order_moves_local = order_moves
    generate_white_local = generate_all_white_moves
    generate_black_local = generate_all_black_moves
    update_hash_local = update_hash
    is_capture_local = is_capture

    if hash_key is None:
        hash_key = compute_hash(white, black, maximizingPlayer)

    if hash_key in tt_local:
        entry = tt_local[hash_key]
        if entry["depth"] >= depth:  # only trust if searched deep enough
            if entry["flag"] == EXACT_l:
                return entry["score"], entry["best_move"]
            elif entry["flag"] == LOWER_l and entry["score"] >= beta:
                return entry["score"], entry["best_move"]
            elif entry["flag"] == UPPER_l and entry["score"] <= alpha:
                return entry["score"], entry["best_move"]
    
    if depth == 0 or game_over(white, black):
        score = quiescence(white, black, alpha, beta, maximizingPlayer, hash_key)
        return score, None

    best_move = None

    if maximizingPlayer:  # White to move
        max_eval = -float('inf')
        moves = generate_white_local(white, black)

        tt_move = tt_local[hash_key]["best_move"] if hash_key in tt_local else None
        moves = order_moves_local(moves, white, black, True, pv_move, tt_move)
        for move in moves:
            new_white, new_black = make_move_local(white, black, move, is_white=True)
            new_hash = update_hash_local(hash_key, piece=0, sq_from=move[0], sq_to=move[1],
                                   captured_piece=1 if is_capture_local(black, white, move, True) else None,
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
        moves = generate_black_local(white, black)

        tt_move = tt_local[hash_key]["best_move"] if hash_key in tt_local else None
        moves = order_moves_local(moves, white, black, False, pv_move, tt_move)
        for move in moves:
            new_white, new_black = make_move_local(white, black, move, is_white=False)
            new_hash = update_hash_local(hash_key, piece=1, sq_from=move[0], sq_to=move[1],
                                   captured_piece=0 if is_capture_local(black, white, move, False) else None,
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
        flag = UPPER_l
    elif score >= beta:
        flag = LOWER_l
    else:
        flag = EXACT_l

    tt_local[hash_key] = {
        "depth": depth,
        "score": score,
        "flag": flag,
        "best_move": best_move
    }

    return score, best_move

def iterative_deepening(white, black, max_depth, time_limit=None, whiteToMove=False):
    start_time = time.time()
    best_move = None
    root_hash = compute_hash(white, black, whiteToMove=whiteToMove)

    for depth in range(1, max_depth + 1):
        # Call minimax with alpha-beta pruning
        eval, move = minimax(white, black, depth, -float('inf'), float('inf'), False, pv_move=best_move, hash_key=root_hash)  # Assume black to move first
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
    board_eval, engine_move = iterative_deepening(white_board, black_board, DEPTH, time_limit=100)
    end = time.perf_counter()
    print(end - start, "seconds for engine move calculation")
    white_board, black_board = make_move(white_board, black_board, engine_move, is_white=False)
    print(board_eval, "Engine move:", engine_move)
    print("Evaluated positions: ", globalCounter)
    evalTime = globalCounter = 0
    if game_over(white_board, black_board):
        if white_board & white_goal_rank7:
            print("White wins!")
        elif black_board & black_goal_rank0:
            print("Black wins!")
        break


    
