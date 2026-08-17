#!/usr/bin/env python3
"""
chess_ai.py — a chess opponent that learns from the person it plays.

Standalone: nothing here imports the server, so it can be driven from a REPL,
a test harness, or the HomeGames Chess engine class alike.

    from chess_ai import ChessAI
    ai = ChessAI(name='Rook', opponent='Robert')
    uci = ai.get_move(board)          # a chess.Board, a FEN, or {'fen': ...}
    ...
    ai.learn_from_game('loss')        # folds the game back into the brain
    ai.save_progress()

Run it directly for a self-test and a short self-play demo:

    python chess_ai.py --test
    python chess_ai.py --selfplay 20

--------------------------------------------------------------------------
WHAT ACTUALLY LEARNS, AND HOW FAST
--------------------------------------------------------------------------
Worth being straight about this up front, because "learns from your games"
invites an expectation that a win/loss bit per game can't meet. One finished
game is one number. A chess game is ~40 decisions. Crediting all forty for a
loss teaches the AI almost nothing about which of them was the mistake, and
that is the whole difficulty.

This file learns on three separate clocks, listed fastest first:

  1. The opening book — pays off within a handful of games.
     Early positions repeat. You have a few pet openings, and after a couple
     of games the AI has seen them enough to know which replies went badly
     *against you specifically*. This is the part you will actually notice,
     and it is the honest reading of "learns from their patterns".

  2. Move-level credit — pays off over tens of games.
     Rather than smearing one outcome over forty moves, each of the AI's own
     moves is scored by what the position did two plies later. That turns one
     game into ~40 training signals instead of 1. Standard TD(0) with a linear
     evaluator; no ML library, about fifteen lines in _learn_weights().

  3. Evaluation weights — pays off over hundreds of games.
     The piece values and positional terms are the weight vector, tuned by the
     TD error from (2). Over ten human games this moves them by a rounding
     error, and any claim otherwise would be theatre. If you want the weights
     genuinely trained, run self_play() a few hundred games; that is what the
     number of games is there for.

The server's Quixx brain learned this the hard way already — see the comment
above AI_ROSTER_SIZE in homegames_server.py, where handicapping bots with
value noise washed out over 84 rounds because good weights are robust to
noise. Small samples move small amounts. So the strength you will feel in the
first dozen games comes mostly from search depth and the book, which is what
skill_level() ramps, and the weight learning accrues quietly underneath.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import chess
import chess.polyglot

BASE = os.path.dirname(os.path.abspath(__file__))

# Sits beside stats.json and ai_profiles.json, and like them it is per-machine
# live data rather than something to track in git.
DATA_FILE = os.path.join(BASE, 'chess_ai.json')
DATA_VERSION = 1


def iso_now():
    return datetime.now().replace(microsecond=0).isoformat()


# ============================================================
#  Evaluation
# ============================================================
# The evaluator is a plain dot product: score = sum(weight[k] * feature[k]).
# Keeping it linear is not an aesthetic choice — it is what makes the learning
# in _learn_weights() a two-line gradient step instead of a backprop engine.
# Every feature below is a *white-positive differential*: mine minus theirs,
# from White's side. Search flips the sign for Black.
#
# Signs are arranged so that a larger weight is always "cares about this more",
# which keeps the bounds table readable and stops a learner from wandering into
# a sign flip that silently inverts a term.
FEATURE_KEYS = [
    'pawn', 'knight', 'bishop', 'rook', 'queen',   # material — weights ARE the piece values
    'pst',            # piece-square tables: the fixed positional prior
    'mobility',       # squares attacked
    'king_safety',    # enemy attackers near their king, less those near mine
    'center',         # attacks on the four central squares
    'pawn_struct',    # their doubled+isolated pawns, less mine
    'bishop_pair',    # holding both bishops
    'development',    # minor pieces off the back rank (opening only)
]

# Defaults are ordinary club-strength values — a deliberately unremarkable
# starting point, since the interesting question is where learning takes them.
DEFAULT_WEIGHTS = {
    'pawn': 100.0, 'knight': 320.0, 'bishop': 330.0, 'rook': 500.0, 'queen': 900.0,
    'pst': 1.0, 'mobility': 4.0, 'king_safety': 6.0, 'center': 8.0,
    'pawn_struct': 9.0, 'bishop_pair': 25.0, 'development': 10.0,
}

# The pawn is deliberately NOT learnable. It is the unit the other weights are
# denominated in: if every weight can drift, the whole vector can scale up
# together, which changes nothing about play but makes the learning rate mean
# something different every game. Pinning the pawn at 100 anchors the scale.
LEARN_KEYS = [k for k in FEATURE_KEYS if k != 'pawn']

# Bounds are wide enough to let the AI form real opinions (a knight worth more
# than a bishop, say) and tight enough that a bad run cannot destroy it. An
# unbounded learner that decides a queen is worth 40 points is not a weaker
# opponent, it is a broken one.
WEIGHT_BOUNDS = {
    'pawn': (100.0, 100.0),
    'knight': (250.0, 400.0), 'bishop': (250.0, 400.0),
    'rook': (400.0, 640.0), 'queen': (750.0, 1100.0),
    'pst': (0.0, 3.0), 'mobility': (0.0, 15.0), 'king_safety': (0.0, 30.0),
    'center': (0.0, 30.0), 'pawn_struct': (0.0, 30.0),
    'bishop_pair': (0.0, 80.0), 'development': (0.0, 40.0),
}

# Per-feature learning rates. Material moves slowest: piece values are the part
# most likely to be right already, and the part most damaging to get wrong.
LEARN_RATE = {
    'knight': 0.020, 'bishop': 0.020, 'rook': 0.020, 'queen': 0.012,
    'pst': 0.004, 'mobility': 0.030, 'king_safety': 0.030, 'center': 0.030,
    'pawn_struct': 0.030, 'bishop_pair': 0.060, 'development': 0.040,
}

# Piece-square tables, written the way a board is read — rank 8 on top — then
# flipped at import so index 0 is a1, matching python-chess square numbering.
# These stay fixed; they are the positional prior the learner adjusts *around*
# via the single 'pst' scaling weight, rather than 384 separately-learned
# numbers that a few dozen games could never fill in.
def _table(rows):
    """Take eight rank-8-first rows of eight and return a1-first squares."""
    out = []
    for row in reversed(rows):
        out.extend(row)
    return out


PST = {
    chess.PAWN: _table([
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [ 50,  50,  50,  50,  50,  50,  50,  50],
        [ 10,  10,  20,  30,  30,  20,  10,  10],
        [  5,   5,  10,  25,  25,  10,   5,   5],
        [  0,   0,   0,  20,  20,   0,   0,   0],
        [  5,  -5, -10,   0,   0, -10,  -5,   5],
        [  5,  10,  10, -20, -20,  10,  10,   5],
        [  0,   0,   0,   0,   0,   0,   0,   0],
    ]),
    chess.KNIGHT: _table([
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-30,   5,  15,  20,  20,  15,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50],
    ]),
    chess.BISHOP: _table([
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20],
    ]),
    chess.ROOK: _table([
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [  5,  10,  10,  10,  10,  10,  10,   5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [  0,   0,   0,   5,   5,   0,   0,   0],
    ]),
    chess.QUEEN: _table([
        [-20, -10, -10,  -5,  -5, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,   5,   5,   5,   5,   0, -10],
        [ -5,   0,   5,   5,   5,   5,   0,  -5],
        [  0,   0,   5,   5,   5,   5,   0,  -5],
        [-10,   5,   5,   5,   5,   5,   0, -10],
        [-10,   0,   5,   0,   0,   0,   0, -10],
        [-20, -10, -10,  -5,  -5, -10, -10, -20],
    ]),
    chess.KING: _table([
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [ 20,  20,   0,   0,   0,   0,  20,  20],
        [ 20,  30,  10,   0,   0,  10,  30,  20],
    ]),
}

# In an endgame the king stops hiding and starts working, so it gets its own
# table rather than being told to cower in a corner that no longer exists.
PST_KING_END = _table([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
])

MATERIAL_KEY = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
                chess.ROOK: 'rook', chess.QUEEN: 'queen'}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

# Non-pawn material below this (in default centipawns, both sides) counts as an
# endgame, which is where the king's table swaps over.
ENDGAME_MATERIAL = 1300

MATE_SCORE = 1_000_000


# ============================================================
#  The searchless policy — how a "learner" bot chooses
# ============================================================
# A learner never builds a game tree. It scores each legal move by a set of
# facts readable straight off the board, and picks from those scores. The
# weights below are the entire brain, and they start at zero: nothing here is
# hand-tuned, which is the point — a learner that shipped with "captures are
# good" already written in would not be learning that, it would be reciting it.
#
# Why features and not a table of positions: chess positions stop repeating
# after about ten plies, so a position->move table learns the opening and then
# faces something new every game forever. Features generalise — "this move
# leaves a piece where a pawn can take it" is the same fact in a position never
# seen before. That generalisation is the job a neural network does in a modern
# engine; a dozen hand-picked features and a linear policy is the version that
# fits in pure Python.
#
# The cost is honest and large: with no lookahead the bot cannot see a
# recapture, so it must *learn* that landing on a defended square tends to end
# badly, rather than working it out. Expect hundreds of games. self_play() is
# how you buy them without playing them yourself.
MOVE_FEATURE_KEYS = [
    'capture',    # value of the piece it takes
    'hanging',    # value it exposes on a square the opponent attacks
    'defended',   # the destination is covered by one of ours
    'check',      # gives check
    'promote',    # promotion value
    'escape',     # the piece was attacked where it stood
    'develop',    # a minor piece leaving the back rank
    'centre',     # moves toward the middle of the board
    'pst',        # piece-square improvement for the moving piece
    'castle',
    'kingwalk',   # king wandering before the endgame
    'pawnpush',   # a pawn advancing
]

# All zero: a fresh learner has no opinions and plays uniformly at random.
DEFAULT_POLICY = {k: 0.0 for k in MOVE_FEATURE_KEYS}
POLICY_BOUNDS = {k: (-4.0, 4.0) for k in MOVE_FEATURE_KEYS}
# Feature magnitudes differ (capture is ~1-9, castle is 0/1), so the rates are
# scaled to give each roughly the same influence per game.
POLICY_RATE = {'capture': 0.035, 'hanging': 0.035, 'defended': 0.06,
               'check': 0.06, 'promote': 0.03, 'escape': 0.035,
               'develop': 0.06, 'centre': 0.05, 'pst': 0.04,
               'castle': 0.08, 'kingwalk': 0.06, 'pawnpush': 0.05}

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _centrality(square):
    """0 at the rim, 3 in the middle four squares."""
    f, r = chess.square_file(square), chess.square_rank(square)
    return 3.0 - max(abs(f - 3.5), abs(r - 3.5)) + 0.5


def move_features(board, move):
    """What the policy knows about one move, read off the position as it
    stands. Nothing here pushes the move and looks at the result — that would
    be a one-ply search, and the whole point of a learner is that it has none.
    """
    f = dict.fromkeys(MOVE_FEATURE_KEYS, 0.0)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return f
    us, them = board.turn, not board.turn
    mine = PIECE_VALUE.get(piece.piece_type, 0) / 100.0

    if board.is_en_passant(move):
        f['capture'] = 1.0
    else:
        victim = board.piece_type_at(move.to_square)
        if victim:
            f['capture'] = PIECE_VALUE[victim] / 100.0

    # Is the destination contested? Our own mover still sits on its old square,
    # so it would count itself as a defender — hence the explicit exclusion.
    defenders = [s for s in board.attackers(us, move.to_square)
                 if s != move.from_square]
    threatened = board.is_attacked_by(them, move.to_square)
    if defenders:
        f['defended'] = 1.0
    if threatened:
        # *Net* material at risk: what the piece is worth less what the move
        # just won. Charging the gross value instead made this fire on nearly
        # every capture, so it moved in lockstep with `capture` and the
        # gradient could not tell "taking is good" from "being takeable is
        # bad" — it learned a positive weight for hanging pieces, which is the
        # one thing a bot with no lookahead cannot afford to get wrong.
        # Subtracting the capture is what breaks that collinearity.
        risk = max(0.0, mine - f['capture'])
        # With a defender the exchange is answered, so only part of it is lost.
        f['hanging'] = risk * (0.35 if defenders else 1.0)

    if board.is_attacked_by(them, move.from_square):
        f['escape'] = mine
    if move.promotion:
        f['promote'] = PIECE_VALUE.get(move.promotion, 0) / 100.0
    if board.gives_check(move):
        f['check'] = 1.0
    if board.is_castling(move):
        f['castle'] = 1.0

    back = 0 if us == chess.WHITE else 7
    if (piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            and chess.square_rank(move.from_square) == back):
        f['develop'] = 1.0
    if piece.piece_type == chess.KING and not board.is_castling(move):
        f['kingwalk'] = 1.0
    if piece.piece_type == chess.PAWN:
        adv = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        f['pawnpush'] = float(adv if us == chess.WHITE else -adv)

    f['centre'] = _centrality(move.to_square) - _centrality(move.from_square)

    table = PST.get(piece.piece_type)
    if table:
        a = move.from_square if us == chess.WHITE else chess.square_mirror(move.from_square)
        b = move.to_square if us == chess.WHITE else chess.square_mirror(move.to_square)
        f['pst'] = (table[b] - table[a]) / 100.0
    return f


def static_eval(board):
    """White-positive centipawn evaluation using the *default* weights.

    Deliberately not any particular bot's learned weights: this is what the UI
    narrates, and commentary should describe the position rather than one
    opponent's idiosyncratic opinion of it. Cheap enough to call once per
    broadcast — a single features() pass, no search.
    """
    f = features(board)
    return sum(DEFAULT_WEIGHTS[k] * f[k] for k in FEATURE_KEYS)


def features(board):
    """Every evaluation term for one position, white-positive.

    Returns a dict keyed by FEATURE_KEYS. Search and learning both go through
    this one function on purpose: an evaluator that is fast in search and a
    separate one that is differentiable for learning is two implementations
    that quietly disagree, and the disagreement shows up as a learner chasing a
    function nobody is actually playing. _selftest() asserts the dot product
    here matches evaluate().
    """
    f = dict.fromkeys(FEATURE_KEYS, 0.0)

    pieces = board.piece_map()

    # One pass for material, piece-square score, pawn files and development.
    pawn_files = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
    non_pawn = 0
    pst_sum = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}
    developed = {chess.WHITE: 0, chess.BLACK: 0}

    for square, piece in pieces.items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        pt = piece.piece_type

        if pt in MATERIAL_KEY:
            f[MATERIAL_KEY[pt]] += sign
            if pt != chess.PAWN:
                non_pawn += DEFAULT_WEIGHTS[MATERIAL_KEY[pt]]
        if pt == chess.PAWN:
            pawn_files[piece.color][chess.square_file(square)] += 1
        if pt == chess.BISHOP:
            bishops[piece.color] += 1
        if pt in (chess.KNIGHT, chess.BISHOP):
            back = 0 if piece.color == chess.WHITE else 7
            if chess.square_rank(square) != back:
                developed[piece.color] += 1

    endgame = non_pawn < ENDGAME_MATERIAL
    for square, piece in pieces.items():
        sign = 1 if piece.color == chess.WHITE else -1
        # Black reads the same table through a vertical mirror.
        idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        if piece.piece_type == chess.KING:
            pst_sum += sign * (PST_KING_END[idx] if endgame else PST[chess.KING][idx])
        else:
            pst_sum += sign * PST[piece.piece_type][idx]
    f['pst'] = float(pst_sum)

    # Mobility as attacked-square count. Genuine legal-move counting needs a
    # null move to ask the same question of the side not to move, and the extra
    # move generation costs more than the accuracy is worth inside a search.
    # Attack span correlates well and is markedly cheaper.
    mob = {chess.WHITE: 0, chess.BLACK: 0}
    center = {chess.WHITE: 0, chess.BLACK: 0}
    for square, piece in pieces.items():
        attacks = board.attacks(square)
        mob[piece.color] += len(attacks)
        for c in CENTER_SQUARES:
            if c in attacks:
                center[piece.color] += 1
    f['mobility'] = float(mob[chess.WHITE] - mob[chess.BLACK])
    f['center'] = float(center[chess.WHITE] - center[chess.BLACK])

    # King safety: how many enemy pieces attack the ring around each king.
    # Positive means *their* king is the draughtier one.
    danger = {}
    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        if ksq is None:
            danger[color] = 0
            continue
        ring = board.attacks(ksq)
        danger[color] = sum(1 for sq in ring
                            if board.is_attacked_by(not color, sq))
    f['king_safety'] = float(danger[chess.BLACK] - danger[chess.WHITE])

    # Pawn structure: doubled and isolated pawns, counted as faults. Positive
    # means they have more faults than we do.
    def faults(color):
        files = pawn_files[color]
        bad = 0
        for i, n in enumerate(files):
            if n == 0:
                continue
            bad += n - 1                                   # doubled
            left = files[i - 1] if i > 0 else 0
            right = files[i + 1] if i < 7 else 0
            if not left and not right:
                bad += 1                                   # isolated
        return bad
    f['pawn_struct'] = float(faults(chess.BLACK) - faults(chess.WHITE))

    f['bishop_pair'] = float((1 if bishops[chess.WHITE] >= 2 else 0)
                             - (1 if bishops[chess.BLACK] >= 2 else 0))

    # Development only means anything while there is a back rank to leave.
    f['development'] = 0.0 if endgame else float(developed[chess.WHITE]
                                                 - developed[chess.BLACK])
    return f


# ============================================================
#  The AI
# ============================================================
class ChessAI:
    """A learning chess opponent.

    One instance per game is the intended use, constructed with the same
    data_path each time so the brain carries across games and sessions.

    Public surface:
        get_move(board)        -> UCI string (or None if there is no move)
        learn_from_game(result)-> dict describing what changed
        save_progress()        -> write the brain to disk
        load_progress()        -> re-read it
        reset()                -> wipe the brain back to defaults
        stats()                -> a display-ready dict
        self_play(n)           -> bootstrap the weights offline
    """

    # --- learning constants ------------------------------------------------
    # How much of a move's credit comes from what happened locally (the
    # position two plies later) versus the game's final result. Local
    # dominates deliberately: the result is one bit spread over forty moves,
    # and the local signal is the only one with any resolution to it.
    LOCAL_SHARE = 0.75

    # Centipawns that count as a "full" local swing. A move that loses a minor
    # piece scores about -1 after the tanh; anything past that saturates, so a
    # single catastrophe cannot dominate a whole game's worth of updates.
    SWING_SCALE = 300.0

    # How much of the weight-learning target is the game's final result rather
    # than the bootstrapped value of the next position. Small on purpose: the
    # result is one bit for the whole game, and leaning on it is what makes a
    # learner chase noise. The bootstrap carries the detail.
    OUTCOME_SHARE = 0.15

    # UCB1 exploration constant, in the same units as the book's stored value
    # (which is bounded to roughly [-1, 1]). Larger tries new moves for longer.
    EXPLORE_C = 0.7

    # Book entries only start steering play once they have been seen enough to
    # mean something. Below this the AI plays its search move and just records.
    BOOK_MIN_VISITS = 2

    # How deep into a game the book applies. Past this, positions stop
    # repeating between games and the table is memorising noise.
    BOOK_MAX_PLY = 24

    def __init__(self, name='Chess AI', data_path=None, opponent=None,
                 think_time=1.0, max_depth=4, seed=None, autoload=True,
                 style='searcher'):
        self.name = name
        self.data_path = data_path or DATA_FILE
        self.opponent = (opponent or 'human').strip().lower() or 'human'
        self.think_time = float(think_time)
        self.max_depth = int(max_depth)
        # 'searcher' looks ahead; 'learner' never does and plays from the move
        # policy alone. Both learn — the difference is only how a move is
        # chosen, which is what makes the pair worth rating against each other.
        self.style = 'learner' if str(style) == 'learner' else 'searcher'
        self.rng = random.Random(seed)

        # Persistent brain.
        self.weights = dict(DEFAULT_WEIGHTS)
        self.policy = dict(DEFAULT_POLICY)
        self.book = {}          # position key -> {uci: [visits, total_value]}
        self.record = {'played': 0, 'won': 0, 'lost': 0, 'drawn': 0}
        self.opponents = {}     # name -> {'played','won','lost','drawn','openings'}
        self.history = []       # recent finished games, newest last
        self.born = iso_now()
        self.updates = 0        # weight updates applied, all-time

        # Per-game scratch, reset by new_game().
        self._trace = []
        self._last_eval = None
        self._nodes = 0
        self._deadline = 0.0
        self._tt = {}

        if autoload:
            self.load_progress()
        self.new_game()

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def load_progress(self):
        """Read the brain from disk. A missing or damaged file is not an
        error — it just means a brand new AI, same as the server's stats and
        profile loaders do."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False

        w = data.get('weights')
        if isinstance(w, dict):
            for k in FEATURE_KEYS:
                if isinstance(w.get(k), (int, float)):
                    self.weights[k] = self._clamp(k, float(w[k]))

        p = data.get('policy')
        if isinstance(p, dict):
            for k in MOVE_FEATURE_KEYS:
                if isinstance(p.get(k), (int, float)):
                    lo, hi = POLICY_BOUNDS[k]
                    self.policy[k] = min(hi, max(lo, float(p[k])))

        book = data.get('book')
        if isinstance(book, dict):
            clean = {}
            for key, moves in book.items():
                if not isinstance(moves, dict):
                    continue
                entry = {}
                for uci, stat in moves.items():
                    # [visits, total_value]
                    if (isinstance(stat, list) and len(stat) == 2
                            and all(isinstance(x, (int, float)) for x in stat)):
                        entry[uci] = [int(stat[0]), float(stat[1])]
                if entry:
                    clean[key] = entry
            self.book = clean

        rec = data.get('record')
        if isinstance(rec, dict):
            for k in self.record:
                if isinstance(rec.get(k), int):
                    self.record[k] = rec[k]

        opps = data.get('opponents')
        if isinstance(opps, dict):
            self.opponents = {k: v for k, v in opps.items() if isinstance(v, dict)}

        hist = data.get('history')
        if isinstance(hist, list):
            self.history = [h for h in hist if isinstance(h, dict)][-50:]

        self.born = data.get('born') or self.born
        if isinstance(data.get('updates'), int):
            self.updates = data['updates']
        return True

    def save_progress(self):
        """Write via a temp file, so a crash mid-write cannot shred the brain.
        Same idiom as save_stats()/save_profiles() in the server."""
        data = {
            'version': DATA_VERSION,
            'name': self.name,
            'born': self.born,
            'updated': iso_now(),
            'record': dict(self.record),
            'updates': self.updates,
            'style': self.style,
            'weights': {k: round(self.weights[k], 4) for k in FEATURE_KEYS},
            # Six places, not four. Policy weights sit near 0.02 and a game
            # nudges them by less again, so rounding at four would quietly
            # discard the smaller lessons every time the file is written.
            'policy': {k: round(self.policy[k], 6) for k in MOVE_FEATURE_KEYS},
            'opponents': self.opponents,
            'book': self.book,
            'history': self.history[-50:],
        }
        tmp = self.data_path + '.tmp'
        try:
            # The directory is made here rather than by the caller. It used to
            # be created only when a game engine was built, so the trainer —
            # which makes a ChessAI directly and never builds an engine — wrote
            # into a folder that did not exist yet and lost the first bot's
            # round to a logged-and-ignored error.
            parent = os.path.dirname(self.data_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.data_path)
            return True
        except OSError as e:
            print('  [chess-ai] could not write %s: %s' % (self.data_path, e))
            return False

    def reset(self, keep_file=False):
        """Wipe the brain back to a newborn. Writes immediately unless asked
        not to, so 'reset' means reset even if the process dies next."""
        self.weights = dict(DEFAULT_WEIGHTS)
        self.policy = dict(DEFAULT_POLICY)
        self.book = {}
        self.record = {'played': 0, 'won': 0, 'lost': 0, 'drawn': 0}
        self.opponents = {}
        self.history = []
        self.updates = 0
        self.born = iso_now()
        self.new_game()
        if not keep_file:
            self.save_progress()

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _clamp(self, key, value):
        lo, hi = WEIGHT_BOUNDS[key]
        return min(hi, max(lo, value))

    def evaluate(self, board):
        """Static score of a position from the side-to-move's point of view."""
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        f = features(board)
        score = 0.0
        for k in FEATURE_KEYS:
            score += self.weights[k] * f[k]
        return score if board.turn == chess.WHITE else -score

    # ------------------------------------------------------------------
    #  Search — alpha-beta, iterative deepening, quiescence
    # ------------------------------------------------------------------
    def _order_moves(self, board, moves, tt_move=None):
        """Cheap ordering. Alpha-beta's whole benefit depends on trying good
        moves first, and MVV-LVA plus the transposition move gets most of it
        for very little work."""
        def score(m):
            if tt_move is not None and m == tt_move:
                return 10_000_000
            s = 0
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square)
                attacker = board.piece_type_at(m.from_square)
                # en passant has no piece on the target square
                v = DEFAULT_WEIGHTS[MATERIAL_KEY[victim]] if victim in MATERIAL_KEY else 100.0
                a = DEFAULT_WEIGHTS[MATERIAL_KEY[attacker]] if attacker in MATERIAL_KEY else 0.0
                s += 100_000 + v * 10 - a
            if m.promotion:
                s += 90_000 + m.promotion * 100
            if board.gives_check(m):
                s += 50_000
            return s
        return sorted(moves, key=score, reverse=True)

    def _quiesce(self, board, alpha, beta):
        """Search on past the horizon until the position is quiet.

        Without this the AI happily plays a move whose refutation is one ply
        past its depth limit — it grabs a defended pawn on the last ply and
        never sees the recapture. This is the single largest difference
        between "plays chess" and "hangs a piece every game".
        """
        self._nodes += 1
        stand = self.evaluate(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        for move in self._order_moves(board, [m for m in board.legal_moves
                                              if board.is_capture(m) or m.promotion]):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _negamax(self, board, depth, alpha, beta, ply):
        self._nodes += 1

        # A repetition inside the search is a draw offer, and the AI should
        # only take it when it is worse off. Without this it shuffles a won
        # position back and forth quite contentedly.
        if ply and board.is_repetition(2):
            return 0
        if board.is_checkmate():
            return -MATE_SCORE + ply         # prefer the faster mate
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if board.can_claim_fifty_moves():
            return 0

        if depth <= 0:
            return self._quiesce(board, alpha, beta)

        key = chess.polyglot.zobrist_hash(board)
        hit = self._tt.get(key)
        tt_move = None
        if hit and hit[0] >= depth:
            return hit[1]
        if hit:
            tt_move = hit[2]

        best = -MATE_SCORE * 2
        best_move = None
        for move in self._order_moves(board, list(board.legal_moves), tt_move):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best, best_move = score, move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            # Checked between moves rather than per node: bailing mid-node
            # leaves a half-searched score, and a slightly late stop is much
            # cheaper than a wrong one.
            if time.monotonic() > self._deadline:
                break

        self._tt[key] = (depth, best, best_move)
        return best

    def search(self, board, think_time=None, max_depth=None):
        """Iterative deepening within a time budget.

        Time rather than fixed depth because the same code runs on a desktop
        and on the Pi that actually hosts HomeGames, and a depth that is
        comfortable on one is a thirty-second pause on the other.
        Returns (best_move, score, depth_reached).
        """
        moves = list(board.legal_moves)
        if not moves:
            return None, 0, 0

        budget = self.think_time if think_time is None else think_time
        limit = self.max_depth if max_depth is None else max_depth
        self._deadline = time.monotonic() + budget
        self._nodes = 0
        self._tt = {}

        best_move, best_score, reached = moves[0], 0, 0
        for depth in range(1, max(1, limit) + 1):
            alpha, beta = -MATE_SCORE * 2, MATE_SCORE * 2
            local_best, local_score = None, -MATE_SCORE * 2
            for move in self._order_moves(board, moves, best_move):
                board.push(move)
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
                board.pop()
                if score > local_score:
                    local_score, local_best = score, move
                if local_score > alpha:
                    alpha = local_score
                if time.monotonic() > self._deadline:
                    break
            # Only trust a depth that finished; a partial pass has seen an
            # arbitrary prefix of the move list and its "best" is meaningless.
            done = time.monotonic() <= self._deadline
            if local_best is not None and (done or reached == 0):
                best_move, best_score, reached = local_best, local_score, depth
            if not done:
                break
        return best_move, best_score, reached

    # ------------------------------------------------------------------
    #  Move selection
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Searchless move selection
    # ------------------------------------------------------------------
    def policy_temperature(self):
        """How randomly the learner picks among its scored moves.

        Starts wide because at zero weights every move scores the same and it
        may as well explore; narrows as games accumulate and the weights start
        to mean something. Never reaches zero — a policy that stops exploring
        cannot discover that its current favourite is bad.
        """
        return max(0.35, 1.6 - 0.02 * self.record['played'])

    def _policy_pick(self, board, legal):
        """Score every legal move, sample one, and return what the gradient
        step will need.

        Sampling from a softmax rather than taking the best move is the
        exploration mechanism, and it is also what makes the update below a
        policy gradient rather than a guess: the expected feature vector is the
        baseline the chosen move is measured against.
        """
        feats = [move_features(board, m) for m in legal]
        scores = [sum(self.policy[k] * f[k] for k in MOVE_FEATURE_KEYS)
                  for f in feats]
        temp = self.policy_temperature()
        top = max(scores)
        exps = [math.exp((s - top) / temp) for s in scores]      # top subtracted
        total = sum(exps) or 1.0                                 # for overflow
        probs = [e / total for e in exps]

        r, acc, idx = self.rng.random(), 0.0, len(legal) - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                idx = i
                break
        expected = {k: sum(p * f[k] for p, f in zip(probs, feats))
                    for k in MOVE_FEATURE_KEYS}
        return legal[idx], feats[idx], expected

    def skill_level(self):
        """Search depth for this game.

        The honest driver of "gets stronger as it plays". Weight learning is
        real but slow (see the module docstring); depth is what a person
        notices across their first dozen games. Starts shallow enough to be
        beatable and climbs to max_depth by roughly the 24th game.
        """
        played = self.record['played']
        return max(2, min(self.max_depth, 2 + played // 8))

    @staticmethod
    def book_key(board):
        """Position identity for the book: pieces, side, castling and en
        passant, but not the clocks. Two games reaching the same position by
        different move orders should share what was learned there."""
        return ' '.join(board.fen().split(' ')[:4])

    def _book_choice(self, board, legal):
        """Pick from the book by UCB1, or return None to defer to search.

        UCB1 rather than epsilon-greedy because the whole regime here is small
        samples. Epsilon-greedy spends a fixed share of its turns on moves it
        already knows are bad; UCB1 concentrates exploration on the moves it is
        genuinely uncertain about, which is what you want when the entire
        dataset is thirty games. The bonus term shrinks as a move gets tried,
        so the book settles down by itself without a schedule to tune.
        """
        entry = self.book.get(self.book_key(board))
        if not entry:
            return None
        total = sum(v[0] for v in entry.values())
        if total < self.BOOK_MIN_VISITS:
            return None

        legal_uci = {m.uci() for m in legal}
        best, best_score = None, None
        for uci, (visits, value) in entry.items():
            if uci not in legal_uci:
                continue
            mean = value / visits if visits else 0.0
            bonus = self.EXPLORE_C * math.sqrt(math.log(total + 1) / (visits + 1))
            score = mean + bonus
            if best_score is None or score > best_score:
                best, best_score = uci, score

        # An unvisited legal move is worth a look before re-trying a known one:
        # its bonus would be the largest of all, so treat it as the pick.
        unseen = [m.uci() for m in legal if m.uci() not in entry]
        if unseen:
            widest = self.EXPLORE_C * math.sqrt(math.log(total + 1))
            if best_score is None or widest > best_score:
                return self.rng.choice(unseen)
        return best

    def get_move(self, board_fen_or_state):
        """Choose a move. Returns a UCI string, or None if there is none.

        Accepts a chess.Board, a FEN string, or a dict carrying 'fen' — the
        server hands over whichever is convenient at the call site.
        """
        board = self._as_board(board_fen_or_state)
        legal = list(board.legal_moves)
        if not legal:
            return None

        source = 'search'
        chosen = None
        extra = None

        # The book is learned too, so both styles consult it.
        if board.ply() <= self.BOOK_MAX_PLY:
            pick = self._book_choice(board, legal)
            if pick:
                try:
                    move = chess.Move.from_uci(pick)
                    if move in legal:
                        chosen, source = move, 'book'
                except ValueError:
                    chosen = None

        if chosen is None:
            if self.style == 'learner':
                chosen, feats, expected = self._policy_pick(board, legal)
                source = 'policy'
                extra = {'pf': feats, 'pe': expected}
            else:
                chosen, _, _ = self.search(board, max_depth=self.skill_level())
                if chosen is None or chosen not in legal:
                    chosen = self.rng.choice(legal)

        self._record_decision(board, chosen, source, extra)
        return chosen.uci()

    @staticmethod
    def _as_board(x):
        if isinstance(x, chess.Board):
            return x
        if isinstance(x, dict):
            fen = x.get('fen') or x.get('board') or chess.STARTING_FEN
            return chess.Board(fen)
        if isinstance(x, str):
            return chess.Board(x)
        raise TypeError('get_move needs a chess.Board, a FEN string, or a dict '
                        'with a "fen" key; got %r' % type(x).__name__)

    # ------------------------------------------------------------------
    #  Learning
    # ------------------------------------------------------------------
    def new_game(self):
        """Clear the per-game trace. Called by __init__, and worth calling
        directly if you reuse one instance across games."""
        self._trace = []
        self._last_eval = None

    def _record_decision(self, board, move, source, extra=None):
        """Log one of the AI's own decisions, plus the eval of the position it
        was looking at.

        The trace is what makes move-level credit possible later. Note the
        second half: on each turn we also close out the *previous* decision by
        recording what the position was worth two plies on, once the opponent
        has actually replied. That is the local signal, and it costs nothing
        beyond an evaluate() we were going to do anyway.
        """
        colour = board.turn
        static = self.evaluate(board)
        if board.turn == chess.BLACK:
            static = -static                      # store white-positive
        if self._trace and self._trace[-1].get('after') is None:
            self._trace[-1]['after'] = static

        step = {
            'key': self.book_key(board),
            'uci': move.uci(),
            'ply': board.ply(),
            'colour': 'w' if colour == chess.WHITE else 'b',
            'before': static,
            'after': None,
            'source': source,
            'features': features(board),
        }
        # A learner also carries the move's own feature vector and the average
        # over everything it could have played. Those two are what the policy
        # gradient subtracts, so they have to be captured at decision time —
        # the alternatives are gone once the move is made.
        if extra:
            step.update(extra)
        self._trace.append(step)

    @staticmethod
    def _normalize_result(result):
        """Accept whatever the caller has to hand and return a score in
        {1.0, 0.5, 0.0} from the AI's own point of view."""
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            v = float(result)
            if v in (1.0, 0.5, 0.0):
                return v
            return 1.0 if v > 0.5 else (0.0 if v < 0.5 else 0.5)
        if isinstance(result, dict):
            for k in ('result', 'score', 'outcome'):
                if k in result:
                    return ChessAI._normalize_result(result[k])
            raise ValueError('result dict needs a "result", "score" or "outcome" key')
        text = str(result).strip().lower()
        if text in ('win', 'won', 'w', '1', '1-0', 'victory'):
            return 1.0
        if text in ('loss', 'lost', 'lose', 'l', '0', '0-1', 'defeat'):
            return 0.0
        if text in ('draw', 'drawn', 'd', 'tie', '0.5', '1/2', '1/2-1/2'):
            return 0.5
        raise ValueError('unrecognised result %r' % (result,))

    def learn_from_game(self, result, final_board=None, opponent=None,
                        winner=None):
        """Fold a finished game back into the brain.

        `result` is from the AI's point of view: 'win'/'loss'/'draw', or
        1/0.5/0, or a dict with one of those under 'result'/'score'/'outcome'.

        `winner` ('w', 'b' or 'draw') only matters when the trace holds moves
        the AI played for *both* sides, which is the self-play case. Given it,
        each traced move is credited by whether its own colour won rather than
        by one game-wide result — otherwise half of every decisive self-play
        game teaches the loser's moves that they worked.

        Returns a small dict describing what changed, which is handy to show
        the player after the game.
        """
        score = self._normalize_result(result)
        who = (opponent or self.opponent or 'human').strip().lower() or 'human'

        # --- headline record ---
        self.record['played'] += 1
        self.record['won' if score == 1.0 else
                    ('lost' if score == 0.0 else 'drawn')] += 1

        opp = self.opponents.setdefault(who, {'played': 0, 'won': 0, 'lost': 0,
                                              'drawn': 0, 'openings': {}})
        opp['played'] += 1
        opp['won' if score == 1.0 else ('lost' if score == 0.0 else 'drawn')] += 1

        # Close out the final decision using the finished position, so the last
        # move of the game gets the same treatment as every other.
        if self._trace and self._trace[-1].get('after') is None:
            if final_board is not None:
                b = self._as_board(final_board)
                tail = self.evaluate(b)
                if b.turn == chess.BLACK:
                    tail = -tail
                self._trace[-1]['after'] = tail
            else:
                self._trace[-1]['after'] = self._trace[-1]['before']

        # --- the learning passes ---
        book_touched = self._learn_book(score, winner)
        moved = self._learn_weights(score, winner)
        policy_moved = self._learn_policy(score, winner)

        # --- opponent's opening fingerprint ---
        # Only their first few moves: enough to recognise "this is the
        # Scandinavian again", short enough that it is a handful of keys rather
        # than a transcript of every game they have played.
        if final_board is not None:
            b = self._as_board(final_board)
            opening = [m.uci() for m in b.move_stack[:6]]
            if opening:
                sig = ' '.join(opening)
                opp['openings'][sig] = opp['openings'].get(sig, 0) + 1
                # keep the dozen they play most; the tail is noise
                if len(opp['openings']) > 12:
                    top = sorted(opp['openings'].items(), key=lambda kv: -kv[1])[:12]
                    opp['openings'] = dict(top)

        self.history.append({
            'at': iso_now(), 'opponent': who,
            'result': {1.0: 'win', 0.5: 'draw', 0.0: 'loss'}[score],
            'moves': len(self._trace), 'depth': self.skill_level(),
        })
        del self.history[:-50]

        summary = {
            'result': {1.0: 'win', 0.5: 'draw', 0.0: 'loss'}[score],
            'record': dict(self.record),
            'book_positions': len(self.book),
            'book_updated': book_touched,
            'weights_moved': moved,
            'policy_moved': policy_moved,
            'style': self.style,
            'updates': self.updates,
            'depth_next_game': 0 if self.style == 'learner' else self.skill_level(),
        }
        self.new_game()
        return summary

    @staticmethod
    def _outcome_for(step, score, winner):
        """The game-result share of one move's credit: +1 good, -1 bad.

        Normally that is just the AI's own result. In self-play the trace holds
        both colours, so each move is judged by whether the side that played it
        was the side that won.
        """
        if winner is None:
            return (score - 0.5) * 2.0           # win 1, draw 0, loss -1
        if winner == 'draw':
            return 0.0
        return 1.0 if step['colour'] == winner else -1.0

    def _learn_book(self, score, winner=None):
        """Credit each of the AI's own opening moves.

        Blends the local swing with the game's result. A move that looked fine
        two plies later but lost the game still gets some blame, and a move
        that hung a rook still gets blamed even if the AI went on to win — the
        second is the important one, because a book that only tracked results
        would happily learn that a blunder is fine as long as the opponent
        blunders back.
        """
        touched = 0
        for step in self._trace:
            if step['ply'] > self.BOOK_MAX_PLY:
                continue
            local = self._local_signal(step)
            outcome = self._outcome_for(step, score, winner)
            value = self.LOCAL_SHARE * local + (1.0 - self.LOCAL_SHARE) * outcome
            entry = self.book.setdefault(step['key'], {})
            stat = entry.setdefault(step['uci'], [0, 0.0])
            stat[0] += 1
            stat[1] += value
            touched += 1
        return touched

    def _learn_policy(self, score, winner=None):
        """REINFORCE on the linear softmax move policy.

        For a softmax over scores w·f, the gradient of log P(chosen) is

            f(chosen) - E[f]

        where E[f] is the probability-weighted average over the moves it could
        have played. So the update is

            w += rate * advantage * (f_chosen - f_expected)

        which reads exactly as it behaves: when a move worked out, shift the
        weights toward whatever made *that* move different from the alternatives
        it was picked over, and away when it did not. Subtracting the expected
        vector is what keeps it honest — without that baseline, every feature
        common to all legal moves would drift upward on any good result, which
        is a bias, not a lesson.

        Only learners record the vectors, so this is a no-op for searchers.
        """
        steps = [s for s in self._trace if 'pf' in s and 'pe' in s]
        if not steps:
            return 0
        moved = {k: 0.0 for k in MOVE_FEATURE_KEYS}
        for step in steps:
            local = self._local_signal(step)
            outcome = self._outcome_for(step, score, winner)
            advantage = (self.LOCAL_SHARE * local
                         + (1.0 - self.LOCAL_SHARE) * outcome)
            if not advantage:
                continue
            pf, pe = step['pf'], step['pe']
            for k in MOVE_FEATURE_KEYS:
                moved[k] += POLICY_RATE[k] * advantage * (pf[k] - pe[k])

        n = float(len(steps))
        changed = 0
        for k in MOVE_FEATURE_KEYS:
            if not moved[k]:
                continue
            lo, hi = POLICY_BOUNDS[k]
            before = self.policy[k]
            self.policy[k] = min(hi, max(lo, before + moved[k] / n))
            if abs(self.policy[k] - before) > 1e-4:
                changed += 1
        return changed

    def _local_signal(self, step):
        """How the position actually developed over the AI's move and the
        reply, squashed to roughly [-1, 1] so one disaster cannot swamp a
        game's worth of ordinary moves."""
        after = step.get('after')
        if after is None:
            return 0.0
        delta = after - step['before']
        if step['colour'] == 'b':
            delta = -delta                       # trace is stored white-positive
        return math.tanh(delta / self.SWING_SCALE)

    def _learn_weights(self, score, winner=None):
        """TD(0) on a linear evaluator.

        The evaluation is V(s) = tanh(w · f / SWING_SCALE), so dV/dw is just f
        (times the tanh derivative, which is folded into the rates), and the
        update is the textbook one:

            delta = target - V(s)
            w    += rate * delta * f

        The target is where the position actually went: V(s') two plies later,
        pulled toward the game's real result by OUTCOME_SHARE.

        Both sides of `delta` have to be *values of states* on one scale. An
        earlier version had V(s) as the absolute evaluation but the target as
        the two-ply *change* in evaluation, which are different quantities: any
        position the AI was winning gave a large positive V(s) against a target
        near zero, so every feature that correctly said "I am winning" got
        marked down. Over 60 self-play games that shrank all eleven weights,
        mobility by 80% — which reads as a preference and was really just the
        mismatch draining the vector. _selftest() check 12 pins it down.

        A *small* uniform shrink is a different thing and is not a bug: a
        shallow search with a crude evaluator really is overconfident, and when
        a third of self-play games are drawn from positions it called winning,
        TD is right to scale its advantages down. Measured after the fix, 60
        games moved the piece values about 0.4% (against 3% before). Large
        uniform drift means the scales have come apart again; small uniform
        drift means it is calibrating.

        Returns the number of weights that moved by a visible amount. Expect
        that to be small and the movements tiny — see the module docstring.
        """
        if not self._trace:
            return 0
        moved = {k: 0.0 for k in LEARN_KEYS}

        for step in self._trace:
            f = step['features']
            outcome = self._outcome_for(step, score, winner)

            # Both in centipawns from the AI's own point of view; the trace
            # stores them white-positive.
            before_cp = step['before']
            after_cp = step['after']
            if step['colour'] == 'b':
                before_cp = -before_cp
                if after_cp is not None:
                    after_cp = -after_cp

            v_now = math.tanh(before_cp / self.SWING_SCALE)
            v_next = (outcome if after_cp is None
                      else math.tanh(after_cp / self.SWING_SCALE))
            target = ((1.0 - self.OUTCOME_SHARE) * v_next
                      + self.OUTCOME_SHARE * outcome)

            delta = target - v_now
            if step['colour'] == 'b':
                delta = -delta                   # features are white-positive

            for k in LEARN_KEYS:
                fv = f[k]
                if not fv:
                    continue
                # Squash the feature too: this is a normalised gradient step,
                # not a raw one, which keeps a single lopsided position from
                # yanking a weight across its whole range.
                moved[k] += LEARN_RATE[k] * delta * math.tanh(fv / 4.0)

        n = float(len(self._trace))
        changed = 0
        for k in LEARN_KEYS:
            step_size = moved[k] / n
            if not step_size:
                continue
            before = self.weights[k]
            # Steps are proportional to each weight's natural scale, so a
            # single learning rate can serve a piece value near 900 and a
            # multiplier near 1 without one of them being frozen and the other
            # thrown across its whole range.
            scale = max(1.0, abs(DEFAULT_WEIGHTS[k]))
            self.weights[k] = self._clamp(k, before + step_size * scale)
            if abs(self.weights[k] - before) > 1e-4:
                changed += 1
        self.updates += 1
        return changed

    # ------------------------------------------------------------------
    #  Reporting
    # ------------------------------------------------------------------
    def stats(self):
        """Display-ready progress, safe to hand straight to a template or JSON
        response."""
        played = self.record['played']
        pct = (lambda n: round(100.0 * n / played, 1) if played else 0.0)
        opp = self.opponents.get(self.opponent, {})
        favourite = None
        if opp.get('openings'):
            favourite = max(opp['openings'].items(), key=lambda kv: kv[1])[0]
        learner = self.style == 'learner'
        return {
            'name': self.name,
            'born': self.born,
            'style': self.style,
            'record': dict(self.record),
            'winRate': pct(self.record['won']),
            'drawRate': pct(self.record['drawn']),
            # A learner has no depth to report; saying 0 rather than omitting it
            # keeps the shape stable for whatever is rendering this.
            'searchDepth': 0 if learner else self.skill_level(),
            'maxDepth': 0 if learner else self.max_depth,
            'policy': {k: round(self.policy[k], 2) for k in MOVE_FEATURE_KEYS},
            'policyLearned': {k: round(self.policy[k], 2)
                              for k in MOVE_FEATURE_KEYS
                              if abs(self.policy[k]) >= 0.01},
            'exploration': round(self.policy_temperature(), 2) if learner else 0,
            'bookPositions': len(self.book),
            'bookMoves': sum(len(v) for v in self.book.values()),
            'weightUpdates': self.updates,
            'weights': {k: round(self.weights[k], 2) for k in FEATURE_KEYS},
            'drift': {k: round(self.weights[k] - DEFAULT_WEIGHTS[k], 2)
                      for k in LEARN_KEYS
                      if abs(self.weights[k] - DEFAULT_WEIGHTS[k]) >= 0.01},
            'opponent': self.opponent,
            'vsOpponent': dict(opp) if opp else None,
            'favouriteOpening': favourite,
            'recent': self.history[-10:],
            'dataFile': self.data_path,
        }

    def progress_report(self):
        """The same thing as a few lines of text, for a terminal or a chat
        message after the game."""
        s = self.stats()
        r = s['record']
        out = ['%s (%s) — %d played: %dW %dL %dD (%.0f%% wins)'
               % (s['name'], s['style'], r['played'], r['won'], r['lost'],
                  r['drawn'], s['winRate'])]
        if s['style'] == 'learner':
            out.append('No search at all · book holds %d positions (%d moves) '
                       '· exploration %.2f'
                       % (s['bookPositions'], s['bookMoves'], s['exploration']))
            if s['policyLearned']:
                bits = ', '.join('%s %+.2f' % (k, v) for k, v in
                                 sorted(s['policyLearned'].items(),
                                        key=lambda kv: -abs(kv[1]))[:5])
                out.append('Move policy so far: ' + bits)
            else:
                out.append('Move policy is still all zeros — it is picking at '
                           'random. Give it games (self_play) before judging it.')
        else:
            out.append('Search depth %d of %d · book holds %d positions (%d moves)'
                       % (s['searchDepth'], s['maxDepth'], s['bookPositions'],
                          s['bookMoves']))
        if s['drift']:
            bits = ', '.join('%s %+.1f' % (k, v) for k, v in
                             sorted(s['drift'].items(), key=lambda kv: -abs(kv[1]))[:4])
            out.append('Weights drifted from default: ' + bits)
        else:
            out.append('Weights still at their defaults — that takes hundreds '
                       'of games, or a self_play() run.')
        if s['favouriteOpening']:
            out.append('Your most-played opening: ' + s['favouriteOpening'])
        return '\n'.join(out)

    # ------------------------------------------------------------------
    #  Offline bootstrap
    # ------------------------------------------------------------------
    def train_against(self, teacher=None, games=50, max_plies=140,
                      verbose=False):
        """Learn by being beaten by something that can see.

        This is how a learner should be bootstrapped, and self_play() is not.
        Measured over ~2,400 decisions: against another learner, a move that
        hangs a pawn or more scores -0.070 against -0.028 for a safe move — a
        gap of only 0.098, because an opponent with no lookahead usually fails
        to take what you left out. Against a searcher the same comparison is
        -0.512 against -0.215, a gap of 0.297. Three times the signal, on the
        one lesson a searchless bot cannot afford to miss.

        The consequence is worth stating plainly: a blind player cannot teach
        itself not to blunder, because its sparring partner does not punish it.
        It needs an opponent that does.

        Only this AI learns; the teacher is a throwaway.
        """
        if self.style != 'learner':
            raise ValueError('train_against is for learners; searchers use '
                             'self_play')
        if teacher is None:
            teacher = ChessAI(name='teacher', autoload=False, style='searcher',
                              think_time=0.05, max_depth=2,
                              seed=self.rng.randrange(1 << 30))
            teacher.record['played'] = 40          # skip its warm-up ramp
        tally = {'games': 0, 'won': 0, 'lost': 0, 'drawn': 0}
        for i in range(games):
            board = chess.Board()
            self.new_game()
            teacher.new_game()
            mine = chess.WHITE if i % 2 == 0 else chess.BLACK
            while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
                who = self if board.turn == mine else teacher
                uci = who.get_move(board)
                if not uci:
                    break
                board.push(chess.Move.from_uci(uci))

            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                result, tag = 0.5, 'drawn'
            elif outcome.winner == mine:
                result, tag = 1.0, 'won'
            else:
                result, tag = 0.0, 'lost'
            tally[tag] += 1
            tally['games'] += 1
            self.learn_from_game(result, final_board=board, opponent='teacher')
            if verbose and (i + 1) % 10 == 0:
                print('  %d/%d: %dW %dL %dD' % (i + 1, games, tally['won'],
                                                tally['lost'], tally['drawn']))
        return tally

    def self_play(self, games=10, think_time=0.05, max_depth=2,
                  max_plies=160, verbose=False):
        """Play the AI against itself to give the weight learning a sample size
        a human never will.

        This is the honest answer to "how do the evaluation weights ever get
        anywhere": a person plays ten games a week, and ten games move a weight
        vector by nothing. A few hundred self-play games at shallow depth take
        minutes and actually shift it.

        Deliberately fast and shallow — the point is volume of gradient steps,
        not quality of play. Returns a small tally.
        """
        tally = {'games': 0, 'white': 0, 'black': 0, 'draw': 0}
        keep_time, keep_depth = self.think_time, self.max_depth
        self.think_time, self.max_depth = think_time, max_depth
        try:
            for i in range(games):
                board = chess.Board()
                self.new_game()
                while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
                    uci = self.get_move(board)
                    if not uci:
                        break
                    board.push(chess.Move.from_uci(uci))

                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    result, tag = 0.5, 'draw'
                elif outcome.winner == chess.WHITE:
                    result, tag = 1.0, 'white'
                else:
                    result, tag = 1.0, 'black'
                tally[tag] += 1
                tally['games'] += 1

                # Both sides were this same brain, so `winner` is what makes
                # the credit land correctly: each traced move is judged by
                # whether its own colour won, rather than half the moves being
                # told a loss was a win.
                self.learn_from_game(
                    0.5 if tag == 'draw' else result, final_board=board,
                    winner={'white': 'w', 'black': 'b', 'draw': 'draw'}[tag])
                if verbose:
                    print('  game %d/%d: %s in %d plies'
                          % (i + 1, games, tag, board.ply()))
        finally:
            self.think_time, self.max_depth = keep_time, keep_depth
        return tally


# ============================================================
#  Self-test and demo
# ============================================================
def _selftest():
    """Cheap invariants worth having, since the failures they catch are the
    silent kind — a mirrored table or a sign flip does not crash, it just makes
    the AI play badly for reasons nobody can see."""
    ok = 0

    ai = ChessAI(data_path=os.path.join(BASE, '_selftest_chess_ai.json'),
                 autoload=False, seed=1)

    # 1. A fresh board is symmetric: every differential feature must be zero.
    f = features(chess.Board())
    bad = {k: v for k, v in f.items() if abs(v) > 1e-9}
    assert not bad, 'start position is not symmetric: %s' % bad
    ok += 1

    # 2. Evaluation agrees with the dot product it is supposed to be.
    b = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1')
    manual = sum(ai.weights[k] * features(b)[k] for k in FEATURE_KEYS)
    assert abs(ai.evaluate(b) - manual) < 1e-6, 'evaluate() != w.f'
    ok += 1

    # 3. Being a knight up must evaluate better for the side that is up.
    up = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1')
    assert ai.evaluate(up) > 0, 'a piece up should favour the side to move'
    ok += 1

    # 4. PST orientation: a white knight on e4 must beat one on a1, and the
    #    mirrored claim must hold for Black.
    w_good = features(chess.Board('4k3/8/8/8/4N3/8/8/4K3 w - - 0 1'))['pst']
    w_bad = features(chess.Board('4k3/8/8/8/8/8/8/N3K3 w - - 0 1'))['pst']
    assert w_good > w_bad, 'white PST is upside down'
    b_good = features(chess.Board('4k3/8/8/4n3/8/8/8/4K3 w - - 0 1'))['pst']
    b_bad = features(chess.Board('n3k3/8/8/8/8/8/8/4K3 w - - 0 1'))['pst']
    assert b_good < b_bad, 'black PST is not mirrored'
    ok += 1

    # 5. Mate in one, found at depth 2. If this breaks, search is broken.
    mate = chess.Board('6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1')
    mv, _, _ = ai.search(mate, think_time=5.0, max_depth=3)
    mate.push(mv)
    assert mate.is_checkmate(), 'missed mate in one, played %s' % mv
    ok += 1

    # 6. Every move it returns is legal, from a few awkward positions.
    for fen in ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'k7/8/8/8/8/8/6q1/7K w - - 0 1',             # exactly one legal move
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1']:     # castling rights
        board = chess.Board(fen)
        uci = ai.get_move(board)
        assert uci is not None
        assert chess.Move.from_uci(uci) in board.legal_moves, 'illegal: %s' % uci
    ok += 1

    # 7. Result parsing, in all the shapes a caller might have to hand.
    for win in ('win', 'Won', '1-0', 1, 1.0, {'result': 'win'}):
        assert ChessAI._normalize_result(win) == 1.0, win
    for loss in ('loss', 'lost', '0-1', 0, 0.0, {'score': 'loss'}):
        assert ChessAI._normalize_result(loss) == 0.0, loss
    for draw in ('draw', 'tie', '1/2-1/2', 0.5, {'outcome': 'draw'}):
        assert ChessAI._normalize_result(draw) == 0.5, draw
    ok += 1

    # 8. Learning runs, records the game, and fills the book.
    ai.new_game()
    board = chess.Board()
    for _ in range(6):
        uci = ai.get_move(board)
        board.push(chess.Move.from_uci(uci))
    summary = ai.learn_from_game('loss', final_board=board)
    assert ai.record['played'] == 1 and ai.record['lost'] == 1
    assert summary['book_updated'] > 0 and len(ai.book) > 0
    ok += 1

    # 9. A save/load round trip preserves the brain exactly.
    ai.weights['knight'] = 333.5
    ai.save_progress()
    twin = ChessAI(data_path=ai.data_path, autoload=True)
    assert abs(twin.weights['knight'] - 333.5) < 1e-6, 'weights did not survive'
    assert twin.book == ai.book, 'book did not survive'
    assert twin.record == ai.record, 'record did not survive'
    ok += 1

    # 10. reset() really does clear it, on disk as well as in memory.
    twin.reset()
    assert twin.record['played'] == 0 and not twin.book
    third = ChessAI(data_path=ai.data_path, autoload=True)
    assert third.record['played'] == 0 and not third.book, 'reset was not persisted'
    ok += 1

    # 11. Weights stay inside their bounds however hard they are pushed.
    ai.weights['knight'] = 999.0
    assert ai._clamp('knight', ai.weights['knight']) <= WEIGHT_BOUNDS['knight'][1]
    ai.weights = dict(DEFAULT_WEIGHTS)
    for _ in range(12):
        ai.new_game()
        b = chess.Board()
        for _ in range(8):
            u = ai.get_move(b)
            if not u:
                break
            b.push(chess.Move.from_uci(u))
        ai.learn_from_game(ai.rng.choice(['win', 'loss', 'draw']), final_board=b)
    for k in LEARN_KEYS:
        lo, hi = WEIGHT_BOUNDS[k]
        assert lo <= ai.weights[k] <= hi, '%s escaped its bounds: %s' % (k, ai.weights[k])
    ok += 1

    # 12. TD scale consistency. In a position that is winning and *stable* —
    #     the evaluation two plies on is the same as it was — there is nothing
    #     to correct, so the weights must barely move. This is the exact check
    #     that catches the failure described in _learn_weights(): with V(s) and
    #     the target on mismatched scales, this case produced a large negative
    #     error and dragged every weight down at once.
    probe = ChessAI(data_path=os.path.join(BASE, '_selftest_td.json'),
                    autoload=False, seed=5)
    pos = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 0 1')      # White a rook up
    ev = probe.evaluate(pos)                                  # white-positive
    assert ev > 0
    probe._trace = [{'key': ChessAI.book_key(pos), 'uci': 'a1a2', 'ply': 2,
                     'colour': 'w', 'before': ev, 'after': ev,
                     'source': 'search', 'features': features(pos)}]
    snapshot = dict(probe.weights)
    probe.learn_from_game('win', winner='w')
    worst = max(abs(probe.weights[k] - snapshot[k]) for k in LEARN_KEYS)
    assert worst < 1.0, ('a stable winning position moved a weight by %.2f; '
                         'V(s) and the TD target are on different scales' % worst)
    ok += 1

    # 13. A learner really does not search, and still returns legal moves.
    lrn = ChessAI(data_path=os.path.join(BASE, '_selftest_learner.json'),
                  autoload=False, seed=11, style='learner')
    assert all(v == 0.0 for v in lrn.policy.values()), 'a new policy must be blank'
    for fen in [chess.STARTING_FEN,
                'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
                'k7/8/8/8/8/8/6q1/7K w - - 0 1']:
        b = chess.Board(fen)
        lrn._nodes = 0
        uci = lrn.get_move(b)
        assert uci and chess.Move.from_uci(uci) in b.legal_moves, uci
        assert lrn._nodes == 0, 'a learner searched %d nodes' % lrn._nodes
    ok += 1

    # 14. The policy gradient moves the right weight in the right direction.
    #     One decision: the move played captured, the alternatives on average
    #     did not, and the position improved. `capture` must go up, and the
    #     features it shared with the alternatives must not move.
    lrn.policy = dict(DEFAULT_POLICY)
    blank = dict.fromkeys(MOVE_FEATURE_KEYS, 0.0)
    chosen = dict(blank); chosen['capture'] = 1.0; chosen['develop'] = 1.0
    average = dict(blank); average['develop'] = 1.0      # shared, so no lesson
    lrn._trace = [{'key': 'x', 'uci': 'a1a2', 'ply': 4, 'colour': 'w',
                   'before': 0.0, 'after': 300.0, 'source': 'policy',
                   'features': features(chess.Board()),
                   'pf': chosen, 'pe': average}]
    lrn.learn_from_game('win', winner='w')
    assert lrn.policy['capture'] > 0.0, 'capturing well taught it nothing'
    assert abs(lrn.policy['develop']) < 1e-9, (
        'a feature shared with the alternatives must not move: %s'
        % lrn.policy['develop'])
    ok += 1

    # 15. Style and policy survive a save/load round trip.
    lrn.save_progress()
    twin2 = ChessAI(data_path=lrn.data_path, autoload=True, style='learner')
    assert twin2.style == 'learner'
    assert abs(twin2.policy['capture'] - lrn.policy['capture']) < 1e-6
    ok += 1

    for path in (ai.data_path, ai.data_path + '.tmp',
                 lrn.data_path, lrn.data_path + '.tmp',
                 probe.data_path, probe.data_path + '.tmp'):
        if os.path.exists(path):
            os.remove(path)
    print('self-test: %d checks passed' % ok)


def _demo(games, think_time):
    """A short human-free demo: the AI plays itself, learns, and reports."""
    path = os.path.join(BASE, 'chess_ai_demo.json')
    ai = ChessAI(name='Demo', data_path=path, opponent='self', autoload=False,
                 think_time=think_time, max_depth=2, seed=7)
    print('Self-playing %d games (this is the offline bootstrap, not a match)...'
          % games)
    start = time.monotonic()
    tally = ai.self_play(games, think_time=think_time, max_depth=2, verbose=True)
    took = time.monotonic() - start
    print('\n%d games in %.1fs — white %d, black %d, draw %d'
          % (tally['games'], took, tally['white'], tally['black'], tally['draw']))
    print('\n' + ai.progress_report())
    ai.save_progress()
    print('\nBrain written to %s' % path)


def _play(think_time, depth):
    """Play a game at the terminal. White is you, Black is the AI — the same
    arrangement the server uses."""
    ai = ChessAI(name='Terminal', opponent='terminal', think_time=think_time,
                 max_depth=depth)
    board = chess.Board()
    print(ai.progress_report())
    print('\nYou are White. Enter moves as UCI (e2e4) or SAN (Nf3). '
          '"quit" to stop.\n')
    while not board.is_game_over(claim_draw=True):
        print(board.unicode(invert_color=True, borders=False))
        if board.turn == chess.WHITE:
            raw = input('\nYour move: ').strip()
            if raw.lower() in ('quit', 'exit', 'q'):
                print('Game abandoned — nothing learned.')
                return
            try:
                move = board.parse_san(raw)
            except ValueError:
                try:
                    move = chess.Move.from_uci(raw)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print('  Not a legal move.')
                    continue
            board.push(move)
        else:
            uci = ai.get_move(board)
            move = chess.Move.from_uci(uci)
            print('\n%s plays %s' % (ai.name, board.san(move)))
            board.push(move)

    print(board.unicode(invert_color=True, borders=False))
    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        print('\nDraw — %s' % outcome.termination.name.lower())
        result = 'draw'
    elif outcome.winner == chess.BLACK:
        print('\n%s wins.' % ai.name)
        result = 'win'
    else:
        print('\nYou win.')
        result = 'loss'
    ai.learn_from_game(result, final_board=board)
    ai.save_progress()
    print('\n' + ai.progress_report())


def main():
    ap = argparse.ArgumentParser(description='Learning chess AI for HomeGames.')
    ap.add_argument('--test', action='store_true', help='run the self-test')
    ap.add_argument('--selfplay', type=int, metavar='N',
                    help='self-play N games and report (for searchers)')
    ap.add_argument('--spar', type=int, metavar='N',
                    help='train a learner for N games against a searcher — the '
                         'right way to bootstrap one, see train_against()')
    ap.add_argument('--play', action='store_true',
                    help='play a game at the terminal as White')
    ap.add_argument('--stats', action='store_true',
                    help='print the saved brain and exit')
    ap.add_argument('--reset', action='store_true', help='wipe the saved brain')
    ap.add_argument('--think', type=float, default=0.3,
                    help='seconds per move (default 0.3)')
    ap.add_argument('--depth', type=int, default=4, help='max search depth')
    args = ap.parse_args()

    if args.reset:
        ChessAI().reset()
        print('Brain reset.')
        return
    if args.stats:
        print(ChessAI().progress_report())
        return
    if args.test:
        _selftest()
        return
    if args.spar:
        path = os.path.join(BASE, 'chess_ai_learner.json')
        lrn = ChessAI(name='Learner', data_path=path, style='learner')
        print('Training a searchless learner for %d games against a searcher...'
              % args.spar)
        start = time.monotonic()
        tally = lrn.train_against(games=args.spar, verbose=True)
        print('\n%d games in %.0fs — %dW %dL %dD'
              % (tally['games'], time.monotonic() - start,
                 tally['won'], tally['lost'], tally['drawn']))
        lrn.save_progress()
        print('\n' + lrn.progress_report())
        print('\nBrain written to %s' % path)
        return
    if args.selfplay:
        _demo(args.selfplay, args.think)
        return
    if args.play:
        _play(args.think, args.depth)
        return

    _selftest()
    print('\nNothing else asked for. Try --play, --selfplay 20, or --stats.')


if __name__ == '__main__':
    main()
