#!/usr/bin/env python3
"""Measure what the chess bots are actually worth, on the real chess scale.

Why this exists
---------------
A rating earned in a closed pool says who is stronger, never how strong. The
bots play thousands of games against each other and a handful against people,
so their ELOs settle into a stable order around whatever ELO_START happened to
be — and the lobby bills that number as the difficulty ("pick a 1050 bot for a
gentle game or a 1350 bot for a hard one"). For that to be honest the scale has
to be tied to something outside this house.

Chess is the one game here with such a yardstick.

How it measures
---------------
Two methods were tried and abandoned before this one, and both failures are
worth recording so nobody repeats them.

*Chaining rung to rung does not work.* The top two searchers share a depth
ceiling of 4 and differ only in blunder rate, 2% against 10.3%, and that
pairing finished 16-0. Measured against Stockfish the same two rungs are 131
Elo apart, which predicts a 0.32 score, not a sweep. Two near-identical engines
make the blunder rate the only variable in the game, so it decides nearly every
one — a maximally discriminating matchup that says almost nothing about how
either fares against a different opponent. A person is a different opponent.

*Handicapping Stockfish with random moves does not work either.* It makes the
engine erratic rather than weak — brilliant play punctuated by free pieces —
and nothing rates consistently against that. Two rungs whose Elo was already
known disagreed by 476 points about the same handicapped reference.

What does work is measuring every rung against opponents that are weak but
*consistent*, and validating each reference against a rung already placed:

1. Stockfish under UCI_LimitStrength at its 1320 floor rates the top of the
   roster directly, wherever a rung can score against it at all.
2. Below that, hand-built deterministic references: a 1-ply material player,
   and a random mover as the floor. The material player is rated by playing a
   rung whose Elo already came from Stockfish, so it is checked, not assumed.

Depth-limited Stockfish was measured too and is transitive with these bots —
it agreed with the 1320 floor about a known rung to within ~90 points — but it
is not weak enough to be useful: even a 1-ply search rates around 1230, because
the evaluation carries it regardless of depth.

Everything is measured **as the server seats it**, not as a bare engine:
Chess.bot_step() rolls against the blunder rate and plays a random move without
consulting the ChessAI at all, and depth_ceiling() caps the search. Both come
off the same handicap, and leaving either out measures something no player ever
faces.

Caveats worth keeping
---------------------
* The ratings describe how a rung fares against a *foreign* opponent, which is
  what a person is. They deliberately do not reproduce the bots' results
  against each other, which are far more spread out for the reason above.
* Elo below roughly 800 is extrapolation more than measurement — the scale was
  never built down there.
* The bots are measured at their ceiling (record['played'] past the ramp).
  skill_level() starts every bot at depth 2 and climbs over its first dozen
  games, so a brand-new bot is weaker than its rating until it has played in.

Needs Stockfish on the PATH (apt install stockfish) and python-chess. Neither
is a runtime dependency of the server — this is a development tool, run when
the search, the evaluation or depth_ceiling() changes and the table in
homegames_server.py goes stale.

    python tools/chess_calibration.py            # ~2 hours, tight
    python tools/chess_calibration.py --quick    # ~30 minutes, rough
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chess
    import chess.engine
except ImportError:
    sys.exit('needs python-chess: pip install chess')

from chess_ai import ChessAI

THINK = 0.6            # Chess.THINK_TIME — the real per-move budget
SF_ELO = 1320          # Stockfish 16's UCI_Elo floor
SF_TIME = 0.3          # UCI_LimitStrength is calibrated for a real clock
MAX_PLIES = 300        # these players win material long before they can mate

# The roster as new_roster() mints it: five searchers with the blunder rate
# spread evenly, plus the searchless learners.
CONFIGS = [
    ('searcher b.02 d4', 0.020, 'searcher'),
    ('searcher b.10 d4', 0.103, 'searcher'),
    ('searcher b.19 d3', 0.185, 'searcher'),
    ('searcher b.27 d3', 0.268, 'searcher'),
    ('searcher b.35 d2', 0.350, 'searcher'),
    ('learner  b.02',    0.020, 'learner'),
]
# Rungs measured against Stockfish directly. The rest are placed off the
# handicapped references; these two also cross-check that ladder.
ANCHORS = [0, 1, 2]     # rungs that can score against Stockfish's floor
VALIDATE = 2            # the rung used to rate the weak references
VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5}


def depth_ceiling(blunder):
    """Chess.depth_ceiling, copied rather than imported: importing the server
    pulls in the HTTP stack and its globals for the sake of one formula."""
    return max(2, min(4, int(round(4.5 - 6.0 * blunder))))


def elo_gap(score):
    """Rating difference implied by a score share, or None at 0 and 1 where it
    is unbounded and the honest answer is "further than this many games see"."""
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


class Seat:
    """A bot exactly as the server seats it: a ChessAI under the blunder
    wrapper from Chess.bot_step()."""

    def __init__(self, name, blunder, style='searcher', seed=0):
        self.name, self.blunder, self.style = name, blunder, style
        self.depth = depth_ceiling(blunder)
        self.ai = ChessAI(name=name, autoload=False, style=style,
                          think_time=THINK, max_depth=self.depth, seed=seed)
        self.ai.record['played'] = 40         # past the skill_level() ramp
        self.rng = random.Random(seed + 7)

    def move(self, board):
        legal = list(board.legal_moves)
        if self.blunder and self.rng.random() < self.blunder:
            return self.rng.choice(legal)     # bypasses the ChessAI entirely
        uci = self.ai.get_move(board)
        if uci:
            m = chess.Move.from_uci(uci)
            if m in legal:
                return m
        return self.rng.choice(legal)


class Stock:
    """Stockfish under UCI_LimitStrength, the fixed point of the whole scale."""

    def __init__(self, engine, elo=SF_ELO):
        self.engine, self.elo = engine, float(elo)
        self.limit = chess.engine.Limit(time=SF_TIME)

    def label(self):
        return 'sf%d' % SF_ELO

    def move(self, board):
        return self.engine.play(board, self.limit).move


class Greedy:
    """1-ply material, and nothing else: the best capture net of what the
    destination square gives back. Weak, but never erratic — which is the
    property the random-move references lacked and why nothing could be rated
    against them consistently.

    The progress terms are not decoration. Without them it wins a queen and
    then shuffles until the position repeats: an early version went 9 wins, 31
    draws, none lost against a random mover, which scores 0.55 and reads as
    "barely better than random". A reference whose rating is set by its
    inability to convert is useless, because that inability does not transfer
    to how it fares against anything else. With them it wins 40 out of 40.
    """

    def __init__(self, seed=3):
        self.rng = random.Random(seed)
        self.elo = None

    def label(self):
        return 'greedy'

    def score(self, board, mv):
        s = 0.0
        victim = board.piece_type_at(mv.to_square)
        if victim:
            s += VAL[victim]
        elif board.is_en_passant(mv):
            s += VAL[chess.PAWN]
        mover = board.piece_at(mv.from_square)
        mine = VAL[mover.piece_type] if mover else 0
        if board.is_attacked_by(not board.turn, mv.to_square):
            defended = any(sq != mv.from_square
                           for sq in board.attackers(board.turn, mv.to_square))
            s -= mine * (0.35 if defended else 1.0)
        if mv.promotion:
            s += VAL.get(mv.promotion, 0)
        if mv.to_square in CENTRE:
            s += 12
        if mover and mover.piece_type == chess.PAWN:
            rank = chess.square_rank(mv.to_square)
            s += 8 * (rank if board.turn == chess.WHITE else 7 - rank)
        board.push(mv)
        try:
            if board.is_checkmate():
                s += 100000
            elif board.is_stalemate():
                s -= 5000              # never stalemate a lost king
            elif board.is_repetition(2):
                s -= 300               # a repeat throws away whatever is won
            elif board.is_check():
                s += 40                # drives the king, which is how it mates
        finally:
            board.pop()
        return s

    def move(self, board):
        best, pick = None, []
        for mv in board.legal_moves:
            v = self.score(board, mv)
            if best is None or v > best:
                best, pick = v, [mv]
            elif v == best:
                pick.append(mv)
        return self.rng.choice(pick)


class Rand:
    """Legal moves uniformly — the conventional floor of the scale."""

    def __init__(self, seed=4):
        self.rng = random.Random(seed)
        self.elo = None

    def label(self):
        return 'random'

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))


def duel(a, b, games, max_plies=MAX_PLIES):
    """a against b, colours alternating. Returns a's score share."""
    pts = 0.0
    for i in range(games):
        board = chess.Board()
        for side in (a, b):
            if hasattr(side, 'ai'):
                side.ai.new_game()
        mine = chess.WHITE if i % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            who = a if board.turn == mine else b
            board.push(who.move(board))
        o = board.outcome(claim_draw=True)
        pts += 0.5 if (o is None or o.winner is None) else (1.0 if o.winner == mine else 0.0)
    return pts / games


def main():
    ap = argparse.ArgumentParser(
        description='Calibrate the chess bots against Stockfish.',
        epilog='See the module docstring for the method and its caveats.')
    ap.add_argument('--anchor-games', type=int, default=40,
                    help='games per rung measured directly against Stockfish')
    ap.add_argument('--ref-games', type=int, default=24,
                    help='games per reference rating and per placement')
    ap.add_argument('--quick', action='store_true',
                    help='fewer games: rough placement, wide error bars')
    ap.add_argument('--json', metavar='PATH', help='also write the raw results')
    args = ap.parse_args()
    if args.quick:
        args.anchor_games, args.ref_games = 12, 10

    sf = shutil.which('stockfish') or '/usr/games/stockfish'
    if not os.path.exists(sf):
        sys.exit('needs Stockfish on the PATH: apt install stockfish')

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': SF_ELO,
                      'Threads': 1, 'Hash': 16})
    t0 = time.time()
    elo = {}

    # --- rungs strong enough to face Stockfish directly ------------------
    stock = Stock(engine)
    print('\nAnchors — %d games vs Stockfish @ %d (%.1fs/move)\n'
          % (args.anchor_games, SF_ELO, SF_TIME))
    print('%-18s %-8s %-12s' % ('bot', 'score', 'Elo'))
    print('-' * 40)
    for i in ANCHORS:
        name, blunder, style = CONFIGS[i]
        s = duel(Seat(name, blunder, style, seed=i * 13), stock, args.anchor_games)
        g = elo_gap(s)
        if g is not None:
            elo[i] = SF_ELO + g
        print('%-18s %-8.3f %-12s'
              % (name, s, '%.0f' % elo[i] if i in elo else 'saturated'), flush=True)

    # --- rate the weak reference against a rung already placed -----------
    if VALIDATE not in elo:
        engine.quit()
        sys.exit('the validating rung did not score against Stockfish; nothing '
                 'to rate the weak reference from')
    greedy = Greedy()
    name, blunder, style = CONFIGS[VALIDATE]
    s = duel(Seat(name, blunder, style, seed=99), greedy, args.ref_games)
    g = elo_gap(s)
    if g is None:
        engine.quit()
        sys.exit('greedy saturated against the validating rung; re-tune it')
    greedy.elo = elo[VALIDATE] - g
    print('\nWeak reference — rated against a rung Stockfish already placed\n')
    print('  %s (%.0f) vs greedy: %.3f  ->  greedy = %.0f'
          % (name, elo[VALIDATE], s, greedy.elo), flush=True)

    # --- everything Stockfish cannot reach -------------------------------
    print('\nPlacements — %d games per cell\n' % args.ref_games)
    print('%-18s %-12s %s' % ('bot', 'vs greedy', 'Elo'))
    print('-' * 44)
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if i in elo:
            continue
        sc = duel(Seat(name, blunder, style, seed=7), greedy, args.ref_games)
        gg = elo_gap(sc)
        if gg is not None:
            elo[i] = greedy.elo + gg
        print('%-18s %-12.3f %s'
              % (name, sc, '%.0f' % elo[i] if i in elo else 'saturated'), flush=True)
    engine.quit()

    print('\nPaste into homegames_server.py:\n')
    print('CHESS_ELO_SEARCHER = [')
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if style == 'searcher' and i in elo:
            print('    (%.3f, %.0f),' % (blunder, elo[i]))
    print(']')
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if style == 'learner' and i in elo:
            print('CHESS_ELO_LEARNER = %.0f' % elo[i])

    se = math.sqrt(0.25 / args.anchor_games)
    print('\nanchor standard error +-%.3f (~+-%.0f Elo near 50%%); '
          'placements are wider' % (se, 695 * se))
    print('%.0f minutes' % ((time.time() - t0) / 60.0))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'sf_elo': SF_ELO, 'anchor_games': args.anchor_games,
                       'ref_games': args.ref_games, 'greedy': greedy.elo,
                       'bots': [{'name': n, 'blunder': b, 'style': st,
                                 'elo': elo.get(i)}
                                for i, (n, b, st) in enumerate(CONFIGS)]}, f, indent=1)


if __name__ == '__main__':
    main()
