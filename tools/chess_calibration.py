#!/usr/bin/env python3
"""Measure what the chess bots are actually worth, on the real chess scale.

Why this exists
---------------
A rating earned in a closed pool says who is stronger, never how strong. The
bots play thousands of games against each other and a handful against people,
so their ELOs settle into a stable order around whatever ELO_START happened to
be — and the lobby bills that number as the difficulty ("pick a 1350 bot for a
hard game"). For that to be honest the scale has to be anchored to something
outside this house.

Chess is the one game here with such a yardstick. This plays each roster
configuration against Stockfish under UCI_LimitStrength at a known Elo and
converts the score to a rating.

Two details that matter for the number to mean anything:

* The bots are measured **as the server seats them**, not as bare engines.
  Chess.bot_step() rolls against the blunder rate and plays a random move
  instead of asking the ChessAI at all, and depth_ceiling() caps the search —
  both come off the same handicap, and leaving either out measures something
  the player never faces.
* Configurations at the bottom of the roster score zero against Stockfish's
  1320 floor, and zero carries no information beyond "weaker". Those are placed
  by chaining: each rung plays the rung above it, where the score is
  informative, and the gaps are added up from an anchored rung.

Needs Stockfish on the PATH (apt install stockfish) and python-chess. Neither is
a runtime dependency of the server — this is a development tool, run when the
search, the evaluation or depth_ceiling() changes and the table in
homegames_server.py goes stale.

    python tools/chess_calibration.py              # full run, ~90 minutes
    python tools/chess_calibration.py --quick      # rough pass, ~15 minutes
"""
import argparse
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

# Real game conditions, straight from the server.
THINK = 0.6                 # Chess.THINK_TIME
SF_ELO = 1320               # Stockfish 16's UCI_Elo floor
SF_TIME = 0.3               # UCI_LimitStrength is calibrated for a real clock

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
# Neighbouring rungs. The learner is tied in at two points so its placement does
# not hang on a single pairing.
LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 4), (5, 3)]
ANCHORS = [0, 2]            # measured against Stockfish directly


def depth_ceiling(blunder):
    """Chess.depth_ceiling, copied rather than imported: importing the server
    pulls in the HTTP stack and its globals for the sake of one formula."""
    return max(2, min(4, int(round(4.5 - 6.0 * blunder))))


class Seat:
    """A bot exactly as the server seats it: a ChessAI under the blunder
    wrapper from Chess.bot_step()."""

    def __init__(self, name, blunder, style='searcher', seed=0):
        self.name, self.blunder, self.style = name, blunder, style
        self.depth = depth_ceiling(blunder)
        self.ai = ChessAI(name=name, autoload=False, style=style,
                          think_time=THINK, max_depth=self.depth, seed=seed)
        # Measured at its ceiling. skill_level() ramps a bot from depth 2 to
        # its cap over its first dozen games, so a brand-new bot is weaker than
        # this — but brains persist and training adds games in twenties, so the
        # ceiling is what a person actually meets.
        self.ai.record['played'] = 40
        self.rng = random.Random(seed + 7)
        self.blunders = 0

    def move(self, board):
        legal = list(board.legal_moves)
        if self.blunder and self.rng.random() < self.blunder:
            self.blunders += 1
            return self.rng.choice(legal)
        uci = self.ai.get_move(board)
        if uci:
            m = chess.Move.from_uci(uci)
            if m in legal:
                return m
        return self.rng.choice(legal)


def elo_gap(score):
    """Rating difference implied by a score share. None at 0 or 1, where the
    difference is unbounded and the honest answer is "further than this many
    games can see"."""
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def vs_engine(seat, engine, games, max_plies=200):
    pts, w, l, d = 0.0, 0, 0, 0
    limit = chess.engine.Limit(time=SF_TIME)
    for i in range(games):
        board = chess.Board()
        seat.ai.new_game()
        mine = chess.WHITE if i % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == mine:
                board.push(seat.move(board))
            else:
                board.push(engine.play(board, limit).move)
        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            pts += 0.5; d += 1
        elif o.winner == mine:
            pts += 1.0; w += 1
        else:
            l += 1
    return pts / games, (w, l, d)


def duel(a, b, games, max_plies=160):
    pts = 0.0
    for i in range(games):
        board = chess.Board()
        a.ai.new_game(); b.ai.new_game()
        mine = chess.WHITE if i % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            who = a if board.turn == mine else b
            board.push(who.move(board))
        o = board.outcome(claim_draw=True)
        pts += 0.5 if (o is None or o.winner is None) else (1.0 if o.winner == mine else 0.0)
    return pts / games


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--anchor-games', type=int, default=40)
    ap.add_argument('--link-games', type=int, default=16)
    ap.add_argument('--quick', action='store_true',
                    help='fewer games; rough placement, wide error bars')
    args = ap.parse_args()
    if args.quick:
        args.anchor_games, args.link_games = 12, 8

    sf = shutil.which('stockfish') or '/usr/games/stockfish'
    if not os.path.exists(sf):
        sys.exit('needs Stockfish on the PATH: apt install stockfish')

    seats = [Seat(n, b, s, seed=i * 13) for i, (n, b, s) in enumerate(CONFIGS)]
    t0 = time.time()

    # --- absolute anchors ------------------------------------------------
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': SF_ELO,
                      'Threads': 1, 'Hash': 16})
    print('\nAnchors — %d games vs Stockfish @ UCI_Elo %d (%.1fs/move)\n'
          % (args.anchor_games, SF_ELO, SF_TIME))
    print('%-18s %-8s %-9s %-12s' % ('bot', 'score', 'W-L-D', 'implied Elo'))
    print('-' * 50)
    anchored = {}
    for i in ANCHORS:
        score, (w, l, d) = vs_engine(seats[i], engine, args.anchor_games)
        gap = elo_gap(score)
        elo = None if gap is None else SF_ELO + gap
        if elo is not None:
            anchored[i] = elo
        print('%-18s %-8.3f %-9s %-12s'
              % (CONFIGS[i][0], score, '%d-%d-%d' % (w, l, d),
                 '%.0f' % elo if elo else 'saturated'), flush=True)
    engine.quit()
    if not anchored:
        sys.exit('no anchor scored against Stockfish — nothing to pin the scale to')

    # --- relative chain --------------------------------------------------
    print('\nChain — %d games per link\n' % args.link_games)
    print('%-18s %-18s %-8s %-9s' % ('stronger', 'weaker', 'score', 'Elo gap'))
    print('-' * 56)
    gaps = {}
    for i, j in LINKS:
        s = duel(seats[i], seats[j], args.link_games)
        g = elo_gap(s)
        gaps[(i, j)] = g
        print('%-18s %-18s %-8.3f %-9s'
              % (CONFIGS[i][0], CONFIGS[j][0], s,
                 '%+.0f' % g if g is not None else 'saturated'), flush=True)

    # --- solve for a rating per configuration ----------------------------
    # Least squares over the observed gaps, with the anchored rungs pinned.
    elo = {i: anchored.get(i, sum(anchored.values()) / len(anchored))
           for i in range(len(CONFIGS))}
    for _ in range(20000):
        for (i, j), g in gaps.items():
            if g is None:
                continue
            err = (elo[i] - elo[j]) - g
            step = 0.01 * err
            if i not in anchored:
                elo[i] -= step
            if j not in anchored:
                elo[j] += step
        for i, v in anchored.items():
            elo[i] = v

    print('\nCalibrated ratings\n')
    print('%-18s %-9s %-7s %s' % ('bot', 'blunder', 'depth', 'Elo'))
    print('-' * 46)
    for i, (name, blunder, style) in enumerate(CONFIGS):
        print('%-18s %-9.3f %-7s %.0f'
              % (name, blunder, '—' if style == 'learner' else seats[i].depth,
                 elo[i]))

    print('\nPaste into homegames_server.py:\n')
    print('CHESS_ELO_SEARCHER = [')
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if style == 'searcher':
            print('    (%.3f, %.0f),' % (blunder, elo[i]))
    print(']')
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if style == 'learner':
            print('CHESS_ELO_LEARNER = %.0f' % elo[i])
    se = math.sqrt(0.25 / args.anchor_games)
    print('\nanchor standard error +-%.3f (~+-%.0f Elo), %.0f minutes'
          % (se, 695 * se, (time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()
