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
The obvious method — play each rung against the rung above it and add up the
gaps — does not work here, and the failure is worth recording so nobody tries
it again. The top two searchers share a depth ceiling of 4 and differ only in
blunder rate, 2% against 10.3%, and that alone was a 16-0 sweep. A blunder in
Quixx costs a few points; in chess it hangs a piece and loses the game, so the
rungs are hundreds of Elo apart and every link saturates.

So the references are built at a spacing we control instead:

1. Stockfish under UCI_LimitStrength at its 1320 floor is the fixed point.
2. Weaker references are the *same engine* under a known random-move rate —
   the same handicap the bots wear — each one rated by playing the reference
   above it. That reaches far below anything Stockfish will pin itself to.
3. Rungs strong enough to score against 1320 are read directly off it. The
   rest are matched against whichever reference is close enough to resolve
   them, which is any score that is not 0 or 1.

Everything is measured **as the server seats it**, not as a bare engine:
Chess.bot_step() rolls against the blunder rate and plays a random move without
consulting the ChessAI at all, and depth_ceiling() caps the search. Both come
off the same handicap, and leaving either out measures something no player ever
faces.

Caveats worth keeping
---------------------
* Elo below roughly 800 is extrapolation more than measurement — the scale was
  never built down there, and the low references rest on scores like 1-in-16.
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
MAX_PLIES = 140

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
ANCHORS = [0, 2]
HANDICAPS = [0.20, 0.45, 0.75]   # random-move rates for the weaker references


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


class Ref:
    """Stockfish at the 1320 floor, optionally wearing a random-move handicap
    so it can be placed anywhere below it."""

    def __init__(self, engine, blunder=0.0, elo=None, seed=5):
        self.engine, self.blunder, self.elo = engine, blunder, elo
        self.rng = random.Random(seed)
        self.limit = chess.engine.Limit(time=SF_TIME)

    def label(self):
        return 'sf' if not self.blunder else 'sf+%.0f%%' % (self.blunder * 100)

    def move(self, board):
        legal = list(board.legal_moves)
        if self.blunder and self.rng.random() < self.blunder:
            return self.rng.choice(legal)
        return self.engine.play(board, self.limit).move


def duel(a, b, games):
    """a against b, colours alternating. Returns a's score share."""
    pts = 0.0
    for i in range(games):
        board = chess.Board()
        for side in (a, b):
            if hasattr(side, 'ai'):
                side.ai.new_game()
        mine = chess.WHITE if i % 2 == 0 else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
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
    ap.add_argument('--ref-games', type=int, default=16,
                    help='games per reference link and per placement')
    ap.add_argument('--quick', action='store_true',
                    help='fewer games: rough placement, wide error bars')
    ap.add_argument('--json', metavar='PATH', help='also write the raw results')
    args = ap.parse_args()
    if args.quick:
        args.anchor_games, args.ref_games = 12, 8

    sf = shutil.which('stockfish') or '/usr/games/stockfish'
    if not os.path.exists(sf):
        sys.exit('needs Stockfish on the PATH: apt install stockfish')

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': SF_ELO,
                      'Threads': 1, 'Hash': 16})
    t0 = time.time()
    elo = {}

    # --- rungs strong enough to face Stockfish directly ------------------
    print('\nAnchors — %d games vs Stockfish @ %d (%.1fs/move)\n'
          % (args.anchor_games, SF_ELO, SF_TIME))
    print('%-18s %-8s %-12s' % ('bot', 'score', 'Elo'))
    print('-' * 40)
    top = Ref(engine, 0.0, float(SF_ELO))
    for i in ANCHORS:
        name, blunder, style = CONFIGS[i]
        s = duel(Seat(name, blunder, style, seed=i * 13), top, args.anchor_games)
        g = elo_gap(s)
        if g is not None:
            elo[i] = SF_ELO + g
        print('%-18s %-8.3f %-12s'
              % (name, s, '%.0f' % elo[i] if i in elo else 'saturated'), flush=True)

    # --- references below the floor --------------------------------------
    print('\nReference ladder — %d games per link\n' % args.ref_games)
    print('%-14s %-10s %-8s' % ('reference', 'vs above', 'Elo'))
    print('-' * 34)
    print('%-14s %-10s %-8.0f' % (top.label(), '—', top.elo), flush=True)
    refs = [top]
    for h in HANDICAPS:
        r = Ref(engine, h)
        above = refs[-1]
        s = duel(r, above, args.ref_games)
        g = elo_gap(s)
        r.elo = above.elo + (g if g is not None else -400.0)
        refs.append(r)
        print('%-14s %-10.3f %-8.0f%s'
              % (r.label(), s, r.elo, '  (saturated, floored)' if g is None else ''),
              flush=True)

    # --- everything else, against a reference that can resolve it --------
    print('\nPlacements — %d games per cell\n' % args.ref_games)
    print('%-18s %-30s %-8s %s' % ('bot', 'readings', 'Elo', 'via'))
    print('-' * 68)
    for i, (name, blunder, style) in enumerate(CONFIGS):
        if i in elo:
            print('%-18s %-30s %-8.0f %s' % (name, '(anchored)', elo[i], 'sf'))
            continue
        readings, cells = [], []
        # Middle of the ladder first: a reading near 0.5 is worth far more than
        # one scraped off the edges, where one game swings the estimate wildly.
        for r in sorted(refs, key=lambda x: abs(x.elo - 600)):
            s = duel(Seat(name, blunder, style, seed=i * 13), r, args.ref_games)
            cells.append('%s:%.2f' % (r.label(), s))
            g = elo_gap(s)
            if g is not None:
                readings.append((abs(s - 0.5), r.elo + g, r.label()))
            if 0.05 < s < 0.95:
                break
        best = min(readings) if readings else None
        if best:
            elo[i] = best[1]
        print('%-18s %-30s %-8s %s'
              % (name, ' '.join(cells), '%.0f' % elo[i] if i in elo else 'unplaced',
                 best[2] if best else '—'), flush=True)
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
                       'ref_games': args.ref_games,
                       'refs': [{'label': r.label(), 'elo': r.elo} for r in refs],
                       'bots': [{'name': n, 'blunder': b, 'style': s,
                                 'elo': elo.get(i)}
                                for i, (n, b, s) in enumerate(CONFIGS)]}, f, indent=1)


if __name__ == '__main__':
    main()
