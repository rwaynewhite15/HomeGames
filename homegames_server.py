#!/usr/bin/env python3
"""
HomeGames — LAN game lobby server.
Pure Python standard library. No installs required.

Run:  python homegames_server.py
Then open http://localhost:4001 (or http://<your-LAN-IP>:4001 from other devices).

Platform: lobby with rooms you can host, join, or spectate.
Games: Quixx (standard / mixed colors / mixed numbers / both), Mancala (Kalah),
       Euchre (standard / stick the dealer). More games plug in later.
"""
import json
import os
import random
import socket
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue, Empty
from urllib.parse import urlparse, parse_qs

PORT = 4001
BASE = os.path.dirname(os.path.abspath(__file__))
SSE_PING = 5        # seconds; also how fast a closed tab leaves "who's here"

LOCK = threading.RLock()
CLIENTS = {}   # cid -> {'name': str, 'queues': [Queue], 'room': rid|None, 'seen': ts}
ROOMS = {}     # rid -> room dict

# ============================================================
#  Quixx engine (server-authoritative)
# ============================================================
POINTS = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]
COLORS = ['red', 'yellow', 'green', 'blue']
VARIANTS = {'standard', 'colors', 'numbers', 'both'}
MAX_SEATS = 4

# AI opponents. There are no difficulty labels: each bot is just a named player
# with its own brain, and its ELO on the leaderboard is what tells you how hard
# it is to beat.
#
# Every bot carries two fixed handicaps decided at birth, neither of them
# learnable — that is deliberate. Training tunes the weights below, so any
# handicap the weights could compensate for would let every bot converge on the
# same strength and flatten the ratings into noise. Measured the hard way: an
# early build handicapped bots with value noise alone, and 84 rounds of
# self-play turned the field into a coin flip, because good weights are simply
# robust to noise.
#
#   blunder — share of decisions taken at random instead of judged. No weight
#             vector recovers from throwing away a third of your turns.
#   noise   — jitter on each judgement; flavour rather than handicap.
AI_ROSTER_SIZE = 5
AI_NAME_POOL = ['Domino', 'Pixel', 'Rusty', 'Cobalt', 'Juniper', 'Marbles',
                'Ember', 'Waffles', 'Zigzag', 'Pepper', 'Comet', 'Biscuit',
                'Sable', 'Tumbler', 'Nutmeg', 'Whisker', 'Bramble', 'Fig',
                'Otter', 'Quill', 'Pebble', 'Sprocket', 'Clover', 'Mango']

# How often each number turns up on two dice, out of 36. Giving up a 2 costs
# almost nothing — it appears once in 36 rolls — while giving up a 7 is six
# times worse. Pricing skips by cell count instead of by probability was worth
# about 59 ELO on its own, measured over 1200 games against a trained bot.
NUM_FREQ = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
FREQ_MEAN = 36.0 / 11.0        # keeps skip_cost on roughly its old scale

# The learnable brain. Kept deliberately small: five numbers is a whole bot.
#
# Features that only fire on rare decisions were tried and dropped — appetite
# for ending the game while ahead, urgency about a row an opponent could lock,
# and penalty fear that grows with your own pile. All three only apply on a
# lock cell or a penalty turn, and none showed any effect over thousands of
# games. The leverage is in the decision made every single turn: which cell to
# mark, and what the skipped cells were really worth.
#   skip_cost    — charged per unit of probability given up by marking right
#   late_skip    — extra charge as a row fills up
#   lock_bonus   — how much it covets the locking cell at the end of a row
#   penalty_fear — value of dodging the −5 for an empty turn
#   threshold    — how good a mark must look before it bothers marking at all
WEIGHT_KEYS = ['skip_cost', 'late_skip', 'lock_bonus', 'penalty_fear', 'threshold']
DEFAULT_WEIGHTS = {'skip_cost': 1.8, 'late_skip': 0.0, 'lock_bonus': 1.0,
                   'penalty_fear': 5.0, 'threshold': -1.5}
BIRTH_JITTER = {'skip_cost': 0.35, 'late_skip': 0.1, 'lock_bonus': 0.3,
                'penalty_fear': 1.5, 'threshold': 1.2}
WEIGHT_BOUNDS = {'skip_cost': (0.0, 4.0), 'late_skip': (0.0, 2.0),
                 'lock_bonus': (0.0, 3.0), 'penalty_fear': (0.0, 12.0),
                 'threshold': (-12.0, 6.0)}
MUTATION = {'skip_cost': 0.25, 'late_skip': 0.15, 'lock_bonus': 0.25,
            'penalty_fear': 1.0, 'threshold': 0.6}


def build_rows(variant):
    mix_nums = variant in ('numbers', 'both')
    mix_cols = variant in ('colors', 'both')
    asc = list(range(2, 13))
    desc = list(reversed(asc))
    nums = [list(asc), list(asc), list(desc), list(desc)]
    if mix_nums:
        nums = [random.sample(asc, 11) for _ in range(4)]
    if not mix_cols:
        cols = [[c] * 11 for c in COLORS]
    else:
        last_col = random.sample(COLORS, 4)  # each color stays lockable
        pool = [c for c in COLORS for _ in range(10)]
        random.shuffle(pool)
        cols = []
        for r in range(4):
            row = pool[r * 10:(r + 1) * 10] + [last_col[r]]
            cols.append(row)
    return [[{'num': nums[r][i], 'color': cols[r][i]} for i in range(11)]
            for r in range(4)]


class Quixx:
    def __init__(self, variant, seats):
        """seats: 2–4 dicts of id / name / kind ('human'|'ai') / profile."""
        self.variant = variant
        self.rows = build_rows(variant)
        # brain snapshot at kickoff, so training mid-game can't change a game
        self.players = [{'cid': s['id'], 'name': s['name'],
                         'kind': s.get('kind', 'human'), 'profile': s.get('profile'),
                         'weights': dict(s.get('weights') or DEFAULT_WEIGHTS),
                         'blunder': float(s.get('blunder') or 0.0),
                         'noise': float(s.get('noise') or 0.0),
                         'marks': [[False] * 11 for _ in range(4)],
                         'penalties': 0} for s in seats]
        self.active = 0
        self.phase = 'roll'          # roll | white | color | over
        self.dice = None
        self.locked = []
        self.removed = []
        self.white_done = [False] * len(self.players)
        self.active_marked = False
        self.over = False
        self.end_reason = None
        self.forfeit_idx = None
        self.log = []                # recent moves, so AI turns are followable

    # ---- helpers ----
    def note(self, text):
        self.log.append(text)
        del self.log[:-8]
    def pidx(self, cid):
        for i, p in enumerate(self.players):
            if p['cid'] == cid:
                return i
        return None

    def last_marked(self, p, r):
        m = self.players[p]['marks'][r]
        return max((i for i in range(11) if m[i]), default=-1)

    def count(self, p, r):
        return sum(self.players[p]['marks'][r])

    def cell_valid(self, p, r, i, total, color):
        if r in self.locked:
            return False
        cell = self.rows[r][i]
        if cell['num'] != total:
            return False
        if color is not None and cell['color'] != color:
            return False
        if i <= self.last_marked(p, r):
            return False
        if i == 10 and self.count(p, r) < 5:
            return False
        return True

    def options(self, p, mode):
        """Cells p *could* mark for 'white' or 'color', ignoring which phase we
        are actually in. The client uses this to let players look back at the
        white-dice options while the color step is on screen."""
        out = set()
        if not self.dice or self.over:
            return out
        if mode == 'white':
            s = self.dice['w1'] + self.dice['w2']
            for r in range(4):
                for i in range(11):
                    if self.cell_valid(p, r, i, s, None):
                        out.add((r, i))
        elif mode == 'color' and p == self.active:
            for c, v in self.dice['colors'].items():
                for w in (self.dice['w1'], self.dice['w2']):
                    for r in range(4):
                        for i in range(11):
                            if self.cell_valid(p, r, i, w + v, c):
                                out.add((r, i))
        return out

    def valid_cells(self, p):
        """Cells p may actually click right now."""
        if self.phase == 'white' and not self.white_done[p]:
            return self.options(p, 'white')
        if self.phase == 'color' and p == self.active:
            return self.options(p, 'color')
        return set()

    # ---- actions (return error string or None) ----
    def roll(self, cid):
        p = self.pidx(cid)
        if self.over or p != self.active or self.phase != 'roll':
            return "It's not your roll."
        self.dice = {'w1': random.randint(1, 6), 'w2': random.randint(1, 6),
                     'colors': {c: random.randint(1, 6)
                                for c in COLORS if c not in self.removed}}
        self.phase = 'white'
        self.white_done = [False] * len(self.players)
        self.active_marked = False
        self.note('%s rolled — white %d' % (self.players[p]['name'],
                                            self.dice['w1'] + self.dice['w2']))

    def mark(self, cid, r, i):
        p = self.pidx(cid)
        if p is None or self.over:
            return 'Invalid.'
        try:
            r, i = int(r), int(i)
        except (TypeError, ValueError):
            return 'Invalid cell.'
        if (r, i) not in self.valid_cells(p):
            return "That cell can't be marked."
        self.players[p]['marks'][r][i] = True
        if p == self.active:
            self.active_marked = True
        cell = self.rows[r][i]
        if i == 10:
            self.locked.append(r)
            lc = self.rows[r][10]['color']
            if lc not in self.removed:
                self.removed.append(lc)
                if self.dice:
                    self.dice['colors'].pop(lc, None)
        self.note('%s marked %s %d%s' % (self.players[p]['name'], cell['color'],
                                         cell['num'], ' 🔒' if i == 10 else ''))
        if self.check_end():
            return
        if self.phase == 'white':
            self.white_done[p] = True
            if all(self.white_done):
                self.phase = 'color'
        else:
            self.end_turn()

    def pass_white(self, cid):
        p = self.pidx(cid)
        if p is None or self.over or self.phase != 'white' or self.white_done[p]:
            return 'Nothing to pass.'
        self.white_done[p] = True
        self.note('%s passed white' % self.players[p]['name'])
        if all(self.white_done):
            self.phase = 'color'

    def skip_color(self, cid):
        p = self.pidx(cid)
        if self.over or p != self.active or self.phase != 'color':
            return 'Invalid.'
        if not self.active_marked:
            self.players[p]['penalties'] += 1
            self.note('%s took a −5' % self.players[p]['name'])
            if self.check_end():
                return
        self.end_turn()

    def end_turn(self):
        self.active = (self.active + 1) % len(self.players)
        self.phase = 'roll'
        self.dice = None

    # ---- AI seats ----
    def bot_value(self, p, r, i, penalty_risk):
        """What an AI thinks marking (r, i) is worth, in points."""
        pl = self.players[p]
        w = pl['weights']
        c = self.count(p, r)
        v = POINTS[c + 1] - POINTS[c]
        if i == 10:
            v += w['lock_bonus'] * (POINTS[min(c + 2, 12)] - POINTS[c + 1])
        # Price the cells given up by how likely they were to ever come back,
        # not by how many there are. Giving up a 2 barely costs anything;
        # giving up a 7 is six times worse.
        lost = 0.0
        for j in range(self.last_marked(p, r) + 1, i):
            lost += NUM_FREQ[self.rows[r][j]['num']]
        lost /= FREQ_MEAN
        v -= (w['skip_cost'] + w['late_skip'] * c / 5.0) * lost
        if penalty_risk:
            v += w['penalty_fear']
        if pl['noise']:
            v += random.uniform(-pl['noise'], pl['noise'])
        return v

    def bot_pick(self, p, mode, penalty_risk):
        best, cell = None, None
        for (r, i) in self.options(p, mode):
            v = self.bot_value(p, r, i, penalty_risk)
            if best is None or v > best:
                best, cell = v, (r, i)
        return cell, best

    def bot_take(self, p, mode, penalty_risk):
        if self.players[p]['blunder'] and random.random() < self.players[p]['blunder']:
            opts = sorted(self.options(p, mode))
            if opts:                      # careless: grabs one without thinking
                r, i = random.choice(opts)
                self.mark(self.players[p]['cid'], r, i)
                return True
        cell, value = self.bot_pick(p, mode, penalty_risk)
        if cell is not None and value >= self.players[p]['weights']['threshold']:
            self.mark(self.players[p]['cid'], cell[0], cell[1])
        elif mode == 'white':
            self.pass_white(self.players[p]['cid'])
        else:
            self.skip_color(self.players[p]['cid'])
        return True

    def bot_step(self):
        """Play one pending AI action. False once the table waits on a human."""
        if self.over:
            return False
        if self.phase == 'roll':
            p = self.players[self.active]
            if p['kind'] != 'ai':
                return False
            self.roll(p['cid'])
            return True
        if self.phase == 'white':
            for i, p in enumerate(self.players):
                if p['kind'] == 'ai' and not self.white_done[i]:
                    return self.bot_take(i, 'white', False)
            return False
        if self.phase == 'color':
            if self.players[self.active]['kind'] != 'ai':
                return False
            # an empty turn costs 5, so a mark is worth penalty_fear more
            return self.bot_take(self.active, 'color', not self.active_marked)
        return False

    # ---- scoring / end ----
    def row_score(self, p, r):
        c = self.count(p, r) + (1 if self.players[p]['marks'][r][10] else 0)
        return POINTS[min(c, 12)]

    def total(self, p):
        return sum(self.row_score(p, r) for r in range(4)) - 5 * self.players[p]['penalties']

    def check_end(self):
        reason = None
        if len(self.locked) >= 2:
            reason = 'Two rows are locked.'
        for p in self.players:
            if p['penalties'] >= 4:
                reason = p['name'] + ' took a 4th penalty.'
        if not reason:
            return False
        self.over = True
        self.phase = 'over'
        self.end_reason = reason
        return True

    def apply(self, cid, body):
        """Dispatch one player action. Returns an error string or None."""
        t = str(body.get('type') or '')
        if t == 'roll':
            return self.roll(cid)
        if t == 'mark':
            return self.mark(cid, body.get('row'), body.get('idx'))
        if t == 'passWhite':
            return self.pass_white(cid)
        if t == 'skipColor':
            return self.skip_color(cid)
        return 'Unknown action.'

    def forfeit(self, quitter_cid):
        p = self.pidx(quitter_cid)
        self.over = True
        self.phase = 'over'
        self.forfeit_idx = p
        quitn = self.players[p]['name'] if p is not None else '?'
        if len(self.players) == 2 and p is not None:
            self.end_reason = ('%s left — %s wins by forfeit.'
                               % (quitn, self.players[1 - p]['name']))
        else:
            self.end_reason = '%s left — the game ended early.' % quitn

    def result_summary(self):
        """Finished-game outcome, handed to the stats backend."""
        return {
            'game': 'quixx',
            'variant': self.variant,
            'reason': self.end_reason,
            'players': [{'name': p['name'], 'score': self.total(i),
                         'penalties': p['penalties'], 'kind': p['kind'],
                         'profile': p['profile'],
                         'forfeited': i == self.forfeit_idx}
                        for i, p in enumerate(self.players)],
        }

    def to_dict(self, viewer=None):
        # everything on a Quixx table is public, so the viewer makes no odds
        return {
            'kind': 'quixx',
            'variant': self.variant,
            'rows': self.rows,
            'players': [{'cid': p['cid'], 'name': p['name'], 'marks': p['marks'],
                         'kind': p['kind'], 'profile': p['profile'],
                         'penalties': p['penalties'], 'total': self.total(i),
                         'rowScores': [self.row_score(i, r) for r in range(4)]}
                        for i, p in enumerate(self.players)],
            'active': self.active, 'phase': self.phase, 'dice': self.dice,
            'locked': self.locked, 'removed': self.removed,
            'whiteDone': self.white_done, 'activeMarked': self.active_marked,
            'over': self.over, 'reason': self.end_reason, 'log': self.log[-5:],
            'valid': [[list(t) for t in sorted(self.valid_cells(i))]
                      for i in range(len(self.players))],
            # look-only: what each step offers, whatever phase we are in
            'options': {m: [[list(t) for t in sorted(self.options(i, m))]
                            for i in range(len(self.players))]
                        for m in ('white', 'color')},
        }


# ============================================================
#  Mancala engine (Kalah), ported from the mancala project
# ============================================================
M_PITS = 6
M_STONES = 4
M_P1_STORE = 6
M_P2_STORE = 13
M_SIDE = {0: list(range(0, 6)), 1: list(range(7, 13))}
M_STORE = {0: M_P1_STORE, 1: M_P2_STORE}
M_OPPOSITE = {i: 12 - i for i in range(0, 6)}
M_OPPOSITE.update({12 - i: i for i in range(0, 6)})


def m_sow(board, player, pit):
    """Sow one pit. Mutates `board`. Returns (extra_turn, captured)."""
    stones = board[pit]
    board[pit] = 0
    idx = pit
    skip = M_STORE[1 - player]
    for _ in range(stones):
        idx = (idx + 1) % 14
        if idx == skip:
            idx = (idx + 1) % 14
        board[idx] += 1
    extra = (idx == M_STORE[player])
    captured = (idx in M_SIDE[player] and board[idx] == 1
                and board[M_OPPOSITE[idx]] > 0)
    if captured:
        extra = False                      # a capture never also grants a turn
        board[M_STORE[player]] += board[idx] + board[M_OPPOSITE[idx]]
        board[idx] = 0
        board[M_OPPOSITE[idx]] = 0
    return extra, captured


def m_sweep(board):
    for i in M_SIDE[0]:
        board[M_P1_STORE] += board[i]
        board[i] = 0
    for i in M_SIDE[1]:
        board[M_P2_STORE] += board[i]
        board[i] = 0


def m_moves(board, player):
    return [i for i in M_SIDE[player] if board[i] > 0]


# A Mancala brain. Every weight at zero is exactly the original algorithm -
# plain stone difference - so the reference bot is the zero brain and anything
# a learner finds is measured against it.
#   side     — stones sitting on your own side, still yours to work with
#   mobility — how many pits you can still play from
#   extra    — pits loaded to land exactly in your store, i.e. a free turn
#   capture  — empty pits on your side facing a loaded pit opposite
M_KEYS = ['side', 'mobility', 'extra', 'capture']
M_DEFAULT = {k: 0.0 for k in M_KEYS}
M_BOUNDS = {'side': (-1.5, 1.5), 'mobility': (-2.0, 2.0),
            'extra': (-3.0, 3.0), 'capture': (-3.0, 3.0)}
M_MUTATION = {'side': 0.12, 'mobility': 0.2, 'extra': 0.3, 'capture': 0.3}
M_JITTER = {'side': 0.15, 'mobility': 0.25, 'extra': 0.4, 'capture': 0.4}
M_REF_DEPTH = 7           # the original's "hard" setting


def m_eval(board, me, w):
    """Stone difference, plus whatever else this brain has learned to value."""
    opp = 1 - me
    v = float(board[M_STORE[me]] - board[M_STORE[opp]])
    if not w:
        return v
    mine, theirs = M_SIDE[me], M_SIDE[opp]
    if w['side']:
        v += w['side'] * (sum(board[i] for i in mine) - sum(board[i] for i in theirs))
    if w['mobility']:
        v += w['mobility'] * (sum(1 for i in mine if board[i])
                              - sum(1 for i in theirs if board[i]))
    if w['extra']:
        v += w['extra'] * (sum(1 for i in mine if board[i] == M_STORE[me] - i)
                           - sum(1 for i in theirs if board[i] == M_STORE[opp] - i))
    if w['capture']:
        v += w['capture'] * (
            sum(1 for i in mine if board[i] == 0 and board[M_OPPOSITE[i]] > 0)
            - sum(1 for i in theirs if board[i] == 0 and board[M_OPPOSITE[i]] > 0))
    return v


def m_search(board, player, me, depth, alpha, beta, w=None):
    """Alpha-beta, as the original did. Handles Mancala's extra-turn rule:
    sowing into your own store leaves it your move, so the node stays a
    maximising one."""
    moves = m_moves(board, player)
    if depth <= 0 or not moves:
        if not moves:
            b = board[:]
            m_sweep(b)
            return m_eval(b, me, w)
        return m_eval(board, me, w)
    maximizing = (player == me)
    best = -10 ** 6 if maximizing else 10 ** 6
    for pit in moves:
        b = board[:]
        extra, _ = m_sow(b, player, pit)
        nxt = player if extra else 1 - player
        val = m_search(b, nxt, me, depth - 1, alpha, beta, w)
        if maximizing:
            best = max(best, val)
            alpha = max(alpha, best)
        else:
            best = min(best, val)
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best


class Mancala:
    """Two-player Kalah. Six pits a side, four stones each, sown
    counter-clockwise past your own store but never the opponent's."""

    def __init__(self, variant, seats):
        self.variant = variant
        self.players = [{'cid': s['id'], 'name': s['name'],
                         'kind': s.get('kind', 'human'),
                         'profile': s.get('profile'),
                         # the reference bot plays the original algorithm: plain
                         # stone difference, deepest search, never careless
                         'reference': bool(s.get('reference')),
                         'weights': (None if s.get('reference')
                                     else dict(s.get('weights') or M_DEFAULT)),
                         'blunder': (0.0 if s.get('reference')
                                     else float(s.get('blunder') or 0.0))}
                        for s in seats]
        self.board = [M_STONES] * 14
        self.board[M_P1_STORE] = 0
        self.board[M_P2_STORE] = 0
        self.active = 0
        self.over = False
        self.end_reason = None
        self.forfeit_idx = None
        self.last = None
        self.log = []

    # ---- helpers ----
    def pidx(self, cid):
        for i, p in enumerate(self.players):
            if p['cid'] == cid:
                return i
        return None

    def note(self, text):
        self.log.append(text)
        del self.log[:-8]

    def valid_moves(self, p=None):
        if self.over:
            return []
        return m_moves(self.board, self.active if p is None else p)

    def total(self, p):
        return self.board[M_STORE[p]]

    # ---- actions ----
    def apply(self, cid, body):
        if str(body.get('type') or '') != 'sow':
            return 'Unknown action.'
        return self.sow(cid, body.get('pit'))

    def sow(self, cid, pit):
        p = self.pidx(cid)
        if self.over:
            return 'The game is over.'
        if p is None or p != self.active:
            return "It's not your turn."
        try:
            pit = int(pit)
        except (TypeError, ValueError):
            return 'Invalid pit.'
        if pit not in M_SIDE[p]:
            return 'That is not your pit.'
        if self.board[pit] <= 0:
            return 'That pit is empty.'

        extra, captured = m_sow(self.board, p, pit)
        name = self.players[p]['name']
        self.last = {'player': p, 'pit': pit, 'extra': extra, 'captured': captured}
        self.note('%s sowed pit %d%s' % (name, M_SIDE[p].index(pit) + 1,
                                         ' — capture!' if captured else
                                         (' — goes again' if extra else '')))
        # either side emptying ends it; so does the mover having nothing left
        if not m_moves(self.board, 0) or not m_moves(self.board, 1):
            self.finish()
        elif not extra:
            self.active = 1 - p
            if not self.valid_moves():
                self.finish()
        elif not self.valid_moves():
            self.finish()

    def finish(self):
        m_sweep(self.board)
        self.over = True
        a, b = self.board[M_P1_STORE], self.board[M_P2_STORE]
        if a == b:
            self.end_reason = 'A tie — %d stones each.' % a
        else:
            win = self.players[0 if a > b else 1]['name']
            self.end_reason = '%s wins %d–%d.' % (win, max(a, b), min(a, b))

    def forfeit(self, quitter_cid):
        p = self.pidx(quitter_cid)
        self.over = True
        self.forfeit_idx = p
        quitn = self.players[p]['name'] if p is not None else '?'
        if p is not None:
            self.end_reason = ('%s left — %s wins by forfeit.'
                               % (quitn, self.players[1 - p]['name']))
        else:
            self.end_reason = '%s left — the game ended early.' % quitn

    # ---- AI ----
    def depth_for(self, p):
        """Search depth from the bot's carelessness, so a bot plays at the
        same relative strength here as it does anywhere else."""
        if self.players[p]['reference']:
            return M_REF_DEPTH
        return max(1, min(7, int(round(6.5 - 15.0 * self.players[p]['blunder']))))

    def bot_step(self):
        if self.over:
            return False
        p = self.active
        if self.players[p]['kind'] != 'ai':
            return False
        moves = self.valid_moves()
        if not moves:
            self.finish()
            return False
        if self.players[p]['blunder'] and random.random() < self.players[p]['blunder']:
            pick = random.choice(moves)
        else:
            depth = self.depth_for(p)
            w = self.players[p]['weights']
            best, pick = None, moves[0]
            for pit in moves:
                b = self.board[:]
                extra, _ = m_sow(b, p, pit)
                nxt = p if extra else 1 - p
                val = m_search(b, nxt, p, depth - 1, -10 ** 6, 10 ** 6, w)
                if best is None or val > best:
                    best, pick = val, pit
        self.sow(self.players[p]['cid'], pick)
        return True

    # ---- payloads ----
    def result_summary(self):
        return {
            'game': 'mancala', 'variant': self.variant, 'reason': self.end_reason,
            'players': [{'name': p['name'], 'score': self.total(i), 'penalties': 0,
                         'kind': p['kind'], 'profile': p['profile'],
                         'forfeited': i == self.forfeit_idx}
                        for i, p in enumerate(self.players)],
        }

    def to_dict(self, viewer=None):
        # both sides of a Mancala board are open, so the viewer makes no odds
        return {
            'kind': 'mancala', 'variant': self.variant, 'board': self.board,
            'players': [{'cid': p['cid'], 'name': p['name'], 'kind': p['kind'],
                         'profile': p['profile'], 'total': self.total(i),
                         'pits': M_SIDE[i], 'store': M_STORE[i]}
                        for i, p in enumerate(self.players)],
            'active': self.active, 'over': self.over, 'reason': self.end_reason,
            'last': self.last, 'log': self.log[-5:],
            'valid': [self.valid_moves(0) if self.active == 0 and not self.over else [],
                      self.valid_moves(1) if self.active == 1 and not self.over else []],
        }


# ============================================================
#  Euchre engine (4 seats, fixed partnerships)
# ============================================================
# A 24-card deck — 9 up to ace — dealt five each. Seats 0 and 2 are one team
# and seats 1 and 3 the other, so partners always sit across the table. One
# side names trump and then has to take three of the five tricks; first team
# to ten points wins the match.
E_SUITS = ['S', 'H', 'D', 'C']
E_RANKS = ['9', 'T', 'J', 'Q', 'K', 'A']
E_ORDER = {r: i for i, r in enumerate(E_RANKS)}
E_SUIT_NAMES = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs'}
E_PIPS = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
E_RANK_NAMES = {'9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
E_MATE = {'S': 'C', 'C': 'S', 'H': 'D', 'D': 'H'}      # the same-colour suit
E_TARGET = 10


def e_deck():
    return [r + s for s in E_SUITS for r in E_RANKS]


def e_card_name(c):
    return E_RANK_NAMES[c[0]] + E_PIPS[c[1]]


def e_suit(card, trump):
    """The suit a card really belongs to. The left bower — the jack matching
    trump in colour — becomes a trump and stops being a member of its printed
    suit, which is the one rule that catches out every new Euchre player."""
    s = card[1]
    if trump and card[0] == 'J' and E_MATE[s] == trump:
        return trump
    return s


def e_power(card, trump, led):
    """Trick-taking power. Trump beats everything, the right bower beats the
    left, and anything off the led suit cannot win at all."""
    s = e_suit(card, trump)
    if trump and s == trump:
        if card[0] == 'J':
            return 220 if card[1] == trump else 210
        return 200 + E_ORDER[card[0]]
    if s == led:
        return 100 + E_ORDER[card[0]]
    return 0


def e_sort_key(card, trump):
    """Display order: trump first, then the other suits, high to low."""
    s = e_suit(card, trump)
    grp = 0 if (trump and s == trump) else 1 + E_SUITS.index(s)
    return (grp, -e_power(card, trump, s))


def e_follow(hand, trump, led):
    """The cards you are allowed to play. You must follow the led suit while
    you hold it — counting the left bower as trump, not as its printed suit."""
    same = [c for c in hand if e_suit(c, trump) == led]
    return same or list(hand)


# What a Euchre bot can learn. Zero is the plain textbook heuristic, so a brain
# that has never trained plays exactly what the reference bot plays.
#   order   — how readily it names trump at all
#   alone   — how readily it drops its partner and plays the hand solo
#   offace  — what it thinks an off-suit ace is worth when bidding
#   void    — what it thinks a short suit is worth
#   lead    — how much it likes leading trump to draw the opposition's
#   partner — how far it trusts a partner who is already winning the trick
E_KEYS = ['order', 'alone', 'offace', 'void', 'lead', 'partner']
E_DEFAULT = {k: 0.0 for k in E_KEYS}
E_BOUNDS = {'order': (-1.5, 1.5), 'alone': (-1.5, 1.5), 'offace': (-1.0, 1.0),
            'void': (-1.0, 1.0), 'lead': (-1.5, 1.5), 'partner': (-1.5, 1.5)}
E_MUTATION = {'order': 0.18, 'alone': 0.18, 'offace': 0.12,
              'void': 0.12, 'lead': 0.2, 'partner': 0.2}
E_JITTER = {'order': 0.25, 'alone': 0.25, 'offace': 0.18,
            'void': 0.18, 'lead': 0.3, 'partner': 0.3}

E_TRUMP_PTS = {'A': 2.0, 'K': 1.5, 'Q': 1.0, 'T': 0.6, '9': 0.5}
# Hand-strength cutoffs, read off the actual distribution over 60k deals rather
# than guessed. Four seats each get a say, so an individual rate compounds:
# ordering on the top 23% of hands is what puts a caller in about 62% of deals,
# roughly how often a real table takes it up. Round two scores the best of the
# three suits left, so the same appetite needs a higher number to mean the same
# thing. Going alone sits up at the 99th percentile, where it belongs.
# These land a caller in ~93% of deals (7% pass out), someone goes alone in
# ~4.5%, and the makers get euchred ~12% of the time.
E_CALL = 5.2        # order up the turned suit
E_CALL2 = 5.9       # name a suit after the upcard is turned down — lower than
                    # E_CALL because it is the last chance before a redeal
E_ALONE = 8.7       # ...and drop your partner


def e_strength(hand, trump, w):
    """Roughly how many of the five tricks a hand should take with `trump`
    named. The bowers do most of the work, an off-suit ace is worth about a
    trick, and a void counts for something because it lets you trump in."""
    pts, held = 0.0, set()
    for c in hand:
        s = e_suit(c, trump)
        held.add(s)
        if s == trump:
            if c[0] == 'J':
                pts += 3.0 if c[1] == trump else 2.5
            else:
                pts += E_TRUMP_PTS[c[0]]
        elif c[0] == 'A':
            pts += 1.0 + 0.3 * w['offace']
        elif c[0] == 'K':
            pts += 0.35
    voids = 3 - len([s for s in held if s != trump])
    return pts + 0.45 * voids * (1.0 + 0.4 * w['void'])


class Euchre:
    """Four-handed Euchre. Partners sit across, the bowers outrank the aces,
    and the side that names trump must take three tricks or be euchred."""

    def __init__(self, variant, seats):
        self.variant = variant if variant in ('standard', 'stick') else 'standard'
        self.players = [{'cid': s['id'], 'name': s['name'],
                         'kind': s.get('kind', 'human'),
                         'profile': s.get('profile'),
                         # the reference bot plays the plain textbook heuristic
                         'reference': bool(s.get('reference')),
                         'weights': (dict(E_DEFAULT) if s.get('reference')
                                     else dict(s.get('weights') or E_DEFAULT)),
                         'blunder': (0.0 if s.get('reference')
                                     else float(s.get('blunder') or 0.0))}
                        for s in seats]
        for p in self.players:                  # a brain from before Euchre existed
            for k in E_KEYS:
                p['weights'].setdefault(k, 0.0)
        self.all_ai = all(p['kind'] == 'ai' for p in self.players)
        self.scores = [0, 0]
        self.dealer = random.randrange(4)
        self.hand_no = 0
        self.over = False
        self.end_reason = None
        self.forfeit_idx = None
        self.log = []
        self.pending = None            # 'trick' | 'hand' — a beat for the table
        self.hand_summary = None
        self.last_trick = None
        self.start_hand()

    # ---- table helpers ----
    def team(self, seat):
        return seat % 2

    def sitting_out(self):
        """The partner of whoever went alone, who sits the hand out."""
        return None if self.alone is None else (self.alone + 2) % 4

    def in_play(self, seat):
        return seat != self.sitting_out()

    def next_seat(self, seat):
        s = (seat + 1) % 4
        return s if self.in_play(s) else (s + 1) % 4

    def pidx(self, cid):
        for i, p in enumerate(self.players):
            if p['cid'] == cid:
                return i
        return None

    def name(self, seat):
        return self.players[seat]['name']

    def note(self, text):
        self.log.append(text)
        del self.log[:-8]

    def total(self, i):
        """A seat's score is its side's score — Euchre is won as a pair."""
        return self.scores[self.team(i)]

    # ---- dealing & bidding ----
    def start_hand(self):
        d = e_deck()
        random.shuffle(d)
        self.hands = [d[i * 5:i * 5 + 5] for i in range(4)]
        self.upcard = d[20]
        self.hand_no += 1
        self.trump = None
        self.maker = None
        self.alone = None
        self.turned = None
        self.phase = 'bid1'
        self.turn = (self.dealer + 1) % 4
        self.trick = []
        self.trick_no = 0
        self.tricks = [0, 0]
        self.leader = (self.dealer + 1) % 4
        self.last_trick = None
        self.hand_summary = None
        self.note('%s deals — %s turned up.'
                  % (self.name(self.dealer), e_card_name(self.upcard)))

    def order_up(self, seat, alone):
        if self.phase != 'bid1':
            return 'Nobody has turned a card up.'
        self.trump = self.upcard[1]
        self.maker = seat
        self.alone = seat if alone else None
        self.hands[self.dealer].append(self.upcard)
        self.note('%s %s — %s is trump%s.'
                  % (self.name(seat),
                     'picks it up' if seat == self.dealer else 'orders it up',
                     E_SUIT_NAMES[self.trump], ', alone' if alone else ''))
        self.phase = 'discard'
        self.turn = self.dealer
        return None

    def discard(self, seat, card):
        if self.phase != 'discard' or seat != self.dealer:
            return 'There is nothing to discard.'
        if card not in self.hands[seat]:
            return 'You do not hold that card.'
        self.hands[seat].remove(card)
        self.note('%s discards.' % self.name(seat))
        self.begin_play()
        return None

    def pass_bid(self, seat):
        if self.phase not in ('bid1', 'bid2'):
            return 'There is no bid to pass on.'
        if self.phase == 'bid2' and seat == self.dealer and self.variant == 'stick':
            return 'Stick the dealer — you have to name a suit.'
        self.note('%s passes.' % self.name(seat))
        if seat != self.dealer:
            self.turn = (seat + 1) % 4
        elif self.phase == 'bid1':
            self.phase = 'bid2'
            self.turned = self.upcard[1]
            self.turn = (self.dealer + 1) % 4
            self.note('%s is turned down.' % E_SUIT_NAMES[self.turned])
        else:
            self.pass_out()
        return None

    def pass_out(self):
        """Everybody passed twice — no trump, nobody scores, deal again."""
        self.hand_summary = {'passed': True,
                             'text': 'Passed out — nobody named trump.'}
        self.note('Passed out — redeal.')
        self.pending = 'hand'

    def call_trump(self, seat, suit, alone):
        if self.phase != 'bid2':
            return 'You cannot name a suit yet.'
        if suit not in E_SUITS:
            return 'Pick a suit.'
        if suit == self.turned:
            return '%s was turned down — pick another suit.' % E_SUIT_NAMES[suit]
        self.trump = suit
        self.maker = seat
        self.alone = seat if alone else None
        self.note('%s names %s%s.' % (self.name(seat), E_SUIT_NAMES[suit],
                                      ', alone' if alone else ''))
        self.begin_play()
        return None

    # ---- trick play ----
    def begin_play(self):
        self.phase = 'play'
        self.leader = (self.dealer + 1) % 4
        if not self.in_play(self.leader):
            self.leader = (self.leader + 1) % 4
        self.turn = self.leader
        self.trick = []
        self.trick_no = 0

    def play(self, seat, card):
        if self.phase != 'play':
            return 'No trick is in progress.'
        hand = self.hands[seat]
        if card not in hand:
            return 'You do not hold that card.'
        if self.trick:
            led = e_suit(self.trick[0]['card'], self.trump)
            if card not in e_follow(hand, self.trump, led):
                return 'You have to follow %s.' % E_SUIT_NAMES[led]
        hand.remove(card)
        self.trick.append({'seat': seat, 'card': card})
        if len(self.trick) >= (3 if self.alone is not None else 4):
            self.resolve_trick()
        else:
            self.turn = self.next_seat(seat)
        return None

    def resolve_trick(self):
        led = e_suit(self.trick[0]['card'], self.trump)
        best = max(self.trick, key=lambda x: e_power(x['card'], self.trump, led))
        w = best['seat']
        self.tricks[self.team(w)] += 1
        self.trick_no += 1
        self.leader = w
        self.last_trick = {'cards': list(self.trick), 'winner': w,
                           'no': self.trick_no}
        self.note('%s takes trick %d with %s.'
                  % (self.name(w), self.trick_no, e_card_name(best['card'])))
        self.pending = 'trick'          # hold, so everyone sees who took it

    def score_hand(self):
        mt = self.team(self.maker)
        won = self.tricks[mt]
        alone = self.alone is not None
        who = self.name(self.maker)
        if won >= 3:
            pts = 4 if (alone and won == 5) else (2 if won == 5 else 1)
            self.scores[mt] += pts
            if alone and won == 5:
                text = '%s went alone and swept all five — 4 points.' % who
            elif won == 5:
                text = '%s marched — all five tricks, 2 points.' % who
            else:
                text = '%s made it with %d tricks — 1 point.' % (who, won)
            euchred = False
        else:
            pts, euchred = 2, True
            self.scores[1 - mt] += pts
            text = '%s was euchred — 2 points the other way.' % who
        self.hand_summary = {'passed': False, 'text': text, 'maker': self.maker,
                             'makerTeam': mt, 'tricks': list(self.tricks),
                             'points': pts, 'euchred': euchred, 'alone': alone}
        self.note(text)
        self.pending = 'hand'

    def resume(self):
        """Clear the pause that lets the table see a finished trick or hand."""
        if not self.pending:
            return None
        if self.pending == 'trick':
            self.pending = None
            self.trick = []
            if self.trick_no >= 5:
                self.score_hand()
            else:
                self.turn = self.leader
            return None
        self.pending = None                       # 'hand'
        self.last_trick = None
        if max(self.scores) >= E_TARGET:
            self.finish()
        else:
            self.dealer = (self.dealer + 1) % 4
            self.start_hand()
        return None

    def finish(self):
        self.over = True
        self.phase = 'over'
        self.turn = None
        a, b = self.scores
        win = 0 if a > b else 1
        self.end_reason = ('%s & %s win %d–%d.'
                           % (self.name(win), self.name(win + 2),
                              max(a, b), min(a, b)))

    def forfeit(self, quitter_cid):
        p = self.pidx(quitter_cid)
        self.over = True
        self.phase = 'over'
        self.turn = None
        self.pending = None
        self.forfeit_idx = p
        if p is None:
            self.end_reason = 'A player left — the game ended early.'
        else:
            other = 1 - self.team(p)
            self.end_reason = ('%s left — %s & %s take it by forfeit.'
                               % (self.name(p), self.name(other),
                                  self.name(other + 2)))

    # ---- actions ----
    def beat_id(self):
        """Identifies the exact pause the table is sitting on, so a nudge meant
        for the last trick cannot skip past the hand summary behind it."""
        return '%s:%d:%d' % (self.pending or '-', self.hand_no, self.trick_no)

    def apply(self, cid, body):
        t = str(body.get('type') or '')
        if t == 'continue':               # anyone at the table can clear a beat
            want = str(body.get('at') or '')
            if want and want != self.beat_id():
                return None               # a stale nudge from another client
            return self.resume()
        if self.over:
            return 'The game is over.'
        if self.pending:
            return 'Wait for the table to settle.'
        seat = self.pidx(cid)
        if seat is None:
            return 'You are not seated at this table.'
        if seat != self.turn:
            return "It's not your turn."
        if t == 'order':
            return self.order_up(seat, bool(body.get('alone')))
        if t == 'pass':
            return self.pass_bid(seat)
        if t == 'call':
            return self.call_trump(seat, str(body.get('suit') or ''),
                                   bool(body.get('alone')))
        if t == 'discard':
            return self.discard(seat, str(body.get('card') or ''))
        if t == 'play':
            return self.play(seat, str(body.get('card') or ''))
        return 'Unknown action.'

    # ---- AI ----
    def worst_card(self, hand, trump, w):
        """The least useful card to let go of: cheap, not trump, and ideally
        the last of its suit so the hand can trump that suit later."""
        counts = {}
        for c in hand:
            s = e_suit(c, trump)
            counts[s] = counts.get(s, 0) + 1

        def cost(c):
            s = e_suit(c, trump)
            v = e_power(c, trump, s)
            if trump and s == trump:
                v += 500                       # never pitch trump if there is a choice
            if c[0] == 'A':
                v += 40
            if counts[s] == 1 and s != trump:
                v -= 25 * (1.0 + 0.5 * w['void'])
            return v
        return min(hand, key=cost)

    def bot_bid1(self, seat):
        p = self.players[seat]
        w = p['weights']
        suit = self.upcard[1]
        hand = list(self.hands[seat])
        if seat == self.dealer:
            # the dealer would take the upcard, so judge the hand it would keep
            hand.append(self.upcard)
            hand.remove(self.worst_card(hand, suit, w))
        s = e_strength(hand, suit, w)
        # Where the upcard ends up. The dealer needs no bonus here — it is
        # already holding the card in the hand valued above; double-counting it
        # was what had the dealer ordering up almost every deal.
        if seat != self.dealer:
            s += 0.25 if self.team(seat) == self.team(self.dealer) else -0.3
        if p['blunder'] and random.random() < p['blunder']:
            s += random.uniform(-1.2, 1.2)
        if s >= E_CALL - 0.35 * w['order']:
            self.order_up(seat, s >= E_ALONE - 0.4 * w['alone'])
        else:
            self.pass_bid(seat)

    def bot_bid2(self, seat):
        p = self.players[seat]
        w = p['weights']
        best, pick = None, None
        for suit in E_SUITS:
            if suit == self.turned:
                continue
            s = e_strength(self.hands[seat], suit, w)
            if best is None or s > best:
                best, pick = s, suit
        if p['blunder'] and random.random() < p['blunder']:
            best += random.uniform(-1.2, 1.2)
        must = seat == self.dealer and self.variant == 'stick'
        if must or best >= E_CALL2 - 0.35 * w['order']:
            self.call_trump(seat, pick, best >= E_ALONE - 0.4 * w['alone'])
        else:
            self.pass_bid(seat)

    def pick_lead(self, seat, w):
        """Leading. The makers draw trump so their bowers clear the way;
        everyone else prefers an off-suit ace and otherwise leads cheap."""
        hand = self.hands[seat]
        trumps = [c for c in hand if e_suit(c, self.trump) == self.trump]
        aces = [c for c in hand
                if c[0] == 'A' and e_suit(c, self.trump) != self.trump]
        ours = self.maker is not None and self.team(self.maker) == self.team(seat)
        want_trump = (ours and len(trumps) >= 2) or w['lead'] > 0.6
        if self.alone == seat and trumps:
            want_trump = True
        if want_trump and trumps:
            return max(trumps, key=lambda c: e_power(c, self.trump, self.trump))
        if aces:
            return aces[0]
        plain = [c for c in hand if e_suit(c, self.trump) != self.trump]
        if not plain:
            return min(trumps, key=lambda c: e_power(c, self.trump, self.trump))
        return self.worst_card(plain, self.trump, w)

    def pick_card(self, seat):
        p = self.players[seat]
        w = p['weights']
        hand = self.hands[seat]
        if not self.trick:
            if p['blunder'] and random.random() < p['blunder']:
                return random.choice(hand)
            return self.pick_lead(seat, w)
        led = e_suit(self.trick[0]['card'], self.trump)
        legal = e_follow(hand, self.trump, led)
        if len(legal) == 1:
            return legal[0]
        if p['blunder'] and random.random() < p['blunder']:
            return random.choice(legal)
        best = max(self.trick, key=lambda x: e_power(x['card'], self.trump, led))
        top = e_power(best['card'], self.trump, led)
        partner_ahead = self.team(best['seat']) == self.team(seat)
        last = len(self.trick) == (2 if self.alone is not None else 3)
        # partner already has it — save the good cards for a trick that needs them
        if partner_ahead and (last or top >= 205 or w['partner'] > 0.4):
            return self.worst_card(legal, self.trump, w)
        winners = [c for c in legal if e_power(c, self.trump, led) > top]
        if winners:
            return min(winners, key=lambda c: e_power(c, self.trump, led))
        return self.worst_card(legal, self.trump, w)

    def bot_step(self):
        if self.over:
            return False
        if self.pending:
            # with nobody watching there is no reason to hold the beat
            if self.all_ai:
                self.resume()
                return True
            return False
        seat = self.turn
        if seat is None or self.players[seat]['kind'] != 'ai':
            return False
        if self.phase == 'bid1':
            self.bot_bid1(seat)
        elif self.phase == 'bid2':
            self.bot_bid2(seat)
        elif self.phase == 'discard':
            self.discard(seat, self.worst_card(self.hands[seat], self.trump,
                                               self.players[seat]['weights']))
        elif self.phase == 'play':
            self.play(seat, self.pick_card(seat))
        else:
            return False
        return True

    # ---- payloads ----
    def result_summary(self):
        return {
            'game': 'euchre', 'variant': self.variant, 'reason': self.end_reason,
            'players': [{'name': p['name'], 'score': self.total(i), 'penalties': 0,
                         'kind': p['kind'], 'profile': p['profile'],
                         'team': self.team(i), 'forfeited': i == self.forfeit_idx}
                        for i, p in enumerate(self.players)],
        }

    def to_dict(self, viewer=None):
        """Table state for one viewer. Hands are secret: you get your own cards
        and nothing but a count for everyone else, so the same broadcast cannot
        leak a hand to an opponent or to the people watching."""
        me = self.pidx(viewer) if viewer else None
        sort_by = self.trump or (self.upcard[1] if self.phase == 'bid1' else None)
        legal = []
        if (me is not None and self.phase == 'play' and me == self.turn
                and not self.pending and not self.over):
            legal = (e_follow(self.hands[me], self.trump,
                              e_suit(self.trick[0]['card'], self.trump))
                     if self.trick else list(self.hands[me]))
        return {
            'kind': 'euchre', 'variant': self.variant,
            'players': [{'cid': p['cid'], 'name': p['name'], 'kind': p['kind'],
                         'profile': p['profile'], 'team': self.team(i),
                         'cards': len(self.hands[i]), 'out': i == self.sitting_out()}
                        for i, p in enumerate(self.players)],
            'me': me, 'legal': legal,
            'hand': (sorted(self.hands[me], key=lambda c: e_sort_key(c, sort_by))
                     if me is not None else []),
            'scores': self.scores, 'target': E_TARGET, 'tricks': self.tricks,
            'dealer': self.dealer, 'turn': self.turn, 'phase': self.phase,
            'trump': self.trump, 'turned': self.turned, 'upcard': self.upcard,
            'maker': self.maker, 'alone': self.alone, 'out': self.sitting_out(),
            'trick': self.trick, 'trickNo': self.trick_no,
            'lastTrick': self.last_trick, 'pending': self.pending,
            'beat': self.beat_id(),
            'handNo': self.hand_no, 'handSummary': self.hand_summary,
            'over': self.over, 'reason': self.end_reason, 'log': self.log[-5:],
        }


# ============================================================
#  Chess engine (2 seats, seat 0 plays White)
# ============================================================
# python-chess is the first dependency this project has ever taken. It is pure
# Python — a py3-none-any wheel, so `pip install chess` needs no compiler on
# the Pi — and it brings correct legal move generation, castling rights, en
# passant, threefold repetition and insufficient-material draws. Those are
# exactly the rules homegrown engines get quietly wrong, and none of them are
# the interesting part of a learning opponent.
#
# The import is guarded so a machine that has not installed it still runs
# Quixx, Mancala and Euchre exactly as before: chess just does not appear in
# the lobby, and start_server() says why on the way up.
try:
    import chess as chesslib
    from chess_ai import ChessAI, static_eval as chess_static_eval
    CHESS_OK = True
    CHESS_WHY = None
except ImportError as _chess_err:            # pragma: no cover - environment
    chesslib = None
    ChessAI = None
    chess_static_eval = None
    CHESS_OK = False
    CHESS_WHY = str(_chess_err)

# One brain file per bot, so each keeps its own opening book and its own
# evaluation weights. A directory rather than one shared file because the
# brains are per-bot and independent; back up the folder and you have them all.
CHESS_BRAIN_DIR = os.path.join(BASE, 'chess_brains')


def chess_brain_path(pid):
    safe = ''.join(c for c in str(pid or 'default') if c.isalnum() or c in '-_')
    return os.path.join(CHESS_BRAIN_DIR, (safe or 'default') + '.json')


class Chess:
    """Standard chess, server-authoritative.

    Two seats: seat 0 plays White, seat 1 plays Black. room_seats() puts humans
    before bots, so a solo player is always White against the AI — which is the
    arrangement the lobby offers. Two humans work equally well.

    The learning lives in ChessAI (chess_ai.py); this class only decides when
    to ask it for a move and when to hand it a finished game. Keeping those two
    phases apart is what stops a half-played or abandoned game from training
    anything.
    """

    # Seconds the AI may think per move. The whole request runs under the
    # global LOCK — as Mancala's alpha-beta already does — so this is also how
    # long one move blocks other clients. Modest on purpose; the AI's search is
    # time-budgeted, so it simply reaches a shallower depth on slower hardware
    # rather than overrunning.
    THINK_TIME = 0.6

    def __init__(self, variant, seats):
        self.variant = variant
        self.players = [{'cid': s['id'], 'name': s['name'],
                         'kind': s.get('kind', 'human'),
                         'profile': s.get('profile'),
                         'blunder': (0.0 if s.get('reference')
                                     else float(s.get('blunder') or 0.0)),
                         'style': s.get('style') or 'searcher'}
                        for s in seats[:2]]
        self.board = chesslib.Board()
        self.over = False
        self.end_reason = None
        self.forfeit_idx = None
        self.scores = [0.5 for _ in self.players]
        self.last = None
        self.log = []
        self.sans = []
        self.taught = False
        self.reports = {}

        try:
            os.makedirs(CHESS_BRAIN_DIR, exist_ok=True)
        except OSError:
            pass

        # One ChessAI per bot seat, loaded from that bot's own brain file.
        self.ai = {}
        for i, p in enumerate(self.players):
            if p['kind'] != 'ai':
                continue
            other = self.players[1 - i]['name'] if len(self.players) > 1 else 'human'
            self.ai[i] = ChessAI(name=p['name'],
                                 data_path=chess_brain_path(p['profile']),
                                 opponent=other, think_time=self.THINK_TIME,
                                 max_depth=self.depth_ceiling(p['blunder']),
                                 style=p['style'])

    @staticmethod
    def depth_ceiling(blunder):
        """A bot's personal strength ceiling, from the handicap it was born
        with — the same idea as Mancala's depth_for(). The ChessAI's own
        skill_level() ramps up to this as it plays, so a new bot starts shallow
        and grows into whatever its handicap allows rather than arriving at
        full strength on game one."""
        return max(2, min(4, int(round(4.5 - 6.0 * blunder))))

    # ---- helpers ----
    @property
    def active(self):
        """Derived from the board rather than tracked alongside it, so the two
        cannot drift apart."""
        return 0 if self.board.turn == chesslib.WHITE else 1

    def pidx(self, cid):
        for i, p in enumerate(self.players):
            if p['cid'] == cid:
                return i
        return None

    def note(self, text):
        self.log.append(text)
        del self.log[:-8]

    # ---- actions ----
    def apply(self, cid, body):
        kind = str(body.get('type') or '')
        if kind == 'move':
            return self.move(cid, body.get('uci') or body.get('move'))
        if kind == 'resign':
            return self.resign(cid)
        return 'Unknown action.'

    def move(self, cid, uci):
        p = self.pidx(cid)
        if self.over:
            return 'The game is over.'
        if p is None or p != self.active:
            return "It's not your turn."
        if not uci:
            return 'No move given.'
        text = str(uci).strip()

        try:
            mv = chesslib.Move.from_uci(text)
        except ValueError:
            try:
                mv = self.board.parse_san(text)     # a UI may find SAN easier
            except ValueError:
                return 'Not a move: %s' % text[:20]

        if mv not in self.board.legal_moves:
            # A promotion sent without naming the piece is the common case, and
            # a queen is what the player almost always meant.
            promo = chesslib.Move(mv.from_square, mv.to_square,
                                  promotion=chesslib.QUEEN)
            if promo in self.board.legal_moves:
                mv = promo
            else:
                return 'Illegal move.'
        self.push(mv, p)

    def resign(self, cid):
        p = self.pidx(cid)
        if self.over:
            return 'The game is over.'
        if p is None:
            return 'You are not playing.'
        self.over = True
        self.scores = [0.0 if i == p else 1.0 for i in range(len(self.players))]
        self.end_reason = ('%s resigned — %s wins.'
                           % (self.players[p]['name'], self.players[1 - p]['name']))
        self.note(self.end_reason)
        self.teach()

    def push(self, mv, p):
        san = self.board.san(mv)                    # must be read before the push
        self.board.push(mv)
        self.sans.append(san)
        checking = self.board.is_check()
        self.last = {'player': p, 'uci': mv.uci(), 'san': san,
                     'from': chesslib.square_name(mv.from_square),
                     'to': chesslib.square_name(mv.to_square),
                     'check': checking}
        self.note('%s played %s' % (self.players[p]['name'], san))
        if self.board.is_game_over(claim_draw=True):
            self.finish()

    def finish(self):
        self.over = True
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            self.end_reason = 'The game ended.'
        elif outcome.winner is None:
            self.end_reason = ('Draw — %s.'
                               % outcome.termination.name.replace('_', ' ').lower())
        else:
            w = 0 if outcome.winner == chesslib.WHITE else 1
            self.scores = [1.0 if i == w else 0.0 for i in range(len(self.players))]
            self.end_reason = ('%s wins by %s.'
                               % (self.players[w]['name'],
                                  outcome.termination.name.replace('_', ' ').lower()))
        self.teach()

    def forfeit(self, quitter_cid):
        p = self.pidx(quitter_cid)
        self.over = True
        self.forfeit_idx = p
        who = self.players[p]['name'] if p is not None else '?'
        if p is not None:
            self.scores = [0.0 if i == p else 1.0 for i in range(len(self.players))]
            self.end_reason = ('%s left — %s wins by forfeit.'
                               % (who, self.players[1 - p]['name']))
        else:
            self.end_reason = '%s left — the game ended early.' % who
        # A walkout says nothing about the moves that preceded it, so the game
        # counts in the record but teaches the AI nothing.
        self.teach(learn=False)

    def teach(self, learn=True):
        """Hand the finished game to each AI seat, then persist what changed.

        Runs exactly once — finish(), resign() and forfeit() can all reach it,
        and a game must not be learned from twice.
        """
        if self.taught:
            return
        self.taught = True
        for i, ai in self.ai.items():
            try:
                if not learn:
                    ai.new_game()          # drop the trace: nothing to credit
                self.reports[i] = ai.learn_from_game(
                    self.scores[i], final_board=self.board,
                    opponent=self.players[1 - i]['name'])
                ai.save_progress()
            except Exception as e:         # never let learning break a game
                print('  [chess] %s could not learn: %s'
                      % (self.players[i]['name'], e))

    # ---- AI ----
    def bot_step(self):
        if self.over:
            return False
        p = self.active
        if self.players[p]['kind'] != 'ai':
            return False
        legal = list(self.board.legal_moves)
        if not legal:
            self.finish()
            return False

        mv = None
        blunder = self.players[p]['blunder']
        if blunder and random.random() < blunder:
            # The handicap it was born with. Note this deliberately bypasses
            # the ChessAI, so a move it did not choose never lands in its book.
            mv = random.choice(legal)
        elif self.ai.get(p):
            uci = self.ai[p].get_move(self.board)
            if uci:
                try:
                    cand = chesslib.Move.from_uci(uci)
                    if cand in legal:
                        mv = cand
                except ValueError:
                    mv = None
        self.push(mv or random.choice(legal), p)
        return True

    # ---- payloads ----
    def rows(self):
        """The board as eight rows of eight, rank 8 first, each cell None or
        a two-character code like 'wQ' — straightforward to render."""
        out = []
        for rank in range(7, -1, -1):
            row = []
            for f in range(8):
                pc = self.board.piece_at(chesslib.square(f, rank))
                row.append(None if pc is None else
                           ('w' if pc.color == chesslib.WHITE else 'b')
                           + pc.symbol().upper())
            out.append(row)
        return out

    def result_summary(self):
        return {
            'game': 'chess', 'variant': self.variant, 'reason': self.end_reason,
            'players': [{'name': p['name'], 'score': self.scores[i], 'penalties': 0,
                         'kind': p['kind'], 'profile': p['profile'],
                         'forfeited': i == self.forfeit_idx}
                        for i, p in enumerate(self.players)],
        }

    def to_dict(self, viewer=None):
        # Chess is perfect information, so every viewer sees the same thing.
        legal = {}
        if not self.over:
            for m in self.board.legal_moves:
                legal.setdefault(chesslib.square_name(m.from_square),
                                 []).append(chesslib.square_name(m.to_square))
        return {
            'kind': 'chess', 'variant': self.variant,
            'fen': self.board.fen(), 'board': self.rows(),
            'players': [{'cid': p['cid'], 'name': p['name'], 'kind': p['kind'],
                         'profile': p['profile'],
                         'colour': 'white' if i == 0 else 'black',
                         'score': self.scores[i]}
                        for i, p in enumerate(self.players)],
            'active': self.active, 'over': self.over, 'reason': self.end_reason,
            'last': self.last, 'log': self.log[-5:],
            'check': self.board.is_check(),
            'moveNo': self.board.fullmove_number,
            'history': list(self.sans),
            'legal': legal,
            # White-positive centipawns, default weights. The commentary reads
            # this; sending it costs one features() pass per broadcast and
            # saves the client from re-implementing an evaluator in JS.
            'eval': int(round(chess_static_eval(self.board))),
            # What the AI knows and what the last game taught it — enough for a
            # "how is it learning?" panel without another round trip.
            'ai': {str(i): a.stats() for i, a in self.ai.items()},
            'learned': {str(i): r for i, r in self.reports.items()},
        }


GAMES = {
    'quixx': {'title': 'Quixx', 'min': 2, 'max': MAX_SEATS, 'engine': Quixx,
              'icon': '🎲', 'variants': VARIANTS,
              'points': POINTS, 'penalty': 5, 'sheet': 'quixx-scoresheet.png',
              'vnames': {'standard': 'Standard', 'colors': 'Mixed Colors',
                         'numbers': 'Mixed Numbers', 'both': 'Both Mixed'},
              'blurb': 'Roll, cross off numbers left to right, lock a row '
                       'before anyone else. 2–4 players.'},
    'mancala': {'title': 'Mancala', 'min': 2, 'max': 2, 'engine': Mancala,
                'icon': '🫘', 'variants': {'standard'},
                'vnames': {'standard': 'Kalah'},
                'blurb': 'Sow stones around the board, capture from an empty '
                         'pit, finish with the fullest store. 2 players.'},
    'euchre': {'title': 'Euchre', 'min': 4, 'max': 4, 'engine': Euchre,
               'icon': '🃏', 'variants': {'standard', 'stick'}, 'teams': True,
               'vnames': {'standard': 'Standard', 'stick': 'Stick the Dealer'},
               'blurb': 'Partners across the table, bowers over aces. Name '
                        'trump and take three tricks — or get euchred. '
                        'First to 10. 4 players.'},
}

# Chess only appears if python-chess is installed. Registering it conditionally
# rather than always is what keeps the dependency optional: without it the
# lobby is exactly what it was before.
if CHESS_OK:
    GAMES['chess'] = {
        'title': 'Chess', 'min': 2, 'max': 2, 'engine': Chess,
        'icon': '♟️', 'variants': {'standard'},
        'vnames': {'standard': 'Standard'},
        'blurb': 'The old one. Play White against an AI that keeps a book on '
                 'you and gets harder the more you play it. 2 players.'}

# ============================================================
#  Stats backend — persisted to stats.json
# ============================================================
STATS_FILE = os.path.join(BASE, 'stats.json')
STATS_VERSION = 1
ELO_START = 1200.0
ELO_K = 24.0          # rating points swapped on an even-odds win
# A rating is only meaningful once it has games behind it. The bots rack up
# thousands against each other while a person plays a handful a week, so a
# human's rating would crawl while the bots' pool drifted away from it. New
# players move at a bigger K until their rating has settled.
ELO_K_NEW = 48.0
ELO_PROVISIONAL = 30  # games before a rating stops being provisional
HISTORY_MAX = 100     # most recent finished games kept in the file

# Players are keyed by lowercased name: client ids are per-session, names are
# what actually persists between visits on a home network.
STATS = {'version': STATS_VERSION, 'updated': None, 'players': {}, 'history': []}


def iso_now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def blank_player(name):
    return {'name': name, 'ai': False, 'profile': None, 'firstSeen': iso_now(),
            'lastPlayed': None, 'games': {}}


def blank_record():
    return {'elo': ELO_START, 'played': 0, 'wins': 0, 'losses': 0, 'ties': 0,
            'forfeits': 0, 'pointsFor': 0, 'pointsAgainst': 0, 'bestScore': 0,
            'penalties': 0, 'streak': 0, 'bestStreak': 0, 'placeSum': 0,
            'seatsSum': 0, 'variants': {}}


def normalize_record(rec):
    """Fill in fields added after a stats.json was first written."""
    for k, v in blank_record().items():
        rec.setdefault(k, v)
    return rec


def load_stats():
    """Read stats.json, tolerating a missing or damaged file."""
    global STATS
    try:
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or not isinstance(data.get('players'), dict):
        data = {'version': STATS_VERSION, 'players': {}, 'history': []}
    if not isinstance(data.get('history'), list):
        data['history'] = []
    data['version'] = STATS_VERSION
    STATS = data
    save_stats()


def save_stats():
    """Write via a temp file so a crash mid-write can't shred the stats."""
    STATS['updated'] = iso_now()
    tmp = STATS_FILE + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(STATS, f, indent=2)
        os.replace(tmp, STATS_FILE)
    except OSError as e:
        print('  [stats] could not write %s: %s' % (STATS_FILE, e))


def elo_expected(a, b):
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def bump_streak(rec, score):
    if score == 1.0:
        rec['streak'] = rec['streak'] + 1 if rec['streak'] > 0 else 1
        rec['bestStreak'] = max(rec['bestStreak'], rec['streak'])
    elif score == 0.0:
        rec['streak'] = rec['streak'] - 1 if rec['streak'] < 0 else -1
    else:
        rec['streak'] = 0


def placements(scores, quit_):
    """1-based finishing places, highest score first, walkouts always last.
    Equal scores share a place."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: (quit_[i], -scores[i]))
    place, step = [0] * n, 0
    for rank, i in enumerate(order):
        prev = order[rank - 1]
        if rank and (quit_[i], scores[i]) != (quit_[prev], scores[prev]):
            step = rank
        place[i] = step + 1
    return place, order


def record_result(summary):
    """Fold one finished 2–4 player game into STATS and save.

    Ratings use a pairwise round robin: every seat is scored against every
    other seat and the K-factor is split between those matchups, so a 2-player
    game behaves exactly as it always did."""
    ps = summary.get('players') or []
    if len(ps) < 2:
        return
    game = summary.get('game') or 'quixx'
    variant = summary.get('variant') or 'standard'
    names = [str(p.get('name') or '?').strip() or '?' for p in ps]
    scores = [int(p.get('score') or 0) for p in ps]
    quit_ = [bool(p.get('forfeited')) for p in ps]
    place, order = placements(scores, quit_)

    # One rating per name. If a name sits twice — two identical bots, or two
    # humans who typed the same thing — only the better finish is rated; you
    # cannot play yourself.
    keep, seen = [], set()
    for i in order:
        key = names[i].lower()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) < 2:
        return

    recs = []
    for i in keep:
        pl = STATS['players'].setdefault(names[i].lower(), blank_player(names[i]))
        pl['name'] = names[i]                   # keep the latest capitalisation
        pl['lastPlayed'] = iso_now()
        pl['ai'] = ps[i].get('kind') == 'ai'
        pl['profile'] = ps[i].get('profile')
        pl.setdefault('games', {})
        rec = normalize_record(pl['games'].setdefault(game, blank_record()))
        pl['games'][game] = rec
        recs.append(rec)

    m = len(keep)
    # Each side moves at its own K, so a newcomer converges quickly against
    # opponents whose ratings are already settled. That does mean a game is no
    # longer strictly zero-sum — the usual trade for letting new ratings find
    # their level in a handful of games rather than a hundred.
    ks = [(ELO_K_NEW if rec['played'] < ELO_PROVISIONAL else ELO_K) / (m - 1)
          for rec in recs]
    deltas = [0.0] * m
    for a in range(m):
        for b in range(a + 1, m):
            pa, pb = place[keep[a]], place[keep[b]]
            sa = 1.0 if pa < pb else (0.5 if pa == pb else 0.0)
            ea = elo_expected(recs[a]['elo'], recs[b]['elo'])
            deltas[a] += ks[a] * (sa - ea)
            deltas[b] += ks[b] * ((1.0 - sa) - (1.0 - ea))

    # How many distinct sides share each place. In a partnership game a pair
    # counts once, so both halves of a winning team are credited with a win
    # instead of looking like they tied with each other.
    units = {}
    for i in keep:
        t = ps[i].get('team')
        units.setdefault(place[i], set()).add(('team', t) if t is not None
                                              else ('seat', i))
    shared = {k: len(v) for k, v in units.items()}

    def against(i):
        """Whose score counts as the one played against."""
        t = ps[i].get('team')
        if t is None:
            return [j for j in keep if j != i]
        return ([j for j in keep if ps[j].get('team') != t]
                or [j for j in keep if j != i])

    entries = []
    for a, i in enumerate(keep):
        rec = recs[a]
        before = rec['elo']
        rec['elo'] = round(before + deltas[a], 1)
        won = place[i] == 1 and shared[1] == 1
        tied = place[i] == 1 and shared[1] > 1
        rec['played'] += 1
        rec['wins'] += won
        rec['ties'] += tied
        rec['losses'] += place[i] != 1
        rec['forfeits'] += quit_[i]
        rec['pointsFor'] += scores[i]
        rec['pointsAgainst'] += max(scores[j] for j in against(i))
        rec['bestScore'] = max(rec['bestScore'], scores[i])
        rec['penalties'] += int(ps[i].get('penalties') or 0)
        rec['placeSum'] += place[i]
        rec['seatsSum'] += m
        bump_streak(rec, 1.0 if won else (0.5 if tied else 0.0))
        v = rec['variants'].setdefault(variant, {'played': 0, 'wins': 0})
        v['played'] += 1
        v['wins'] += won
        entries.append({'name': names[i], 'score': scores[i], 'place': place[i],
                        'result': 'win' if won else ('tie' if tied else 'loss'),
                        'elo': rec['elo'], 'eloDelta': round(deltas[a], 1),
                        'forfeited': quit_[i], 'ai': ps[i].get('kind') == 'ai',
                        'profile': ps[i].get('profile')})

    # Self-play counts towards ratings and records — that is the whole point —
    # but it stays out of the recent-games list so training cannot bury the
    # games people actually played.
    if not summary.get('selfPlay'):
        entries.sort(key=lambda e: e['place'])
        STATS['history'].insert(0, {
            'ts': iso_now(), 'game': game, 'variant': variant, 'seats': len(ps),
            'reason': summary.get('reason'), 'players': entries})
        del STATS['history'][HISTORY_MAX:]
    save_stats()


def leaderboard(game, limit=25):
    # Bots from an older roster keep their record — the games people won
    # against them really happened — but are flagged so the board can say so.
    current = set(p['name'].lower() for p in roster())
    rows = []
    for key, pl in STATS['players'].items():
        rec = (pl.get('games') or {}).get(game)
        if not rec or not rec.get('played'):
            continue
        rec = normalize_record(rec)
        rows.append({'name': pl.get('name') or key,
                     'ai': bool(pl.get('ai')), 'profile': pl.get('profile'),
                     'provisional': rec['played'] < ELO_PROVISIONAL,
                     'retired': bool(pl.get('ai')) and key.split(' ')[0] not in current
                     and key not in current,
                     'elo': round(rec['elo'], 1), 'played': rec['played'],
                     'wins': rec['wins'], 'losses': rec['losses'],
                     'ties': rec['ties'], 'best': rec['bestScore'],
                     'avg': round(rec['pointsFor'] / rec['played'], 1),
                     'streak': rec['streak'], 'lastPlayed': pl.get('lastPlayed')})
    rows.sort(key=lambda r: (-r['elo'], -r['wins'], r['name'].lower()))
    return rows[:limit]


def stats_payload():
    return {'leaderboard': {g: leaderboard(g) for g in GAMES},
            'recent': STATS['history'][:10],
            'profiles': profiles_payload(), 'training': dict(TRAINING),
            'eloTrack': ELO_TRACK}


def record_room_result(r):
    """Called once per finished game, from both the normal and forfeit paths."""
    g = r.get('engine')
    if r.get('recorded') or not g or not getattr(g, 'over', False):
        return
    if not hasattr(g, 'result_summary'):
        return
    r['recorded'] = True
    record_result(g.result_summary())


# ============================================================
#  AI profiles & self-play training — persisted to ai_profiles.json
# ============================================================
PROFILES_FILE = os.path.join(BASE, 'ai_profiles.json')
PROFILES = {'version': 3, 'updated': None, 'profiles': {}}

# What a bot's learnable weights are, per game. Adding a game here is all the
# trainer needs to start tuning brains for it.
BRAINS = {
    'quixx': {'keys': WEIGHT_KEYS, 'default': DEFAULT_WEIGHTS,
              'bounds': WEIGHT_BOUNDS, 'mutation': MUTATION, 'jitter': BIRTH_JITTER},
    'mancala': {'keys': M_KEYS, 'default': M_DEFAULT, 'bounds': M_BOUNDS,
                'mutation': M_MUTATION, 'jitter': M_JITTER},
    'euchre': {'keys': E_KEYS, 'default': E_DEFAULT, 'bounds': E_BOUNDS,
               'mutation': E_MUTATION, 'jitter': E_JITTER},
}

# Rounds are attempts, not gains: most are rejected, so a small run usually
# changes nothing. 30 gives a 5-bot roster ~6 attempts each, which is where
# improvements actually start showing up.
DEFAULT_ROUNDS = 30
MAX_ROUNDS = 200
TRAIN_GAMES = 120        # games in a candidate's first screening
CONFIRM_GAMES = 160      # a promising candidate has to prove it a second time
ADOPT_EDGE = 0.545       # share needed to earn that second look
CONFIRM_EDGE = 0.52      # share needed across both, to actually be adopted
BASELINE_GAMES = 200     # games used to measure a profile against its original
TRAINING = {'running': False, 'round': 0, 'rounds': 0, 'who': None,
            'adopted': 0, 'games': 0, 'game': 'quixx', 'started': None,
            'finished': None, 'log': []}


def birth_weights(game='quixx'):
    spec = BRAINS[game]
    w = {}
    for k in spec['keys']:
        lo, hi = spec['bounds'][k]
        w[k] = round(min(hi, max(lo, spec['default'][k]
                                 + random.gauss(0.0, spec['jitter'][k]))), 3)
    return w


def blank_brain(game, weights=None):
    w = dict(weights) if weights else birth_weights(game)
    return {'weights': w, 'baseline': dict(w), 'generation': 0, 'trained': 0,
            'adopted': 0, 'vsBaseline': None, 'reference': False, 'history': []}


def brain(pid, game):
    """A profile's brain for one game, created on first use.

    A game with no entry in BRAINS has no weights for the trainer to tune and
    returns None — chess is the case in point. Its bots learn through their own
    ChessAI brain files instead, because the screening the trainer does here
    (120 games, then 160 to confirm) is minutes per game at chess speeds rather
    than milliseconds, and a chess bot that only learned by self-play would
    never notice the person it is actually playing.
    """
    p = PROFILES['profiles'].get(pid)
    if not p or game not in BRAINS:
        return None
    return p.setdefault('brains', {}).setdefault(game, blank_brain(game))


# Bots that play chess without any search at all, choosing from a learned move
# policy instead. Kept alongside the searchers rather than replacing them: the
# whole question about a learner is whether it is getting better, and that only
# has an answer if there is something rated to measure it against.
CHESS_LEARNERS = 2


def mint_profile(name, share, style='searcher'):
    """One bot. `share` spreads the handicaps across the roster.

    A learner gets almost no blunder rate. Its weakness is already structural —
    it cannot see a recapture — and stacking a random-move handicap on top
    would just make its rating measure the dice instead of the learning.
    """
    pid = 'ai_' + uuid.uuid4().hex[:8]
    learner = style == 'learner'
    return pid, {
        'id': pid, 'name': name, 'style': style,
        'blunder': 0.02 if learner else round(0.02 + 0.33 * share, 3),
        'noise': 0.2 if learner else round(0.2 + 2.6 * share, 2),
        'brains': {g: blank_brain(g) for g in BRAINS},
        'born': iso_now()}


def new_roster():
    """Mint a fresh field of bots: random names, and blunder rates spread
    across the range so the ratings have something real to separate."""
    total = AI_ROSTER_SIZE + CHESS_LEARNERS
    names = random.sample(AI_NAME_POOL, total)
    spread = [i / float(AI_ROSTER_SIZE - 1) for i in range(AI_ROSTER_SIZE)]
    random.shuffle(spread)
    styles = (['searcher'] * AI_ROSTER_SIZE) + (['learner'] * CHESS_LEARNERS)
    shares = spread + [0.0] * CHESS_LEARNERS
    profiles, order = {}, []
    for n, share, style in zip(names, shares, styles):
        pid, prof = mint_profile(n, share, style)
        profiles[pid] = prof
        order.append(pid)
    return {'version': 3, 'profiles': profiles, 'order': order}


def ensure_learners(data):
    """Top the roster up with searchless bots, without touching what is there.

    Existing bots keep their style and their ratings: converting one would
    silently invalidate an ELO earned by a completely different way of playing.
    New learners simply join at the default rating and earn their own.
    """
    have = sum(1 for p in data['profiles'].values()
               if p.get('style') == 'learner')
    if have >= CHESS_LEARNERS:
        return 0
    taken = set(p['name'] for p in data['profiles'].values())
    pool = [n for n in AI_NAME_POOL if n not in taken]
    added = 0
    for _ in range(CHESS_LEARNERS - have):
        if not pool:
            break
        name = random.choice(pool)
        pool.remove(name)
        pid, prof = mint_profile(name, 0.0, 'learner')
        data['profiles'][pid] = prof
        data['order'].append(pid)
        added += 1
    return added


def load_profiles():
    """Read ai_profiles.json; mint a new roster if it is missing or damaged."""
    global PROFILES
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    fresh = (not isinstance(data, dict) or not isinstance(data.get('profiles'), dict)
             or not data['profiles'])
    if fresh:
        data = new_roster()
    order = [p for p in (data.get('order') or []) if p in data['profiles']]
    order += [p for p in sorted(data['profiles']) if p not in order]
    data['order'] = order
    for pid, p in data['profiles'].items():
        p.setdefault('id', pid)
        p.setdefault('name', pid)
        p.setdefault('blunder', 0.1)
        p.setdefault('noise', 1.0)
        # Anything minted before styles existed searched, and its rating was
        # earned that way. Leave it a searcher.
        p.setdefault('style', 'searcher')
        p.setdefault('born', iso_now())
        # v2 kept one Quixx brain at the top level; move it under brains.quixx
        # so the thousands of games already trained into it are not thrown away
        if 'brains' not in p:
            p['brains'] = {'quixx': {
                'weights': p.get('weights') or dict(DEFAULT_WEIGHTS),
                'baseline': p.get('baseline') or dict(DEFAULT_WEIGHTS),
                'generation': p.get('generation', 0), 'trained': p.get('trained', 0),
                'adopted': p.get('adopted', 0), 'vsBaseline': p.get('vsBaseline'),
                'reference': False, 'history': p.get('history') or []}}
        for k in ('weights', 'baseline', 'generation', 'trained', 'adopted',
                  'vsBaseline', 'history'):
            p.pop(k, None)
        for game, spec in BRAINS.items():
            b = p['brains'].setdefault(game, blank_brain(game))
            b.setdefault('reference', False)
            b.setdefault('history', [])
            for f, d in (('generation', 0), ('trained', 0), ('adopted', 0),
                         ('vsBaseline', None)):
                b.setdefault(f, d)
            b.setdefault('weights', dict(spec['default']))
            b.setdefault('baseline', dict(b['weights']))
            for k in spec['keys']:
                lo, hi = spec['bounds'][k]
                b['weights'][k] = min(hi, max(lo, float(b['weights'].get(k, spec['default'][k]))))
                b['baseline'].setdefault(k, spec['default'][k])
            for d in (b['weights'], b['baseline']):   # drop retired weights
                for k in [k for k in d if k not in spec['keys']]:
                    del d[k]
    designate_reference(data, 'mancala')
    if CHESS_OK:
        added = ensure_learners(data)
        if added and not fresh:
            print('  [ai] added %d searchless chess bot(s) to the roster' % added)
    data['version'] = 3
    PROFILES = data
    save_profiles()
    return fresh


def designate_reference(data, game):
    """One bot per game plays the original algorithm untouched, as the yardstick
    everything else is measured against. The sharpest bot gets the job so the
    benchmark really is the one to beat."""
    profs = [data['profiles'][p] for p in data['order'] if p in data['profiles']]
    if not profs:
        return
    current = [p for p in profs if p['brains'].get(game, {}).get('reference')]
    if current:
        return
    pick = min(profs, key=lambda p: p.get('blunder', 1.0))
    b = pick['brains'].setdefault(game, blank_brain(game))
    b['reference'] = True
    b['weights'] = dict(BRAINS[game]['default'])     # zero == the original
    b['baseline'] = dict(b['weights'])


def roster():
    return [PROFILES['profiles'][p] for p in PROFILES['order']
            if p in PROFILES['profiles']]


def save_profiles():
    PROFILES['updated'] = iso_now()
    tmp = PROFILES_FILE + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(PROFILES, f, indent=2)
        os.replace(tmp, PROFILES_FILE)
    except OSError as e:
        print('  [ai] could not write %s: %s' % (PROFILES_FILE, e))


def profile_seat(pid, name=None, game='quixx'):
    """A seat dict for one AI, carrying a snapshot of its brain for this game."""
    p = PROFILES['profiles'].get(pid)
    if not p:
        return None
    b = brain(pid, game) or {}
    return {'id': 'ai:' + pid, 'name': name or p['name'], 'kind': 'ai',
            'profile': pid, 'weights': dict(b.get('weights') or {}),
            'reference': bool(b.get('reference')),
            'blunder': p['blunder'], 'noise': p['noise'],
            'style': p.get('style', 'searcher')}


def mutate(w, game='quixx'):
    spec = BRAINS[game]
    out = {}
    for k in spec['keys']:
        lo, hi = spec['bounds'][k]
        out[k] = round(min(hi, max(lo, w[k] + random.gauss(0.0, spec['mutation'][k]))), 3)
    return out


def sim_seat(name, prof, weights, reference=False):
    """A throwaway seat used for unrated trial games."""
    return {'id': 'sim:' + name, 'name': name, 'kind': 'ai', 'profile': prof['id'],
            'weights': dict(weights), 'reference': reference,
            'blunder': prof['blunder'], 'noise': prof['noise']}


def sim_game(seats, game='quixx', variant=None):
    """Play a table of AI out to the end, as fast as the CPU allows."""
    spec = GAMES[game]
    seats = seats[:spec['max']]
    g = spec['engine'](variant or random.choice(sorted(spec['variants'])), seats)
    for _ in range(4000):
        if not g.bot_step():
            break
    return g


def match(prof, wa, wb, games, game='quixx'):
    """Unrated head-to-head between two brains wearing the same bot's
    handicaps. Returns wa's score share, seats alternated so going first is
    not an advantage. In a partnership game each brain plays a whole side, so
    seats 0 and 2 share one brain and seats 1 and 3 the other."""
    pts = 0.0
    for i in range(games):
        ia = i % 2
        ws = [None, None]
        ws[ia], ws[1 - ia] = wa, wb
        if GAMES[game].get('teams'):
            seats = [sim_seat('S%d' % k, prof, ws[k % 2]) for k in range(4)]
        else:
            seats = [sim_seat('S0', prof, ws[0]), sim_seat('S1', prof, ws[1])]
        g = sim_game(seats, game)
        ta, tb = g.total(ia), g.total(1 - ia)
        pts += 0.5 if ta == tb else (1.0 if ta > tb else 0.0)
    return pts / games


def train_round(pid, game='quixx'):
    """One hill-climbing step: mutate, screen the mutant against the incumbent,
    and if it looks good make it prove that again on fresh games before it is
    adopted. A single 120-game screen is only about one standard error wide, so
    without the second look a coin-flip mutation gets adopted on luck often
    enough to walk a bot backwards."""
    with LOCK:
        prof = dict(PROFILES['profiles'][pid])
        b = brain(pid, game)
        if b.get('reference'):
            return None, False, 0        # the yardstick never moves
        base = dict(b['weights'])
    cand = mutate(base, game)
    share = match(prof, cand, base, TRAIN_GAMES, game)      # slow, runs unlocked
    played = TRAIN_GAMES
    if share >= ADOPT_EDGE:
        again = match(prof, cand, base, CONFIRM_GAMES, game)
        share = ((share * TRAIN_GAMES + again * CONFIRM_GAMES)
                 / float(TRAIN_GAMES + CONFIRM_GAMES))
        played += CONFIRM_GAMES
    with LOCK:
        b = brain(pid, game)
        b['trained'] += played
        adopted = played > TRAIN_GAMES and share >= CONFIRM_EDGE
        if adopted:
            b['weights'] = cand
            b['generation'] += 1
            b['adopted'] += 1
            b['history'].insert(0, {'ts': iso_now(), 'generation': b['generation'],
                                    'share': round(share, 3), 'weights': cand})
            del b['history'][30:]
        save_profiles()
    return share, adopted, played


def measure_baseline(pid, game='quixx'):
    """How a bot's current brain fares against the one it was born with."""
    with LOCK:
        prof = dict(PROFILES['profiles'][pid])
        b = brain(pid, game)
        if b.get('reference'):
            return None
        cur, base = dict(b['weights']), dict(b['baseline'])
    share = match(prof, cur, base, BASELINE_GAMES, game)
    with LOCK:
        brain(pid, game)['vsBaseline'] = round(share, 3)
        save_profiles()
    return share


def ladder_game(game='quixx'):
    """One *rated* AI-vs-AI game. Goes through the normal stats path, so the
    bots' ELOs move against each other exactly like human games do."""
    with LOCK:
        pool = PROFILES['order'][:]
    if len(pool) < 2:
        return None
    cap = min(GAMES[game]['max'], len(pool))
    if cap < GAMES[game]['min']:
        return None            # not enough bots on the roster to fill a table
    lo = min(GAMES[game]['min'], cap)
    picks = random.sample(pool, random.randint(lo, cap))
    with LOCK:
        seats = [profile_seat(p, None, game) for p in picks]
    seats = [s for s in seats if s]
    if len(seats) < 2:
        return None
    g = sim_game(seats, game)
    summary = g.result_summary()
    summary['selfPlay'] = True
    with LOCK:
        record_result(summary)
    return g


def run_training(rounds, ladder=4, game='quixx', verbose=False):
    """Work through the roster: one hill-climbing round per bot, then a few
    rated games so the ratings keep moving. Safe to run off-thread."""
    TRAINING['game'] = game
    with LOCK:
        track_elo(game)               # a starting point, so the graph opens flat
    for i in range(rounds):
        if not TRAINING['running']:
            break
        with LOCK:
            order = PROFILES['order'][:]
        if not order:
            break
        pid = order[i % len(order)]
        name = PROFILES['profiles'][pid]['name']
        with LOCK:
            TRAINING['round'] = i + 1
            TRAINING['who'] = name
        share, adopted, played = train_round(pid, game)
        for _ in range(ladder):
            ladder_game(game)
        if share is None:                 # the reference bot, left alone
            line = '%s is the reference — plays the original, never trains' % name
        else:
            line = '%s %s (%d%% in trials)' % (
                name, 'improved' if adopted else 'held its ground', round(share * 100))
        with LOCK:
            TRAINING['games'] += played + ladder
            TRAINING['adopted'] += 1 if adopted else 0
            TRAINING['log'].insert(0, line)
            del TRAINING['log'][6:]
            track_elo(game)
            broadcast_stats()
        if verbose:
            print('  round %d/%d — %s' % (i + 1, rounds, line))
    with LOCK:
        order = PROFILES['order'][:]
    for pid in order:
        if not TRAINING['running']:
            break
        share = measure_baseline(pid, game)
        if verbose and share is not None:
            print('  %s vs. the brain it was born with: %d%%'
                  % (PROFILES['profiles'][pid]['name'], round(share * 100)))


ELO_TRACK_MAX = 90        # points kept per bot; the window scrolls past that
ELO_TRACK = {}            # game -> {name: [elo, ...]}


def track_elo(game):
    """Sample every bot's rating so the training graph has a line to draw.
    Older points fall off the left as new ones arrive."""
    series = ELO_TRACK.setdefault(game, {})
    live = set()
    for p in roster():
        live.add(p['name'])
        rec = ((STATS['players'].get(p['name'].lower()) or {}).get('games') or {}).get(game)
        s = series.setdefault(p['name'], [])
        s.append(round(rec['elo'], 1) if rec else ELO_START)
        del s[:-ELO_TRACK_MAX]
    for gone in [n for n in series if n not in live]:
        del series[gone]
    return series


def calibrate(games=90, game='quixx'):
    """A quick rated ladder so a brand-new roster has meaningful ratings the
    first time anyone looks at the leaderboard."""
    for _ in range(games):
        ladder_game(game)
    with LOCK:
        broadcast_stats()


def train_worker(rounds, ladder, game='quixx'):
    try:
        run_training(rounds, ladder, game)
    finally:
        with LOCK:
            TRAINING['running'] = False
            TRAINING['who'] = None
            TRAINING['finished'] = iso_now()
            save_profiles()
            broadcast_stats()


def profiles_payload():
    """The AI roster with the rating each has actually earned — no difficulty
    labels, the ELO is the difficulty."""
    out = []
    for p in roster():
        pl = STATS['players'].get(p['name'].lower()) or {}
        elos, played = {}, {}
        for gk, rec in (pl.get('games') or {}).items():
            if rec.get('played'):
                elos[gk] = round(rec['elo'], 1)
                played[gk] = rec['played']
        brains = {}
        for gk in BRAINS:
            b = (p.get('brains') or {}).get(gk) or {}
            brains[gk] = {'generation': b.get('generation', 0),
                          'trained': b.get('trained', 0),
                          'adopted': b.get('adopted', 0),
                          'vsBaseline': b.get('vsBaseline'),
                          'reference': bool(b.get('reference')),
                          'weights': b.get('weights') or {}}
        q = brains.get('quixx', {})
        out.append({'id': p['id'], 'name': p['name'],
                    'style': p.get('style', 'searcher'),
                    'elos': elos, 'plays': played, 'brains': brains,
                    'elo': elos.get('quixx'), 'played': played.get('quixx', 0),
                    'generation': q.get('generation', 0), 'trained': q.get('trained', 0),
                    'adopted': q.get('adopted', 0), 'vsBaseline': q.get('vsBaseline'),
                    'weights': q.get('weights', {})})
    out.sort(key=lambda r: (r['elo'] is None, -(r['elo'] or 0)))
    return out


# ============================================================
#  Lobby / rooms
# ============================================================

def make_bots(ids):
    """Seat the chosen bots. The same bot picked twice is numbered, so each
    seat at the table is a distinct name (and a distinct rating)."""
    out, counts = [], {}
    for pid in ids:
        p = PROFILES['profiles'].get(pid)
        if not p:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        name = p['name'] if counts[pid] == 1 else '%s %d' % (p['name'], counts[pid])
        out.append({'id': 'ai:%s:%d' % (pid, counts[pid]), 'profile': pid, 'name': name})
    return out


def room_seats(r):
    """Turn order: humans in the order they joined, then the AI seats."""
    out = [{'id': c, 'name': CLIENTS[c]['name'], 'kind': 'human'}
           for c in r['players'] if c in CLIENTS]
    for b in r['bots']:
        seat = profile_seat(b['profile'], b['name'], r['game'])
        if seat:
            seat['id'] = b['id']
            out.append(seat)
    return out


def refresh_status(r):
    if r['status'] in ('playing', 'finished'):
        return
    seated = len(r['players']) + len(r['bots'])
    r['status'] = ('ready' if len(r['players']) >= r['humans']
                   and seated >= GAMES[r['game']]['min'] else 'waiting')


def parse_table(body, joined, cap=MAX_SEATS):
    """Validate a humans + AI table setup against the game's own seat limit.
    Raises ValueError with a message."""
    try:
        humans = int(body.get('humans', 2))
    except (TypeError, ValueError):
        raise ValueError('Bad number of human players.')
    picks = body.get('bots')
    if picks is None:
        picks = []
    if not isinstance(picks, list) or len(picks) > MAX_SEATS:
        raise ValueError('Bad AI list.')
    picks = [str(x) for x in picks]
    for pid in picks:
        if pid not in PROFILES['profiles']:
            raise ValueError('Unknown AI player.')
    if humans < 1 or humans > cap:
        raise ValueError('Humans must be between 1 and %d.' % cap)
    if humans < joined:
        raise ValueError('%d players have already joined.' % joined)
    if humans + len(picks) > cap:
        raise ValueError('This game seats %d.' % cap)
    if humans + len(picks) < 2:
        raise ValueError('A game needs at least 2 seats.')
    return humans, picks


def room_summary(r):
    return {'id': r['id'], 'name': r['name'], 'game': r['game'],
            'gameTitle': GAMES[r['game']]['title'], 'variant': r['variant'],
            'status': r['status'], 'humans': r['humans'], 'maxSeats': MAX_SEATS,
            'bots': [{'profile': b['profile'], 'name': b['name']} for b in r['bots']],
            'players': [CLIENTS[c]['name'] for c in r['players'] if c in CLIENTS],
            'spectators': len(r['spectators']),
            'host': CLIENTS.get(r['host'], {}).get('name', '?')}


def room_full(r, cid=None):
    """The room as one client sees it. Games with hidden information build a
    different state per viewer, so this is never cached or shared."""
    d = room_summary(r)
    d['playerIds'] = list(r['players'])
    d['hostId'] = r['host']
    d['state'] = r['engine'].to_dict(cid) if r['engine'] else None
    return d


def presence():
    """Who is actually here. A client counts as online only while it holds an
    open event stream, so stale sessions drop off the list by themselves."""
    out = []
    for cid, c in CLIENTS.items():
        if not c['queues']:
            continue
        where, rid = 'Choosing a game', None
        r = ROOMS.get(c.get('room'))
        if r:
            rid = r['id']
            if cid not in r['players']:
                where = 'Spectating ' + GAMES[r['game']]['title']
            elif r['status'] == 'playing':
                where = 'Playing ' + GAMES[r['game']]['title']
            else:
                where = 'Waiting to start'
        elif c.get('lobby') in GAMES:
            where = GAMES[c['lobby']]['title'] + ' lobby'
        out.append({'cid': cid, 'name': c['name'], 'where': where, 'room': rid})
    out.sort(key=lambda p: p['name'].lower())
    return out


def games_payload():
    out = []
    for key, g in GAMES.items():
        rooms = [r for r in ROOMS.values() if r['game'] == key]
        out.append({'key': key, 'title': g['title'], 'icon': g.get('icon', '🎮'),
                    'blurb': g.get('blurb', ''), 'min': g['min'], 'max': g['max'],
                    'variants': sorted(g.get('variants', VARIANTS)),
                    'vnames': g.get('vnames', {}),
                    'points': g.get('points'), 'penalty': g.get('penalty'),
                    'sheet': g.get('sheet'),
                    'open': len([r for r in rooms if r['status'] == 'waiting']),
                    'playing': len([r for r in rooms if r['status'] == 'playing']),
                    'here': len([c for c in CLIENTS.values()
                                 if c['queues'] and c.get('lobby') == key])})
    return out


def hub_payload():
    return {'rooms': [room_summary(r) for r in ROOMS.values()],
            'players': presence(), 'games': games_payload()}


def player_detail(name):
    """Everything known about one player, for the profile card."""
    key = str(name or '').strip().lower()
    pl = STATS['players'].get(key)
    if not pl:
        return None
    games = {}
    for gk, rec in (pl.get('games') or {}).items():
        rec = normalize_record(rec)
        if not rec['played']:
            continue
        board = leaderboard(gk, limit=10000)
        rank = None
        for i, row in enumerate(board):
            if row['name'].lower() == key:
                rank = i + 1
                break
        games[gk] = {
            'title': GAMES[gk]['title'] if gk in GAMES else gk,
            'elo': round(rec['elo'], 1), 'rank': rank, 'of': len(board),
            'played': rec['played'], 'wins': rec['wins'],
            'losses': rec['losses'], 'ties': rec['ties'],
            'best': rec['bestScore'],
            'avg': round(rec['pointsFor'] / rec['played'], 1),
            'streak': rec['streak'], 'bestStreak': rec['bestStreak'],
            'penalties': rec['penalties'], 'forfeits': rec['forfeits'],
            'variants': rec.get('variants') or {}}
    recent = [h for h in STATS['history']
              if any(str(p.get('name', '')).lower() == key for p in h['players'])][:8]
    return {'name': pl.get('name') or name, 'ai': bool(pl.get('ai')),
            'firstSeen': pl.get('firstSeen'), 'lastPlayed': pl.get('lastPlayed'),
            'games': games, 'recent': recent}


def push(cid, event, data):
    c = CLIENTS.get(cid)
    if not c:
        return
    for q in list(c['queues']):
        q.put((event, data))


def broadcast_lobby():
    data = hub_payload()
    for cid in list(CLIENTS):
        push(cid, 'lobby', data)


def broadcast_room(r):
    # built per recipient: in Euchre the same table looks different from every
    # seat, and a spectator must not be handed anybody's cards
    for cid in list(r['players']) + list(r['spectators']):
        push(cid, 'room', room_full(r, cid))


def broadcast_stats():
    data = stats_payload()
    for cid in list(CLIENTS):
        push(cid, 'stats', data)


def advance(r):
    """Let the AI seats play until the table waits on a human, then bank the
    result if that finished the game."""
    g = r.get('engine')
    if not g:
        return
    # generous: a full Euchre match is many hands of many small bot decisions
    for _ in range(3000):
        if not g.bot_step():
            break
    if g.over and r['status'] == 'playing':
        r['status'] = 'finished'
        record_room_result(r)
        broadcast_lobby()
        broadcast_stats()


def leave_room(cid, notify=True):
    """Remove client from its room. Handles forfeits and empty-room cleanup."""
    c = CLIENTS.get(cid)
    if not c or not c['room']:
        return
    rid = c['room']
    c['room'] = None
    r = ROOMS.get(rid)
    if not r:
        return
    if cid in r['spectators']:
        r['spectators'].discard(cid)
    elif cid in r['players']:
        if r['status'] == 'playing' and r['engine'] and not r['engine'].over:
            r['engine'].forfeit(cid)
            r['status'] = 'finished'
            record_room_result(r)
            if notify:
                broadcast_stats()
        r['players'].remove(cid)
        if r['host'] == cid and r['players']:
            r['host'] = r['players'][0]
        refresh_status(r)
    if not r['players']:
        del ROOMS[rid]
        r = None
    elif notify:
        broadcast_room(r)
    if notify:
        broadcast_lobby()


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return '127.0.0.1'


# ============================================================
#  HTTP handler
# ============================================================
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass  # keep the console clean

    # ---------- responses ----------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def err(self, msg, code=400):
        self.send_json({'error': msg}, code)

    def read_body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > 65536:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ---------- GET ----------
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            try:
                with open(os.path.join(BASE, 'index.html'), 'rb') as f:
                    body = f.read()
            except OSError:
                self.err('index.html missing next to homegames_server.py', 500)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        elif u.path == '/api/health':
            # Read-only liveness, used by deploy/deploy.sh to decide whether
            # restarting right now would land on top of somebody's game.
            # Rooms and clients live only in memory, so a restart drops every
            # game in progress; `playing` is what tells the deploy to wait.
            with LOCK:
                body = {'ok': True,
                        'players': sum(1 for c in CLIENTS.values() if c['queues']),
                        'rooms': len(ROOMS),
                        'playing': sum(1 for r in ROOMS.values()
                                       if r['status'] == 'playing'),
                        'chess': CHESS_OK}
            self.send_json(body)
        elif u.path == '/api/events':
            self.handle_sse(u)
        elif u.path == '/favicon.ico':
            self.send_response(204)
            self.send_header('Content-Length', '0')
            self.end_headers()
        else:
            self.send_asset(u.path)

    ASSET_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                   '.gif': 'image/gif', '.svg': 'image/svg+xml',
                   '.css': 'text/css', '.js': 'application/javascript'}

    def send_asset(self, path):
        """Serve a file sitting next to the server. Only the basename is used
        and the result must still live in BASE, so no request can climb out of
        the folder."""
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in self.ASSET_TYPES:
            self.err('Not found', 404)
            return
        full = os.path.abspath(os.path.join(BASE, name))
        if os.path.dirname(full) != os.path.abspath(BASE) or not os.path.isfile(full):
            self.err('Not found', 404)
            return
        try:
            with open(full, 'rb') as f:
                body = f.read()
        except OSError:
            self.err('Not found', 404)
            return
        self.send_response(200)
        self.send_header('Content-Type', self.ASSET_TYPES[ext])
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(body)

    def handle_sse(self, u):
        qs = parse_qs(u.query)
        cid = (qs.get('cid') or [''])[0]
        with LOCK:
            c = CLIENTS.get(cid)
            if not c:
                self.err('Unknown client — say hello first.', 403)
                return
            q = Queue()
            c['queues'].append(q)
            c['seen'] = time.time()
            # initial snapshot
            q.put(('lobby', hub_payload()))
            q.put(('stats', stats_payload()))
            if c['room'] and c['room'] in ROOMS:
                q.put(('room', room_full(ROOMS[c['room']], cid)))
            broadcast_lobby()              # tell everyone else they arrived
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                try:
                    # The keepalive doubles as disconnect detection: a dropped
                    # client is only noticed when a write fails, so this
                    # interval is how long a stale name sits in "who's here".
                    event, data = q.get(timeout=SSE_PING)
                    payload = ('event: %s\ndata: %s\n\n'
                               % (event, json.dumps(data))).encode('utf-8')
                except Empty:
                    payload = b': ping\n\n'
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with LOCK:
                c = CLIENTS.get(cid)
                if c and q in c['queues']:
                    c['queues'].remove(q)
                broadcast_lobby()          # they just dropped off the who's-here list

    # ---------- POST ----------
    def do_POST(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split('/') if p]  # ['api','rooms','<id>','join']
        body = self.read_body()
        if len(parts) < 2 or parts[0] != 'api':
            self.err('Not found', 404)
            return
        with LOCK:
            try:
                self.route(parts[1:], body)
            except Exception as e:  # keep the server alive no matter what
                self.err('Server error: %s' % e, 500)

    def route(self, parts, body):
        cmd = parts[0]

        if cmd == 'hello':
            name = str(body.get('name') or '').strip()[:14] or 'Player'
            cid = str(body.get('cid') or '')
            if cid and cid in CLIENTS:
                CLIENTS[cid]['name'] = name
                CLIENTS[cid]['seen'] = time.time()
            else:
                cid = uuid.uuid4().hex[:12]
                CLIENTS[cid] = {'name': name, 'queues': [], 'room': None,
                                'lobby': None, 'seen': time.time()}
            CLIENTS[cid].setdefault('lobby', None)
            room = None
            if CLIENTS[cid]['room'] and CLIENTS[cid]['room'] in ROOMS:
                room = room_full(ROOMS[CLIENTS[cid]['room']], cid)
            broadcast_lobby()
            payload = {'ok': True, 'cid': cid, 'name': name, 'room': room,
                       'lobby': CLIENTS[cid].get('lobby'),
                       'stats': stats_payload()}
            payload.update(hub_payload())
            self.send_json(payload)
            return

        if cmd == 'stats':
            self.send_json({'ok': True, 'stats': stats_payload()})
            return

        if cmd == 'player':
            d = player_detail(body.get('name'))
            if not d:
                self.err('No record for that player yet.', 404)
                return
            self.send_json({'ok': True, 'player': d})
            return

        # POST /api/chess/stats            -> every bot's chess brain
        # POST /api/chess/reset {profile}  -> wipe one bot's brain, or all of
        #                                     them if no profile is named
        if cmd == 'chess':
            if not CHESS_OK:
                self.err('Chess is unavailable: %s. Run "pip install chess".'
                         % CHESS_WHY, 501)
                return
            sub = parts[1] if len(parts) > 1 else 'stats'

            if sub == 'stats':
                out = []
                for p in roster():
                    ai = ChessAI(name=p['name'],
                                 data_path=chess_brain_path(p['id']),
                                 max_depth=Chess.depth_ceiling(p['blunder']),
                                 style=p.get('style', 'searcher'))
                    out.append({'profile': p['id'], 'name': p['name'],
                                'style': p.get('style', 'searcher'),
                                'stats': ai.stats()})
                self.send_json({'ok': True, 'bots': out})
                return

            if sub == 'reset':
                pid = body.get('profile')
                targets = [p for p in roster() if not pid or p['id'] == pid]
                if not targets:
                    self.err('Unknown AI player.', 404)
                    return
                for p in targets:
                    ChessAI(name=p['name'],
                            data_path=chess_brain_path(p['id']),
                            max_depth=Chess.depth_ceiling(p['blunder']),
                            style=p.get('style', 'searcher')).reset()
                self.send_json({'ok': True,
                                'reset': [p['name'] for p in targets]})
                return

            self.err('Unknown chess action.', 404)
            return

        if cmd == 'train':
            if body.get('stop'):
                TRAINING['running'] = False
                broadcast_stats()
                self.send_json({'ok': True})
                return
            if TRAINING['running']:
                self.err('Training is already running.')
                return
            try:
                rounds = int(body.get('rounds') or DEFAULT_ROUNDS)
            except (TypeError, ValueError):
                rounds = DEFAULT_ROUNDS
            rounds = max(1, min(MAX_ROUNDS, rounds))
            ladder = max(0, min(20, int(body.get('ladder') or 4)))
            tgame = str(body.get('game') or 'quixx')
            if tgame not in BRAINS:
                self.err('That game has no trainable AI.')
                return
            TRAINING.update({'running': True, 'round': 0, 'rounds': rounds,
                             'who': None, 'adopted': 0, 'games': 0, 'game': tgame,
                             'started': iso_now(), 'finished': None, 'log': []})
            threading.Thread(target=train_worker, args=(rounds, ladder, tgame),
                             daemon=True).start()
            broadcast_stats()
            self.send_json({'ok': True})
            return

        # everything below needs a known client
        cid = str(body.get('cid') or '')
        c = CLIENTS.get(cid)
        if not c:
            self.err('Unknown client — refresh the page.', 403)
            return
        c['seen'] = time.time()

        if cmd == 'rooms' and len(parts) == 1:
            # create a room
            game = str(body.get('game') or 'quixx')
            variant = str(body.get('variant') or 'standard')
            if game not in GAMES:
                self.err('Unknown game.')
                return
            if variant not in GAMES[game].get('variants', VARIANTS):
                self.err('Unknown variant.')
                return
            try:
                humans, levels = parse_table(body, 1, GAMES[game]['max'])
            except ValueError as e:
                self.err(str(e))
                return
            leave_room(cid, notify=False)
            rid = uuid.uuid4().hex[:8]
            ROOMS[rid] = {'id': rid, 'game': game, 'variant': variant,
                          'name': c['name'] + "'s " + GAMES[game]['title'],
                          'host': cid, 'players': [cid], 'spectators': set(),
                          'humans': humans, 'bots': make_bots(levels),
                          'status': 'waiting', 'engine': None, 'recorded': False,
                          'created': time.time()}
            refresh_status(ROOMS[rid])
            c['room'] = rid
            broadcast_lobby()
            broadcast_room(ROOMS[rid])
            self.send_json({'ok': True, 'room': room_full(ROOMS[rid], cid)})
            return

        if cmd == 'leave':
            leave_room(cid)
            self.send_json({'ok': True})
            return

        if cmd == 'lobby':
            # which game's lobby this client is browsing; None means the hub
            game = body.get('game')
            game = str(game) if game else None
            if game is not None and game not in GAMES:
                self.err('Unknown game.')
                return
            if game is None:
                leave_room(cid, notify=False)
            c['lobby'] = game
            broadcast_lobby()
            self.send_json({'ok': True, 'lobby': game})
            return

        if cmd == 'rooms' and len(parts) >= 3:
            rid, action = parts[1], parts[2]
            r = ROOMS.get(rid)
            if not r:
                self.err('That room no longer exists.', 404)
                return

            if action == 'join':
                if cid in r['players']:
                    self.send_json({'ok': True, 'room': room_full(r, cid)})
                    return
                if r['status'] not in ('waiting',) or len(r['players']) >= r['humans']:
                    self.err('Room is full or already playing — try spectating.')
                    return
                leave_room(cid, notify=False)
                r['players'].append(cid)
                c['room'] = rid
                refresh_status(r)
                broadcast_lobby()
                broadcast_room(r)
                self.send_json({'ok': True, 'room': room_full(r, cid)})
                return

            if action == 'spectate':
                if cid in r['players']:
                    self.err('You are playing in this room.')
                    return
                leave_room(cid, notify=False)
                r['spectators'].add(cid)
                c['room'] = rid
                broadcast_lobby()
                broadcast_room(r)
                self.send_json({'ok': True, 'room': room_full(r, cid)})
                return

            if action == 'config':
                if cid != r['host']:
                    self.err('Only the host can change the table.')
                    return
                if r['status'] == 'playing':
                    self.err('Finish the game before changing the table.')
                    return
                try:
                    humans, levels = parse_table(body, len(r['players']),
                                                 GAMES[r['game']]['max'])
                except ValueError as e:
                    self.err(str(e))
                    return
                r['humans'] = humans
                r['bots'] = make_bots(levels)
                refresh_status(r)
                broadcast_lobby()
                broadcast_room(r)
                self.send_json({'ok': True, 'room': room_full(r, cid)})
                return

            if action == 'start':
                if cid != r['host']:
                    self.err('Only the host can start.')
                    return
                seats = room_seats(r)
                need = GAMES[r['game']]['min']
                if len(seats) < need:
                    self.err('%s needs %d seats — add a player or an AI.'
                             % (GAMES[r['game']]['title'], need))
                    return
                if r['status'] == 'playing':
                    self.err('Already playing.')
                    return
                r['engine'] = GAMES[r['game']]['engine'](r['variant'], seats)
                r['status'] = 'playing'
                r['recorded'] = False          # a rematch is its own result
                advance(r)                     # in case an AI moves first
                broadcast_lobby()
                broadcast_room(r)
                self.send_json({'ok': True})
                return

            if action == 'action':
                if r['status'] != 'playing' or not r['engine']:
                    self.err('Game is not in progress.')
                    return
                # each engine owns its own verbs
                e = r['engine'].apply(cid, body)
                if e:
                    self.err(e)
                    return
                advance(r)                     # AI seats reply, then bank a result
                broadcast_room(r)
                self.send_json({'ok': True})
                return

            self.err('Unknown room action.', 404)
            return

        self.err('Not found', 404)


def reset_ai():
    """Wipe the bots' ratings and re-baseline their brains.

    Ratings only mean something relative to the evaluation that earned them, so
    after the scoring changes the old numbers are worse than useless — they
    label a bot's difficulty wrongly. Human records and game history are left
    completely alone."""
    load_stats()
    load_profiles()
    names = set(p['name'].lower() for p in roster())
    cleared = 0
    for key in list(STATS['players']):
        pl = STATS['players'][key]
        if pl.get('ai') and key in names:
            del STATS['players'][key]
            cleared += 1
    save_stats()
    for p in roster():
        for b in (p.get('brains') or {}).values():
            b['baseline'] = dict(b['weights'])   # today's brain is the new yardstick
            b['vsBaseline'] = None
    save_profiles()
    print('\n  Cleared ratings for %d bots; human records and history untouched.'
          % cleared)
    print('  Re-baselined %d brains against their current weights.' % len(roster()))
    print('  Playing calibration games...')
    TRAINING['running'] = True
    for g in BRAINS:
        calibrate(90, g)
    TRAINING['running'] = False
    for g in BRAINS:
        print('  %s:' % g)
        for row in sorted(profiles_payload(), key=lambda r: -(r['elos'].get(g) or 0)):
            print('    %-12s elo %-8s %d rated games'
                  % (row['name'], row['elos'].get(g, '—'), row['plays'].get(g, 0)))
    print()


def train_cli(rounds, game='quixx'):
    """python homegames_server.py --train [rounds] [game] — bulk training."""
    load_stats()
    if load_profiles():
        print('  Minted a new roster: %s'
              % ', '.join(p['name'] for p in roster()))
    ref = [p['name'] for p in roster() if (p['brains'].get(game) or {}).get('reference')]
    if ref:
        print('  Reference (plays the original, never trains): %s' % ref[0])
    print('\n  Training the AI by self-play: %d rounds '
          '(%d trial games each, plus rated ladder games).\n' % (rounds, TRAIN_GAMES))
    TRAINING.update({'running': True, 'rounds': rounds, 'round': 0, 'game': game,
                     'adopted': 0, 'games': 0, 'started': iso_now()})
    start = time.time()
    try:
        run_training(rounds, ladder=4, game=game, verbose=True)
    except KeyboardInterrupt:
        print('\n  Stopped early — progress so far is saved.')
    finally:
        TRAINING['running'] = False
        save_profiles()
    print('\n  %d rounds, %d improvements, %d games, %.0fs.'
          % (TRAINING['round'], TRAINING['adopted'], TRAINING['games'],
             time.time() - start))
    print('  Ratings after self-play (%s):' % game)
    rows = sorted(profiles_payload(),
                  key=lambda r: -(r['elos'].get(game) or 0))
    for row in rows:
        b = row['brains'].get(game, {})
        print('    %-12s elo %-8s %5d rated  gen %-3d %s'
              % (row['name'], row['elos'].get(game, '—'),
                 row['plays'].get(game, 0), b.get('generation', 0),
                 '(reference)' if b.get('reference') else ''))
    print()


def main():
    args = sys.argv[1:]
    if args and args[0] == '--reset-ai':
        reset_ai()
        return
    if args and args[0] == '--train':
        try:
            rounds = int(args[1]) if len(args) > 1 else DEFAULT_ROUNDS
        except ValueError:
            rounds = DEFAULT_ROUNDS
        tgame = args[2] if len(args) > 2 else 'quixx'
        if tgame not in BRAINS:
            print('  Unknown game %r — try one of: %s'
                  % (tgame, ', '.join(sorted(BRAINS))))
            return
        train_cli(max(1, min(500, rounds)), tgame)
        return
    load_stats()
    fresh = load_profiles()
    if fresh:
        # a brand-new roster has no ratings yet, and an unrated bot tells you
        # nothing about how hard it is — so play them in before anyone looks
        TRAINING['running'] = True
        threading.Thread(target=lambda: (calibrate(),
                                         TRAINING.update({'running': False})),
                         daemon=True).start()
    server = ThreadingHTTPServer(('', PORT), Handler)
    server.daemon_threads = True
    ip = lan_ip()
    print()
    print('  =========================================')
    print('   HomeGames server is running')
    print('   This computer:  http://localhost:%d' % PORT)
    print('   Your network:   http://%s:%d' % (ip, PORT))
    print('  =========================================')
    print('  Share the network address with other players.')
    print('  Stats & ELO: %s (%d players tracked)'
          % (os.path.basename(STATS_FILE), len(STATS['players'])))
    print('  AI roster:   %s' % ', '.join(p['name'] for p in roster()))
    if fresh:
        print('               (new — playing calibration games to rate them)')
    if CHESS_OK:
        print('  Chess:       on — brains in %s/'
              % os.path.basename(CHESS_BRAIN_DIR))
    else:
        # Said plainly rather than left to a missing lobby tile, since "where
        # did chess go" is otherwise a puzzle with no clue attached.
        print('  Chess:       off — needs python-chess ("pip install chess")')
    print('  Bulk-train with:  python homegames_server.py --train 30')
    print('  Press Ctrl+C to stop.')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nHomeGames server stopped.')


if __name__ == '__main__':
    main()
