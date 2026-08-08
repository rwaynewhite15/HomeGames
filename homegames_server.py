#!/usr/bin/env python3
"""
HomeGames — LAN game lobby server.
Pure Python standard library. No installs required.

Run:  python homegames_server.py
Then open http://localhost:4001 (or http://<your-LAN-IP>:4001 from other devices).

Platform: lobby with rooms you can host, join, or spectate.
Games: Quixx (standard / mixed colors / mixed numbers / both). More games plug in later.
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

    def to_dict(self):
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


def m_search(board, player, me, depth, alpha, beta):
    """Alpha-beta on the stone difference, the same evaluation the original
    used. Handles Mancala's extra-turn rule: sowing into your own store
    leaves it your move, so the node stays a maximising one."""
    moves = m_moves(board, player)
    if depth <= 0 or not moves:
        if not moves:
            b = board[:]
            m_sweep(b)
            return b[M_STORE[me]] - b[M_STORE[1 - me]]
        return board[M_STORE[me]] - board[M_STORE[1 - me]]
    maximizing = (player == me)
    best = -10 ** 6 if maximizing else 10 ** 6
    for pit in moves:
        b = board[:]
        extra, _ = m_sow(b, player, pit)
        nxt = player if extra else 1 - player
        val = m_search(b, nxt, me, depth - 1, alpha, beta)
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
                         'blunder': float(s.get('blunder') or 0.0)}
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
            best, pick = None, moves[0]
            for pit in moves:
                b = self.board[:]
                extra, _ = m_sow(b, p, pit)
                nxt = p if extra else 1 - p
                val = m_search(b, nxt, p, depth - 1, -10 ** 6, 10 ** 6)
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

    def to_dict(self):
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


GAMES = {
    'quixx': {'title': 'Quixx', 'min': 2, 'max': MAX_SEATS, 'engine': Quixx,
              'icon': '🎲', 'variants': VARIANTS,
              'vnames': {'standard': 'Standard', 'colors': 'Mixed Colors',
                         'numbers': 'Mixed Numbers', 'both': 'Both Mixed'},
              'blurb': 'Roll, cross off numbers left to right, lock a row '
                       'before anyone else. 2–4 players.'},
    'mancala': {'title': 'Mancala', 'min': 2, 'max': 2, 'engine': Mancala,
                'icon': '🫘', 'variants': {'standard'},
                'vnames': {'standard': 'Kalah'},
                'blurb': 'Sow stones around the board, capture from an empty '
                         'pit, finish with the fullest store. 2 players.'},
}

# ============================================================
#  Stats backend — persisted to stats.json
# ============================================================
STATS_FILE = os.path.join(BASE, 'stats.json')
STATS_VERSION = 1
ELO_START = 1200.0
ELO_K = 24.0          # rating points swapped on an even-odds win
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
    k = ELO_K / (m - 1)
    deltas = [0.0] * m
    for a in range(m):
        for b in range(a + 1, m):
            pa, pb = place[keep[a]], place[keep[b]]
            sa = 1.0 if pa < pb else (0.5 if pa == pb else 0.0)
            ea = elo_expected(recs[a]['elo'], recs[b]['elo'])
            deltas[a] += k * (sa - ea)
            deltas[b] += k * ((1.0 - sa) - (1.0 - ea))

    shared = {}
    for i in keep:
        shared[place[i]] = shared.get(place[i], 0) + 1

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
        rec['pointsAgainst'] += max(scores[j] for j in keep if j != i)
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
            'profiles': profiles_payload(), 'training': dict(TRAINING)}


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
PROFILES = {'version': 1, 'updated': None, 'profiles': {}}

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
            'adopted': 0, 'games': 0, 'started': None, 'finished': None,
            'log': []}


def birth_weights():
    w = {}
    for k in WEIGHT_KEYS:
        lo, hi = WEIGHT_BOUNDS[k]
        w[k] = round(min(hi, max(lo, DEFAULT_WEIGHTS[k]
                                 + random.gauss(0.0, BIRTH_JITTER[k]))), 3)
    return w


def new_roster():
    """Mint a fresh field of bots: random names, and blunder rates spread
    across the range so the ratings have something real to separate."""
    names = random.sample(AI_NAME_POOL, AI_ROSTER_SIZE)
    spread = [i / float(AI_ROSTER_SIZE - 1) for i in range(AI_ROSTER_SIZE)]
    random.shuffle(spread)
    profiles, order = {}, []
    for n, share in zip(names, spread):
        pid = 'ai_' + uuid.uuid4().hex[:8]
        w = birth_weights()
        profiles[pid] = {
            'id': pid, 'name': n,
            'blunder': round(0.02 + 0.33 * share, 3),
            'noise': round(0.2 + 2.6 * share, 2),
            'weights': w,
            'baseline': dict(w),          # never changes: the yardstick
            'generation': 0, 'trained': 0, 'adopted': 0, 'vsBaseline': None,
            'born': iso_now(), 'history': []}
        order.append(pid)
    return {'version': 2, 'profiles': profiles, 'order': order}


def load_profiles():
    """Read ai_profiles.json; mint a new roster if it is missing or damaged."""
    global PROFILES
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    fresh = (not isinstance(data, dict) or not isinstance(data.get('profiles'), dict)
             or not data['profiles'] or data.get('version') != 2)
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
        p.setdefault('generation', 0)
        p.setdefault('trained', 0)
        p.setdefault('adopted', 0)
        p.setdefault('vsBaseline', None)
        p.setdefault('born', iso_now())
        p.setdefault('history', [])
        p.setdefault('weights', dict(DEFAULT_WEIGHTS))
        p.setdefault('baseline', dict(p['weights']))
        for k in WEIGHT_KEYS:
            lo, hi = WEIGHT_BOUNDS[k]
            p['weights'][k] = min(hi, max(lo, float(p['weights'].get(k, DEFAULT_WEIGHTS[k]))))
            p['baseline'].setdefault(k, DEFAULT_WEIGHTS[k])
        # drop weights from an experiment that has since been removed
        for d in (p['weights'], p['baseline']):
            for k in [k for k in d if k not in WEIGHT_KEYS]:
                del d[k]
    data['version'] = 2
    PROFILES = data
    save_profiles()
    return fresh


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


def profile_seat(pid, name=None):
    """A seat dict for one AI, carrying a snapshot of its brain."""
    p = PROFILES['profiles'].get(pid)
    if not p:
        return None
    return {'id': 'ai:' + pid, 'name': name or p['name'], 'kind': 'ai',
            'profile': pid, 'weights': dict(p['weights']),
            'blunder': p['blunder'], 'noise': p['noise']}


def mutate(w):
    out = {}
    for k in WEIGHT_KEYS:
        lo, hi = WEIGHT_BOUNDS[k]
        out[k] = round(min(hi, max(lo, w[k] + random.gauss(0.0, MUTATION[k]))), 3)
    return out


def sim_seat(name, prof, weights):
    """A throwaway seat used for unrated trial games."""
    return {'id': 'sim:' + name, 'name': name, 'kind': 'ai', 'profile': prof['id'],
            'weights': dict(weights), 'blunder': prof['blunder'],
            'noise': prof['noise']}


def sim_game(seats, variant=None):
    """Play a table of AI out to the end, as fast as the CPU allows."""
    g = Quixx(variant or random.choice(sorted(VARIANTS)), seats)
    for _ in range(4000):
        if not g.bot_step():
            break
    return g


def match(prof, wa, wb, games):
    """Unrated head-to-head between two brains wearing the same bot's
    handicaps. Returns wa's score share, seats alternated so going first is
    not an advantage."""
    pts = 0.0
    for i in range(games):
        ia = i % 2
        ws = [None, None]
        ws[ia], ws[1 - ia] = wa, wb
        g = sim_game([sim_seat('S0', prof, ws[0]), sim_seat('S1', prof, ws[1])])
        ta, tb = g.total(ia), g.total(1 - ia)
        pts += 0.5 if ta == tb else (1.0 if ta > tb else 0.0)
    return pts / games


def train_round(pid):
    """One hill-climbing step: mutate, screen the mutant against the incumbent,
    and if it looks good make it prove that again on fresh games before it is
    adopted. A single 120-game screen is only about one standard error wide, so
    without the second look a coin-flip mutation gets adopted on luck often
    enough to walk a bot backwards."""
    with LOCK:
        prof = dict(PROFILES['profiles'][pid])
        base = dict(prof['weights'])
    cand = mutate(base)
    share = match(prof, cand, base, TRAIN_GAMES)           # slow, runs unlocked
    played = TRAIN_GAMES
    if share >= ADOPT_EDGE:
        again = match(prof, cand, base, CONFIRM_GAMES)
        share = ((share * TRAIN_GAMES + again * CONFIRM_GAMES)
                 / float(TRAIN_GAMES + CONFIRM_GAMES))
        played += CONFIRM_GAMES
    with LOCK:
        prof = PROFILES['profiles'][pid]
        prof['trained'] += played
        adopted = played > TRAIN_GAMES and share >= CONFIRM_EDGE
        if adopted:
            prof['weights'] = cand
            prof['generation'] += 1
            prof['adopted'] += 1
            prof['history'].insert(0, {'ts': iso_now(), 'generation': prof['generation'],
                                       'share': round(share, 3), 'weights': cand})
            del prof['history'][30:]
        save_profiles()
    return share, adopted, played


def measure_baseline(pid):
    """How a bot's current brain fares against the one it was born with."""
    with LOCK:
        prof = dict(PROFILES['profiles'][pid])
        cur, base = dict(prof['weights']), dict(prof['baseline'])
    share = match(prof, cur, base, BASELINE_GAMES)
    with LOCK:
        PROFILES['profiles'][pid]['vsBaseline'] = round(share, 3)
        save_profiles()
    return share


def ladder_game():
    """One *rated* AI-vs-AI game. Goes through the normal stats path, so the
    bots' ELOs move against each other exactly like human games do."""
    with LOCK:
        pool = PROFILES['order'][:]
    if len(pool) < 2:
        return None
    picks = random.sample(pool, random.randint(2, min(MAX_SEATS, len(pool))))
    with LOCK:
        seats = [profile_seat(p) for p in picks]
    seats = [s for s in seats if s]
    if len(seats) < 2:
        return None
    g = sim_game(seats)
    summary = g.result_summary()
    summary['selfPlay'] = True
    with LOCK:
        record_result(summary)
    return g


def run_training(rounds, ladder=4, verbose=False):
    """Work through the roster: one hill-climbing round per bot, then a few
    rated games so the ratings keep moving. Safe to run off-thread."""
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
        share, adopted, played = train_round(pid)
        for _ in range(ladder):
            ladder_game()
        line = '%s %s (%d%% in trials)' % (
            name, 'improved' if adopted else 'held its ground', round(share * 100))
        with LOCK:
            TRAINING['games'] += played + ladder
            TRAINING['adopted'] += 1 if adopted else 0
            TRAINING['log'].insert(0, line)
            del TRAINING['log'][6:]
            broadcast_stats()
        if verbose:
            print('  round %d/%d — %s' % (i + 1, rounds, line))
    with LOCK:
        order = PROFILES['order'][:]
    for pid in order:
        if not TRAINING['running']:
            break
        share = measure_baseline(pid)
        if verbose:
            print('  %s vs. the brain it was born with: %d%%'
                  % (PROFILES['profiles'][pid]['name'], round(share * 100)))


def calibrate(games=90):
    """A quick rated ladder so a brand-new roster has meaningful ratings the
    first time anyone looks at the leaderboard."""
    for _ in range(games):
        ladder_game()
    with LOCK:
        broadcast_stats()


def train_worker(rounds, ladder):
    try:
        run_training(rounds, ladder)
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
        out.append({'id': p['id'], 'name': p['name'],
                    'elos': elos, 'plays': played,
                    'elo': elos.get('quixx'), 'played': played.get('quixx', 0),
                    'generation': p['generation'], 'trained': p['trained'],
                    'adopted': p['adopted'], 'vsBaseline': p.get('vsBaseline'),
                    'weights': p['weights']})
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
        seat = profile_seat(b['profile'], b['name'])
        if seat:
            seat['id'] = b['id']
            out.append(seat)
    return out


def refresh_status(r):
    if r['status'] in ('playing', 'finished'):
        return
    seated = len(r['players']) + len(r['bots'])
    r['status'] = ('ready' if len(r['players']) >= r['humans'] and seated >= 2
                   else 'waiting')


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


def room_full(r):
    d = room_summary(r)
    d['playerIds'] = list(r['players'])
    d['hostId'] = r['host']
    d['state'] = r['engine'].to_dict() if r['engine'] else None
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
    d = room_full(r)
    for cid in list(r['players']) + list(r['spectators']):
        push(cid, 'room', d)


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
    for _ in range(500):
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
        elif u.path == '/api/events':
            self.handle_sse(u)
        elif u.path == '/favicon.ico':
            self.send_response(204)
            self.send_header('Content-Length', '0')
            self.end_headers()
        else:
            self.err('Not found', 404)

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
                q.put(('room', room_full(ROOMS[c['room']])))
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
                room = room_full(ROOMS[CLIENTS[cid]['room']])
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
            TRAINING.update({'running': True, 'round': 0, 'rounds': rounds,
                             'who': None, 'adopted': 0, 'games': 0,
                             'started': iso_now(), 'finished': None, 'log': []})
            threading.Thread(target=train_worker, args=(rounds, ladder),
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
            self.send_json({'ok': True, 'room': room_full(ROOMS[rid])})
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
                    self.send_json({'ok': True, 'room': room_full(r)})
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
                self.send_json({'ok': True, 'room': room_full(r)})
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
                self.send_json({'ok': True, 'room': room_full(r)})
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
                self.send_json({'ok': True, 'room': room_full(r)})
                return

            if action == 'start':
                if cid != r['host']:
                    self.err('Only the host can start.')
                    return
                seats = room_seats(r)
                if len(seats) < GAMES[r['game']]['min']:
                    self.err('A game needs at least 2 seats — add a player or an AI.')
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
        p['baseline'] = dict(p['weights'])      # today's brain is the new yardstick
        p['vsBaseline'] = None
    save_profiles()
    print('\n  Cleared ratings for %d bots; human records and history untouched.'
          % cleared)
    print('  Re-baselined %d brains against their current weights.' % len(roster()))
    print('  Playing calibration games...')
    TRAINING['running'] = True
    calibrate()
    TRAINING['running'] = False
    for row in profiles_payload():
        print('    %-12s elo %-8s %d rated games'
              % (row['name'], row['elo'], row['played']))
    print()


def train_cli(rounds):
    """python homegames_server.py --train [rounds] — bulk training, no server."""
    load_stats()
    if load_profiles():
        print('  Minted a new roster: %s'
              % ', '.join(p['name'] for p in roster()))
    print('\n  Training the AI by self-play: %d rounds '
          '(%d trial games each, plus rated ladder games).\n' % (rounds, TRAIN_GAMES))
    TRAINING.update({'running': True, 'rounds': rounds, 'round': 0,
                     'adopted': 0, 'games': 0, 'started': iso_now()})
    start = time.time()
    try:
        run_training(rounds, ladder=4, verbose=True)
    except KeyboardInterrupt:
        print('\n  Stopped early — progress so far is saved.')
    finally:
        TRAINING['running'] = False
        save_profiles()
    print('\n  %d rounds, %d improvements, %d games, %.0fs.'
          % (TRAINING['round'], TRAINING['adopted'], TRAINING['games'],
             time.time() - start))
    print('  Ratings after self-play:')
    for row in profiles_payload():
        print('    %-12s elo %-8s %d rated games'
              % (row['name'], row['elo'] if row['elo'] is not None else '—',
                 row['played']))
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
        train_cli(max(1, min(500, rounds)))
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
    print('  Bulk-train with:  python homegames_server.py --train 30')
    print('  Press Ctrl+C to stop.')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nHomeGames server stopped.')


if __name__ == '__main__':
    main()
