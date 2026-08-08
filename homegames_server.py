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
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue, Empty
from urllib.parse import urlparse, parse_qs

PORT = 4001
BASE = os.path.dirname(os.path.abspath(__file__))

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

# AI opponents. Each difficulty plays under its own name, so each one builds
# its own ELO on the leaderboard alongside the humans.
#   skip_cost — points it charges itself per cell given up by marking further right
#   threshold — how good a mark has to look before it bothers
#   noise     — randomness added to every candidate; high noise = careless
AI_LEVELS = {
    'easy':   {'title': 'Rookie Bot', 'tag': 'Easy',
               'skip_cost': 0.3, 'threshold': -99.0, 'noise': 5.0},
    'medium': {'title': 'Sharp Bot', 'tag': 'Medium',
               'skip_cost': 0.9, 'threshold': -1.5, 'noise': 1.0},
    'hard':   {'title': 'Ace Bot', 'tag': 'Hard',
               'skip_cost': 1.2, 'threshold': -0.5, 'noise': 0.0},
}
AI_ORDER = ['easy', 'medium', 'hard']


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
        """seats: 2–4 dicts of id / name / kind ('human'|'ai') / level."""
        self.variant = variant
        self.rows = build_rows(variant)
        self.players = [{'cid': s['id'], 'name': s['name'],
                         'kind': s.get('kind', 'human'), 'level': s.get('level'),
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
    def bot_value(self, p, r, i, bonus):
        """What an AI thinks marking (r, i) is worth, in points."""
        cfg = AI_LEVELS[self.players[p]['level']]
        skipped = i - self.last_marked(p, r) - 1
        c = self.count(p, r)
        gain = POINTS[c + 1] - POINTS[c]
        if i == 10:
            gain += POINTS[min(c + 2, 12)] - POINTS[c + 1]   # locking scores twice
        v = gain - cfg['skip_cost'] * skipped + bonus
        if cfg['noise']:
            v += random.uniform(-cfg['noise'], cfg['noise'])
        return v

    def bot_pick(self, p, mode, bonus):
        best, cell = None, None
        for (r, i) in self.options(p, mode):
            v = self.bot_value(p, r, i, bonus)
            if best is None or v > best:
                best, cell = v, (r, i)
        return cell, best

    def bot_take(self, p, mode, bonus):
        cell, value = self.bot_pick(p, mode, bonus)
        if cell is not None and value >= AI_LEVELS[self.players[p]['level']]['threshold']:
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
                    return self.bot_take(i, 'white', 0.0)
            return False
        if self.phase == 'color':
            if self.players[self.active]['kind'] != 'ai':
                return False
            # an empty turn costs 5, so any mark is worth that much more
            return self.bot_take(self.active, 'color',
                                 0.0 if self.active_marked else 5.0)
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
                         'level': p['level'],
                         'forfeited': i == self.forfeit_idx}
                        for i, p in enumerate(self.players)],
        }

    def to_dict(self):
        return {
            'kind': 'quixx',
            'variant': self.variant,
            'rows': self.rows,
            'players': [{'cid': p['cid'], 'name': p['name'], 'marks': p['marks'],
                         'kind': p['kind'], 'level': p['level'],
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


GAMES = {'quixx': {'title': 'Quixx', 'min': 2, 'max': MAX_SEATS, 'engine': Quixx}}

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
    return {'name': name, 'ai': False, 'level': None, 'firstSeen': iso_now(),
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
        pl['level'] = ps[i].get('level')
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
                        'level': ps[i].get('level')})

    entries.sort(key=lambda e: e['place'])
    STATS['history'].insert(0, {
        'ts': iso_now(), 'game': game, 'variant': variant, 'seats': len(ps),
        'reason': summary.get('reason'), 'players': entries})
    del STATS['history'][HISTORY_MAX:]
    save_stats()


def leaderboard(game, limit=25):
    rows = []
    for key, pl in STATS['players'].items():
        rec = (pl.get('games') or {}).get(game)
        if not rec or not rec.get('played'):
            continue
        rec = normalize_record(rec)
        rows.append({'name': pl.get('name') or key,
                     'ai': bool(pl.get('ai')), 'level': pl.get('level'),
                     'elo': round(rec['elo'], 1), 'played': rec['played'],
                     'wins': rec['wins'], 'losses': rec['losses'],
                     'ties': rec['ties'], 'best': rec['bestScore'],
                     'avg': round(rec['pointsFor'] / rec['played'], 1),
                     'avgPlace': (round(rec['placeSum'] / rec['played'], 2)
                                  if rec['placeSum'] else None),
                     'streak': rec['streak'], 'lastPlayed': pl.get('lastPlayed')})
    rows.sort(key=lambda r: (-r['elo'], -r['wins'], r['name'].lower()))
    return rows[:limit]


def stats_payload():
    return {'leaderboard': {g: leaderboard(g) for g in GAMES},
            'recent': STATS['history'][:10]}


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
#  Lobby / rooms
# ============================================================

def make_bots(levels):
    """Turn a list of difficulties into seated AI players. A repeated level is
    numbered, so each bot at the table is a distinct name (and rating)."""
    out, counts = [], {}
    for lv in levels:
        counts[lv] = counts.get(lv, 0) + 1
        title = AI_LEVELS[lv]['title']
        if counts[lv] > 1:
            title += ' %d' % counts[lv]
        out.append({'id': 'ai:%s:%d' % (lv, counts[lv]), 'level': lv, 'name': title})
    return out


def room_seats(r):
    """Turn order: humans in the order they joined, then the AI seats."""
    out = [{'id': c, 'name': CLIENTS[c]['name'], 'kind': 'human'}
           for c in r['players'] if c in CLIENTS]
    out += [{'id': b['id'], 'name': b['name'], 'kind': 'ai', 'level': b['level']}
            for b in r['bots']]
    return out


def refresh_status(r):
    if r['status'] in ('playing', 'finished'):
        return
    seated = len(r['players']) + len(r['bots'])
    r['status'] = ('ready' if len(r['players']) >= r['humans'] and seated >= 2
                   else 'waiting')


def parse_table(body, joined):
    """Validate a humans + AI table setup. Raises ValueError with a message."""
    try:
        humans = int(body.get('humans', 2))
    except (TypeError, ValueError):
        raise ValueError('Bad number of human players.')
    levels = body.get('bots')
    if levels is None:
        levels = []
    if not isinstance(levels, list) or len(levels) > MAX_SEATS:
        raise ValueError('Bad AI list.')
    levels = [str(x) for x in levels]
    for lv in levels:
        if lv not in AI_LEVELS:
            raise ValueError('Unknown AI difficulty.')
    if humans < 1 or humans > MAX_SEATS:
        raise ValueError('Humans must be between 1 and %d.' % MAX_SEATS)
    if humans < joined:
        raise ValueError('%d players have already joined.' % joined)
    if humans + len(levels) > MAX_SEATS:
        raise ValueError('That is more than %d seats.' % MAX_SEATS)
    if humans + len(levels) < 2:
        raise ValueError('A game needs at least 2 seats.')
    return humans, levels


def room_summary(r):
    return {'id': r['id'], 'name': r['name'], 'game': r['game'],
            'gameTitle': GAMES[r['game']]['title'], 'variant': r['variant'],
            'status': r['status'], 'humans': r['humans'], 'maxSeats': MAX_SEATS,
            'bots': [{'level': b['level'], 'name': b['name']} for b in r['bots']],
            'players': [CLIENTS[c]['name'] for c in r['players'] if c in CLIENTS],
            'spectators': len(r['spectators']),
            'host': CLIENTS.get(r['host'], {}).get('name', '?')}


def room_full(r):
    d = room_summary(r)
    d['playerIds'] = list(r['players'])
    d['hostId'] = r['host']
    d['state'] = r['engine'].to_dict() if r['engine'] else None
    return d


def push(cid, event, data):
    c = CLIENTS.get(cid)
    if not c:
        return
    for q in list(c['queues']):
        q.put((event, data))


def broadcast_lobby():
    data = {'rooms': [room_summary(r) for r in ROOMS.values()]}
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
            q.put(('lobby', {'rooms': [room_summary(r) for r in ROOMS.values()]}))
            q.put(('stats', stats_payload()))
            if c['room'] and c['room'] in ROOMS:
                q.put(('room', room_full(ROOMS[c['room']])))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                try:
                    event, data = q.get(timeout=15)
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
                                'seen': time.time()}
            room = None
            if CLIENTS[cid]['room'] and CLIENTS[cid]['room'] in ROOMS:
                room = room_full(ROOMS[CLIENTS[cid]['room']])
            broadcast_lobby()
            self.send_json({'ok': True, 'cid': cid, 'name': name,
                            'games': {k: {'title': v['title']} for k, v in GAMES.items()},
                            'room': room,
                            'rooms': [room_summary(r) for r in ROOMS.values()],
                            'stats': stats_payload()})
            return

        if cmd == 'stats':
            self.send_json({'ok': True, 'stats': stats_payload()})
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
            if variant not in VARIANTS:
                self.err('Unknown variant.')
                return
            try:
                humans, levels = parse_table(body, 1)
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
                    humans, levels = parse_table(body, len(r['players']))
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
                g = r['engine']
                t = str(body.get('type') or '')
                if t == 'roll':
                    e = g.roll(cid)
                elif t == 'mark':
                    e = g.mark(cid, body.get('row'), body.get('idx'))
                elif t == 'passWhite':
                    e = g.pass_white(cid)
                elif t == 'skipColor':
                    e = g.skip_color(cid)
                else:
                    e = 'Unknown action.'
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


def main():
    load_stats()
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
    print('  Press Ctrl+C to stop.')
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nHomeGames server stopped.')


if __name__ == '__main__':
    main()
