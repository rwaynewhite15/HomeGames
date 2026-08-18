# HomeGames

A LAN game lobby you run at home. One person starts the server; everyone on the
network opens it in a browser to host, join, or spectate games.

You land in the **hub**: pick a game, and see who else is online and what they
are doing. Tap anyone — person or bot — for their profile: rating, rank,
win/loss record, best and average score, streaks, results by variant, and their
recent games with the ELO swing on each. Choosing a game drops you into that
game's own lobby, where you host, join, or spectate.

Games so far:

- **Quixx** — 2–4 players, four variants (Standard, Mixed Colors, Mixed
  Numbers, Both Mixed).
- **Mancala** — 2 players, standard Kalah. Ported from the `mancala` project;
  the rules were checked move-for-move against the original over 400 games.
- **Euchre** — 4 players in fixed partnerships, two variants (Standard, Stick
  the Dealer). Both bowers, ordering up, going alone, and the full 1/2/4-point
  scoring to ten. The first game here with hidden information: hands are dealt
  server-side and each client is sent only its own cards, so opponents and
  spectators genuinely cannot see them.
- **Chess** — 2 players. Seat 0 plays White, so a solo player is White against
  the AI. The only game needing an install (`pip install chess`), and the only
  one whose bots learn while you play them rather than only in training: each
  keeps an opening book on the person it is facing. See
  [How the AI learn](#how-the-ai-learn).

Each game keeps its own leaderboard and ELO, so a bot can be strong at one and
weak at another. Euchre is scored as a partnership — both halves of a winning
pair are credited with the win, and ratings move against the other side.

## Requirements

- Python 3.7+ — Quixx, Mancala and Euchre need the standard library and
  nothing else.
- Chess additionally needs [python-chess](https://python-chess.readthedocs.io):
  `pip install -r requirements.txt`. It is pure Python, so it installs on the
  Pi without a compiler. Skip it and everything else still works — the server
  starts normally, chess just does not appear in the lobby and the banner
  says why.

## Run it

**Windows:** double-click `HomeGames.bat`

**Any OS:** `python homegames_server.py`

The console prints two addresses:

- `http://localhost:4001` — this computer
- `http://<your-LAN-IP>:4001` — share this with other players on your network

> If other devices can't connect, allow Python through Windows Firewall
> (private networks) when prompted, or open TCP port 4001.

## Running it on the Pi

`deploy/install.sh` sets the Pi up to run HomeGames as a service and to pick up
new code from `main` by itself. Run it from the repo, as the user who should own
the server — not as root:

```
cd ~/HomeGames && ./deploy/install.sh
```

That installs:

- **`homegames.service`** — the server, started at boot, restarted if it
  crashes. It runs out of a virtualenv at `.venv/`, because Raspberry Pi OS
  (Bookworm and later) marks the system interpreter externally-managed under
  PEP 668 and `pip install chess` against it fails outright.
- **`homegames-deploy.timer`** — every two minutes, fetches `main`; if there is
  a new commit it fast-forwards, reinstalls dependencies *if `requirements.txt`
  changed*, and restarts the server. Polling rather than a webhook on purpose:
  a webhook means opening a port through your router to the internet, which is
  a much bigger thing to own than a `git fetch`.
- a sudoers rule letting that one user run exactly one command,
  `systemctl restart homegames.service`.

**Deploys wait for your game to finish.** The server keeps `CLIENTS` and
`ROOMS` in memory only — `stats.json`, `ai_profiles.json` and `chess_brains/`
are written as games *finish*, so results and AI learning survive a restart,
but any game still in progress does not. So the deploy asks `/api/health` first
and steps aside while `playing` is non-zero, for up to about half an hour
(15 ticks) before going ahead regardless, so one abandoned room cannot block
deploys forever.

It also refuses to pull over uncommitted changes in the Pi's working tree, and
refuses anything that will not fast-forward — both are cases where a human
should look rather than an unattended script guessing.

```
sudo systemctl status homegames      # is it up?
journalctl -u homegames -f           # the server's own log
journalctl -u homegames-deploy -f    # what the deploys have been doing
./deploy/deploy.sh                   # deploy right now, don't wait for the timer
```

## How it works

- `homegames_server.py` — server: HTTP + Server-Sent Events lobby, rooms
  (host / join / spectate), and a server-authoritative Quixx engine so all
  clients stay in sync and nobody can mis-mark.
  Presence is derived from open event streams rather than a timeout, so
  "who's here" is accurate; `SSE_PING` sets how quickly a closed tab drops off
  (a disconnect is only noticed when a keepalive write fails).
- `index.html` — the whole client app (lobby + game UI), served by the server.
- `stats.json` — the stats backend: lifetime records and ELO ratings, written
  by the server after every finished game.
- `ai_profiles.json` — the AI roster: each bot's name, handicaps, learned
  weights and training history.
- `chess_ai.py` — the learning chess opponent. Standalone: it imports nothing
  from the server and runs on its own for testing or a terminal game.
- `chess_brains/` — one JSON brain per bot: its opening book, evaluation
  weights, record and what it has noticed about each person it plays.

## Players and AI

A table seats **2 to 4**, in any mix of people and AI. The host picks the mix
**before hosting** (in the lobby) and can still change it **after** — the
waiting room has the same controls until the game starts. Removing a human seat
below the number already joined is refused; everything else is live, and other
players see the table update immediately.

**There are no difficulty labels.** The AI are just named players — Rusty,
Pepper, Marbles — and the **ELO beside each name is the difficulty**. Pick a
1050 bot for a gentle game or a 1350 bot for a hard one. A roster of five is
minted on first run with random names, and the server plays a quick
calibration ladder so their ratings mean something before you ever look.

AI seats take their turns the moment it is their move, so play always pauses
where a human has a decision. The move log under the dice tray shows what
everyone just did.

## How the AI learn

Each bot scores every possible mark as points gained minus the cells that mark
gives up, using five weights that **self-play tunes**: `skip_cost`,
`late_skip`, `lock_bonus`, `penalty_fear` and `threshold`. Training is hill
climbing — mutate the weights, play the mutant against the incumbent, keep it
only if it wins. A promising candidate has to prove itself twice, on fresh
games, before it is adopted; a single 120-game screen is only about one
standard error wide, so without that second look coin-flip mutations get
adopted on luck and walk a bot backwards.

Two things are fixed at birth and never learned: `blunder` (the share of
decisions taken at random) and `noise` (jitter on each judgement). That is
deliberate. Training tunes the weights, so a handicap the weights could
compensate for would let every bot converge on the same strength and flatten
the ratings into noise — which is exactly what happened in an early build that
used noise alone: 84 rounds of self-play turned the field into a coin flip,
because good weights are simply robust to noise. A blunder rate cannot be
learned around.

Between training rounds the bots play **rated** games against each other, so
their ELOs develop on their own. Those games count towards ratings and records
but are kept out of the recent-games list, so training cannot bury the games
people actually played.

Train them from the lobby's **The AI** panel, or in bulk from the shell:

```
python homegames_server.py --train 30            # quixx
python homegames_server.py --train 30 mancala
python homegames_server.py --train 20 chess      # a different mechanism, see below
```

Chess trains from the same button but not by the same method. The hill-climbing
trainer screens on 120 games and confirms on 160, which is milliseconds a game
at Quixx and minutes a game at chess, so chess bots train through their own
ChessAI instead: searchers self-play to tune their evaluation, learners spar
against the sharpest searcher on the roster. Rounds are shorter as a result —
20 games a bot — and rated games between chess bots run alongside so the ELOs
keep moving. A bot that is sitting in a live game is skipped for that round,
because training and a finished game both write the same brain file and the
loser of that race would lose its learning.

`ai_profiles.json` holds each bot's brain, the brain it was born with, its
generation count and its win rate against that original self. Delete it to
mint a fresh roster with new names.

A sanity check worth repeating after heavy training — do the ratings still
predict who wins? On a 5-bot roster, 400 games per pair, the higher-rated bot
won 10 of 10 pairings. Gaps under ~50 points are inside the noise and will
sometimes invert.

### Chess learns differently

The trainer above does not tune chess. A screening round is 120 games and then
160 to confirm, which is milliseconds a game at Quixx and minutes a game at
chess — and a bot trained only against itself would never notice the person it
is actually playing. So chess bots learn *from your games instead*, in
`chess_ai.py`, on three clocks:

1. **The opening book** — pays off within a handful of games. Early positions
   repeat, you have a few pet openings, and the book records how each reply
   actually turned out against *you*. Move choice is UCB1 rather than
   epsilon-greedy, because with thirty games of data you want exploration
   spent on the moves it is uncertain about, not a fixed share of turns thrown
   at moves already known to be bad.
2. **Move-level credit** — pays off over tens of games. Each of the AI's moves
   is scored by what the position did two plies later, so one game is ~40
   training signals rather than the single bit a win/loss gives you.
3. **Evaluation weights** — pays off over hundreds. The piece values and
   positional terms are tuned by TD(0) on that signal. Ten human games move
   them by a rounding error; that is honest rather than disappointing, and it
   is what `self_play()` is for.

So the strength you feel across your first dozen games is mostly search depth
(which ramps from 2 up to each bot's ceiling) and the book. Each bot's
`blunder` handicap still applies and still caps its ceiling, exactly as it does
elsewhere — a bot that throws away a fifth of its moves is beatable no matter
how good its book gets.

### Two kinds of chess bot

Each bot carries a **style** alongside the `blunder` and `noise` it is born
with:

- **searcher** — alpha-beta with quiescence, 2–4 ply. This is what makes the
  default opponent respectable, and it is doing far more work than the learning
  is. Measured: the same bot with quiescence removed, at identical depth and
  with the identical evaluation, loses 12–0.
- **learner** — no search whatsoever. It scores each legal move on a dozen
  readable facts (what it takes, what it leaves hanging, whether the square is
  defended, development, centralisation) and samples from a softmax over those
  scores. The weights start at **zero** — nothing is hand-tuned, so anything it
  ends up believing, it worked out. Learning is REINFORCE: shift the weights
  toward what made the played move different from the alternatives it was
  picked over, scaled by how the game went.

Learners exist to be measured. They join the roster at the default rating and
their ELO against the searchers is the only honest answer to "is the learning
working". Existing bots keep their style and their ratings — a rating earned by
searching would mean nothing after a silent conversion.

**Bootstrap a learner against a searcher, not against itself.** This matters
more than it sounds. Over ~2,400 measured decisions, a move that hangs a pawn
or more scores −0.070 against another learner versus −0.028 for a safe move — a
gap of 0.098. Against a searcher the same gap is 0.297, three times larger. An
opponent with no lookahead usually fails to take what you left hanging, so a
blind player cannot teach itself not to blunder. It needs an opponent that
punishes it.

```
python chess_ai.py --spar 300      # learner vs a searcher; the right bootstrap
```

The difference that makes, from zero weights, same code, only the opponent
changed:

| feature   | 400 games of self-play | 300 games against a searcher |
|-----------|------------------------|------------------------------|
| `hanging` | **+0.022** (wrong sign)| **-0.490** (correct, largest)|
| `capture` | +0.718                 | +0.302                       |
| `centre`  | +0.343                 | +0.257                       |
| `develop` | +0.063                 | +0.209                       |

Taught by something that punishes blunders, it works out for itself that
leaving a piece takeable is the worst thing it can do. Taught by another blind
bot, it concludes the opposite.

That sparring run went 0W-297L-3D against a depth-2 searcher, which is the
honest measure of where a learner starts. Its ceiling is structural: it cannot
see a recapture, so it plays very well-informed one-move chess and will not
beat the searchers. How close it gets is the interesting question, and the
leaderboard answers it.

Two features stay near zero and that is expected. `defended` is redundant —
`hanging` already scales its risk down when the square is covered, so there is
no independent signal left for it. `promote` needs endgames a learner rarely
reaches while it is still losing almost every game.

### Chess ratings are measured

Everywhere else the ELO beside a bot's name is earned in a closed pool, which
settles who is stronger without ever establishing what any of them is worth.
Chess is the one game here with an outside yardstick, so its bots carry a
rating that was measured rather than accumulated:

| bot | blunder | depth | ELO | how |
|-----|---------|-------|-----|-----|
| 1st | 0.020 | 4 | **1381** | 20-13-7 vs Stockfish 16 at 1320 |
| 2nd | 0.103 | 4 | **1250** | 9-15-6 vs Stockfish 16 at 1320 |
| 3rd | 0.185 | 3 | **850** | 2-37-1 vs Stockfish 16 at 1320 |
| 4th | 0.268 | 3 | **538** | placed against a 1-ply reference |
| 5th | 0.350 | 2 | **360** | placed against a 1-ply reference |
| learner | 0.020 | — | **0** | 0.512 vs a random mover |

Bots seed at these values instead of 1200, and the searchers' centre of mass is
held there — every bot shifts by the same amount, so rankings and the spread
they earned survive, but the field as a whole cannot wander off the scale. Your
own rating is never touched by any of it. Re-measure with
`python tools/chess_calibration.py` if the search, the evaluation or
`depth_ceiling()` changes.

**A learner starts at zero because it really is a random mover.** Not
approximately: `DEFAULT_POLICY` is all zeros, so every legal move scores zero
and the softmax it samples from is uniform. Measured at 0.512 against a random
mover, which is the same statement. That is a seed and not a verdict — learners
are never pinned, so a learner sparring its way up off the floor keeps every
point it takes, and that climb is the whole measurement.

**The bots' results against each other do not predict their results against
you.** The top two searchers share a depth ceiling and differ only in blunder
rate, and that pairing finished **16-0** — while against Stockfish the same two
are 131 points apart, which predicts 0.32. Two near-identical engines make the
blunder rate the only variable in the game, so it decides nearly every one. A
person is a *different* opponent, and against a different opponent the gap is
much smaller. The measured numbers describe the game you will actually get.

Which is also why the difficulty ladder is steeper than it looks. A blunder
costs a few points at Quixx and costs the game at chess, so spreading
`blunder` from 0.02 to 0.35 across five bots does not produce five difficulty
levels — it produces two playable opponents and three that hang pieces. Handing
Stockfish itself a 20% random-move rate cost it roughly 600 points.

Two calibration methods failed before one worked, and both are recorded in
`tools/chess_calibration.py` because both look obviously correct until measured.
Chaining rung to rung saturates, for the reason above. Handicapping Stockfish
with random moves makes it erratic rather than weak — brilliant play punctuated
by free pieces — and nothing rates consistently against it: two rungs with
known ratings disagreed by 476 points about the same reference. Depth-limited
Stockfish is properly transitive but cannot be made weak enough, since even a
1-ply search rates about 1230 on its evaluation alone. What works is a
hand-built deterministic reference, rated against a rung Stockfish has already
placed.

Brains live one file per bot in `chess_brains/`. Back up the folder, or reset
one (or all) from `POST /api/chess/reset`. `chess_ai.py` also runs standalone:

```
python chess_ai.py --test          # self-test
python chess_ai.py --play          # play it at the terminal
python chess_ai.py --selfplay 200  # bootstrap a searcher's evaluation weights
python chess_ai.py --spar 300      # bootstrap a learner against a searcher
python chess_ai.py --stats         # what it has learned so far
```

## Stats & ELO

Every finished game is folded into `stats.json` and shown on the lobby
leaderboard. Players are keyed by **name** (case-insensitive) — client ids only
last a session, so the name is what carries across visits. Use the same name to
keep building a rating. **Each AI is a player too**, carrying its own ELO, so
beating a highly rated bot is worth more than beating a weak one.

Per player, per game: ELO, games played, W-L-T, forfeits, best and average
score, average finishing place, total penalties, current/best win streak, and a
per-variant breakdown. The file also keeps the last 100 results with the ELO
change for each.

- Everyone starts at **1200**, except chess bots, which start at the rating
  their configuration was **measured** at — see [Chess ratings are
  measured](#chess-ratings-are-measured). K-factor is **24** once a rating has settled,
  but **48** for a player's first 30 games, shown as `?` on the leaderboard.
  Without that, ratings across the two populations never meet: the bots play
  thousands of games against each other while a person plays a handful a week,
  so a human's rating crawls while the bot pool drifts. Each side moving at its
  own K means a game is no longer strictly zero-sum — the usual trade for
  letting a new rating find its level in a few games instead of a hundred.
- **Outside chess, a bot's rating is only comparable to yours if you have
  actually played it.** Self-play keeps the bots ranked correctly against *each
  other*, but the pool's level against humans is anchored only by the games you
  play against them. Chess is the exception: its bots are measured against an
  outside engine, so their numbers mean something before you sit down.
- With 3–4 seats, ratings are a **pairwise round robin** — every seat is scored
  against every other and the K-factor is split between those matchups. Ratings
  stay zero-sum, and a 2-player game behaves exactly as it always did.
- Places are by final score, with a tie sharing a place.
- Leaving a game in progress ends it, and the quitter places last regardless of
  the score on their sheet.
- A name can only be rated once per game, so if the same bot sits at a table
  twice, only the better finish counts — you cannot play yourself. The second
  seat is numbered (`Rusty 2`) and rates separately.

To wipe the history, stop the server and delete `stats.json` (or reset it to
`{"version": 1, "players": {}, "history": []}`) — it is recreated on startup.
`POST /api/stats` returns the leaderboard and recent results as JSON.

## Adding more games

Game engines register in the `GAMES` dict in `homegames_server.py`
(`title`, `icon`, `blurb`, `variants`, `min`/`max` players, `engine` class). A
new entry appears on the hub automatically with its own lobby, room list,
leaderboard and ELO.

An engine needs:

| Member | Purpose |
|---|---|
| `__init__(variant, seats)` | seats are `{id, name, kind, profile, …}` |
| `apply(cid, body)` | handle one action, return an error string or `None` |
| `to_dict(viewer=None)` | state to broadcast; `kind` selects the client renderer. `viewer` is the recipient's `cid`, so a game with hidden information can build a different payload per seat — `None` means a spectator |
| `bot_step()` | play one pending AI move, `False` when waiting on a human |
| `result_summary()` | final scores, so the game lands on the leaderboard |
| `forfeit(cid)`, `over`, `total(i)` | shared lifecycle |

AI seats carry a `blunder` rate. Quixx turns it into random moves alongside
learned weights; Mancala turns it into search depth; Euchre jitters the
hand-strength estimate a bot bids on and occasionally picks a random legal
card — so a bot plays at roughly its usual relative strength in any of them.

`broadcast_room` builds the payload once per recipient rather than once per
room, which is what lets Euchre keep hands secret. A partnership game also sets
`teams: True` in its `GAMES` entry and puts a `team` index on each player in
`result_summary()`; the trainer then seats one brain per side for its 2v2
head-to-head, and the stats layer counts a pair as a single side when working
out who won. An engine needs a constructor
`(variant, seats)` — where each seat is `{id, name, kind, level}` — plus action
methods and `to_dict()` for broadcasting state. The client renders by
`state.kind`. To support AI seats, add a `bot_step()` that plays one pending AI
action and returns `False` when the table is waiting on a human; to appear on
the leaderboard, add a `result_summary()`. Seats carry a `weights` dict plus
`blunder`/`noise`, so an engine that reads those gets self-play training for
free.
