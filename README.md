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

Each game keeps its own leaderboard and ELO, so a bot can be strong at one and
weak at another. Euchre is scored as a partnership — both halves of a winning
pair are credited with the win, and ratings move against the other side.

## Requirements

- Python 3.7+ (standard library only — nothing to install)

## Run it

**Windows:** double-click `HomeGames.bat`

**Any OS:** `python homegames_server.py`

The console prints two addresses:

- `http://localhost:4001` — this computer
- `http://<your-LAN-IP>:4001` — share this with other players on your network

> If other devices can't connect, allow Python through Windows Firewall
> (private networks) when prompted, or open TCP port 4001.

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
python homegames_server.py --train 30
```

`ai_profiles.json` holds each bot's brain, the brain it was born with, its
generation count and its win rate against that original self. Delete it to
mint a fresh roster with new names.

A sanity check worth repeating after heavy training — do the ratings still
predict who wins? On a 5-bot roster, 400 games per pair, the higher-rated bot
won 10 of 10 pairings. Gaps under ~50 points are inside the noise and will
sometimes invert.

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

- Everyone starts at **1200**. K-factor is **24** once a rating has settled,
  but **48** for a player's first 30 games, shown as `?` on the leaderboard.
  Without that, ratings across the two populations never meet: the bots play
  thousands of games against each other while a person plays a handful a week,
  so a human's rating crawls while the bot pool drifts. Each side moving at its
  own K means a game is no longer strictly zero-sum — the usual trade for
  letting a new rating find its level in a few games instead of a hundred.
- **A bot's rating is only comparable to yours if you have actually played it.**
  Self-play keeps the bots ranked correctly against *each other*, but the pool's
  level against humans is anchored only by the games you play against them.
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
