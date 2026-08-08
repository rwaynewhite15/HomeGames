# HomeGames

A LAN game lobby you run at home. One person starts the server; everyone on the
network opens it in a browser to host, join, or spectate games.

First game: **Quixx** (2–4 players, humans and/or AI) with four variants —
Standard, Mixed Colors, Mixed Numbers, and Both Mixed.

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
- `index.html` — the whole client app (lobby + game UI), served by the server.
- `stats.json` — the stats backend: lifetime records and ELO ratings, written
  by the server after every finished game.

## Players and AI

A table seats **2 to 4**, in any mix of people and AI. The host picks the mix
**before hosting** (in the lobby) and can still change it **after** — the
waiting room has the same controls until the game starts. Removing a human seat
below the number already joined is refused; everything else is live, and other
players see the table update immediately.

Three AI strengths, each with its own personality and its own rating:

| Bot | Difficulty | How it plays |
|---|---|---|
| Rookie Bot | Easy | Marks almost anything it can reach, and picks near-randomly |
| Sharp Bot | Medium | Weighs the points gained against the cells given up |
| Ace Bot | Hard | Same, but pickier early and hunts locks; no randomness |

Head to head over 500 games each: Ace beats Rookie 475–23, Ace beats Sharp
329–167, Sharp beats Rookie 445–54.

AI seats take their turns the moment it is their move, so play always pauses
where a human has a decision. The move log under the dice tray shows what
everyone just did.

## Stats & ELO

Every finished game is folded into `stats.json` and shown on the lobby
leaderboard. Players are keyed by **name** (case-insensitive) — client ids only
last a session, so the name is what carries across visits. Use the same name to
keep building a rating. **Each AI difficulty is a player too**: Rookie, Sharp
and Ace each carry their own ELO, so beating an Ace Bot is worth more than
beating a Rookie.

Per player, per game: ELO, games played, W-L-T, forfeits, best and average
score, average finishing place, total penalties, current/best win streak, and a
per-variant breakdown. The file also keeps the last 100 results with the ELO
change for each.

- Everyone starts at **1200**; K-factor is **24**, so an even 2-player matchup
  moves ±12 and an upset moves more.
- With 3–4 seats, ratings are a **pairwise round robin** — every seat is scored
  against every other and the K-factor is split between those matchups. Ratings
  stay zero-sum, and a 2-player game behaves exactly as it always did.
- Places are by final score, with a tie sharing a place.
- Leaving a game in progress ends it, and the quitter places last regardless of
  the score on their sheet.
- A name can only be rated once per game, so if two identical bots sit at the
  same table, only the better finish counts — you cannot play yourself. Bots of
  the same difficulty are numbered (`Sharp Bot 2`) and rate separately.

To wipe the history, stop the server and delete `stats.json` (or reset it to
`{"version": 1, "players": {}, "history": []}`) — it is recreated on startup.
`POST /api/stats` returns the leaderboard and recent results as JSON.

## Adding more games

Game engines register in the `GAMES` dict in `homegames_server.py`
(`title`, `min`/`max` players, `engine` class). An engine needs a constructor
`(variant, seats)` — where each seat is `{id, name, kind, level}` — plus action
methods and `to_dict()` for broadcasting state. The client renders by
`state.kind`. To support AI seats, add a `bot_step()` that plays one pending AI
action and returns `False` when the table is waiting on a human; to appear on
the leaderboard, add a `result_summary()`.
